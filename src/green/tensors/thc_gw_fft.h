#ifndef GREEN_TENSORS_THC_GW_FFT_H
#define GREEN_TENSORS_THC_GW_FFT_H

#include <green/integrals/thc_factor_data.h>
#include <green/ndarray/ndarray.h>
#include <unsupported/Eigen/FFT>
#include <Eigen/LU>
#include <array>
#include <iostream>
#include <limits>
#include <green/tensors/thc_host_fft.h>

namespace green::tensors {
  using thc_matrix=integrals::thc_factor_data::matrix;
  using thc_complex=std::complex<double>;
  using thc_tensor4=ndarray::ndarray<thc_complex,4>;
  using thc_tensor5=ndarray::ndarray<thc_complex,5>;

  /** Full Cartesian mesh embedding. A shifted k mesh is a coset of the q
   * group. Actual Bloch X/G values retain its phases; only integer indices
   * enter the FFT. No transform over an IBZ list is permitted.
   */
  class thc_momentum_fft {
    std::array<size_t,3> _shape{},_stride{};
    std::vector<size_t> _kmap,_qmap,_negative;
    size_t _nk;
    mutable std::vector<thc_complex> _buffer;
    mutable thc_host_fft _host_fft;
    mutable size_t _fft_calls=0;
    static double periodic(double x) {
      x-=std::floor(x);return std::abs(x-1)<1e-9 || std::abs(x)<1e-9?0:x;
    }
    void transform(std::vector<thc_matrix>& field,bool inverse)const {
      if(_nk>1)++_fft_calls;
      if(thc_host_fft::enabled()){
        const size_t entries=field[0].size();_buffer.resize(_nk*entries);
        for(size_t k=0;k<_nk;++k)std::copy_n(field[k].data(),entries,_buffer.data()+k*entries);
        _host_fft.transform(_buffer.data(),_shape,entries,inverse);
        for(size_t k=0;k<_nk;++k){
          std::copy_n(_buffer.data()+k*entries,entries,field[k].data());
          if(inverse)field[k]/=double(_nk);
        }
        return;
      }
      Eigen::FFT<double> fft;
      for(size_t axis=0;axis<3;++axis) {
        const size_t count=_shape[axis],stride=_stride[axis];if(count==1)continue;
        std::vector<thc_complex> in(count),out(count);
        for(size_t base=0;base<_nk;++base) {
          if((base/stride)%count)continue;
          for(Eigen::Index element=0;element<field[0].size();++element) {
            for(size_t i=0;i<count;++i)in[i]=field[base+i*stride].data()[element];
            if(inverse)fft.inv(out,in);else fft.fwd(out,in);
            for(size_t i=0;i<count;++i)field[base+i*stride].data()[element]=out[i];
          }
        }
      }
    }
  public:
    ~thc_momentum_fft()=default;
    thc_momentum_fft(const thc_momentum_fft&)=delete;
    thc_momentum_fft& operator=(const thc_momentum_fft&)=delete;
    const std::array<size_t,3>& shape()const{return _shape;}
    const std::vector<size_t>& k_map()const{return _kmap;}
    const std::vector<size_t>& q_map()const{return _qmap;}
    const std::vector<size_t>& negative_map()const{return _negative;}
    size_t fft_calls()const{return _fft_calls;}
    static const char* cpu_backend(){return thc_host_fft::backend();}
    explicit thc_momentum_fft(const integrals::thc_factor_data& factors):
      thc_momentum_fft(factors.kmesh_scaled(),factors.qmesh_scaled(),factors.nk()) {}
    thc_momentum_fft(const std::vector<double>& k,const std::vector<double>& q,size_t nk):_nk(nk) {
      if(!nk || k.size()!=3*nk || q.size()!=3*nk)throw std::runtime_error("THC FFT requires a complete closed regular k/q mesh");
      std::array<double,3> origin{};
      for(size_t a=0;a<3;++a) {
        std::vector<double> axis;for(size_t i=0;i<_nk;++i)axis.push_back(periodic(k[3*i+a]));
        std::sort(axis.begin(),axis.end());
        axis.erase(std::unique(axis.begin(),axis.end(),[](double x,double y){return std::abs(x-y)<1e-8;}),axis.end());
        _shape[a]=axis.size();origin[a]=axis.front();
      }
      if(_shape[0]*_shape[1]*_shape[2]!=_nk)throw std::runtime_error("THC FFT requires a Cartesian mesh, not an IBZ/arbitrary list");
      _stride={_shape[1]*_shape[2],_shape[2],1};
      std::vector<bool> seen_k(_nk,false),seen_q(_nk,false);
      for(size_t i=0;i<_nk;++i) {
        size_t ik=0,iq=0;
        for(size_t a=0;a<3;++a) {
          double x=(periodic(k[3*i+a])-origin[a])*_shape[a],y=periodic(q[3*i+a])*_shape[a];
          if(std::abs(x-std::round(x))>1e-7 || std::abs(y-std::round(y))>1e-7)
            throw std::runtime_error("THC FFT mesh is not uniformly commensurate");
          ik+=(size_t(std::llround(x))%_shape[a])*_stride[a];
          iq+=(size_t(std::llround(y))%_shape[a])*_stride[a];
        }
        if(seen_k[ik] || seen_q[iq])throw std::runtime_error("THC FFT embedding has repeated points");
        seen_k[ik]=seen_q[iq]=true;_kmap.push_back(ik);_qmap.push_back(iq);
      }
      for(size_t i=0;i<_nk;++i) {
        size_t minus=0;
        for(size_t a=0;a<3;++a)minus+=((_shape[a]-(i/_stride[a])%_shape[a])%_shape[a])*_stride[a];
        _negative.push_back(minus);
      }
    }
    // C[x]=sum_y A[x+y]*B[y]/Nk. The negative Fourier index is
    // deliberate: this is a complex correlation without conjugating B.
    std::vector<thc_matrix> correlate(const std::vector<thc_matrix>& a,const std::vector<thc_matrix>& b,
                                      bool right_is_q,bool result_is_q)const {
      if(a.size()!=_nk || b.size()!=_nk)throw std::runtime_error("THC FFT field shape mismatch");
      return correlate_prepared(a,prepare_right(b,right_is_q),result_is_q);
    }
    std::vector<thc_matrix> prepare_right(const std::vector<thc_matrix>& b,bool right_is_q=true)const {
      if(b.size()!=_nk)throw std::runtime_error("THC FFT prepared field shape mismatch");
      std::vector<thc_matrix> right(_nk);
      for(size_t i=0;i<_nk;++i)right[(right_is_q?_qmap:_kmap)[i]]=b[i];
      transform(right,false);return right;
    }
    std::vector<thc_matrix> correlate_prepared(const std::vector<thc_matrix>& a,const std::vector<thc_matrix>& right,
                                             bool result_is_q=false)const {
      if(a.size()!=_nk || right.size()!=_nk)throw std::runtime_error("THC FFT prepared correlation shape mismatch");
      std::vector<thc_matrix> left(_nk),product(_nk),result(_nk);
      for(size_t i=0;i<_nk;++i)left[_kmap[i]]=a[i];
      transform(left,false);
      for(size_t i=0;i<_nk;++i)product[i]=left[i].cwiseProduct(right[_negative[i]])/double(_nk);
      transform(product,true);
      for(size_t i=0;i<_nk;++i)result[i]=std::move(product[(result_is_q?_qmap:_kmap)[i]]);
      return result;
    }
  };

  inline bool thc_auxiliary_screening(const std::string& choice,size_t rank,size_t auxiliary) {
    if(choice!="auto" && choice!="point" && choice!="auxiliary")throw std::runtime_error("thc_gw_screening must be auto, point or auxiliary");
    return choice=="auxiliary" || (choice=="auto" && auxiliary<rank);
  }
  inline thc_matrix thc_dielectric_solve(const thc_matrix& P,const thc_matrix& rhs) {
    thc_matrix A=thc_matrix::Identity(P.rows(),P.rows())-P;
    thc_matrix solution=A.partialPivLu().solve(rhs);
    double residual=(A*solution-rhs).norm()/std::max(1.,rhs.norm());
    if(!solution.allFinite() || !std::isfinite(residual) || residual>1e-9)throw std::runtime_error("native THC screening residual failed");
    return solution;
  }
  /** Auxiliary dielectric core; expansion to interpolation space is deferred. */
  inline thc_matrix thc_screened_core(const thc_matrix& P) {
    return thc_dielectric_solve(P,P);
  }
  /** Exact low-rank identity for the same fitted interaction. Z is never inverted. */
  inline thc_matrix thc_screened_correlation(const thc_matrix& m,const thc_matrix& Z,
                                             const thc_matrix& response,bool auxiliary) {
    thc_matrix P,rhs;
    if(auxiliary){P=m.adjoint()*response*m;rhs=P;}
    else {P=Z*response;rhs=P*Z;}
    thc_matrix solution=thc_dielectric_solve(P,rhs);
    if(auxiliary)return m*solution*m.adjoint();
    return solution;
  }

  inline void check_thc_fft_workspace(const integrals::thc_factor_data& f,size_t nt,size_t nw,size_t budget,bool auxiliary=false) {
    const double r=f.rank(),n=f.nao(),Q=f.naux(),d=auxiliary?Q:r;
    // Histories and IR scratch use d=Q in auxiliary mode. Momentum fields
    // remain r*r because the q-dependent M cannot pass through a momentum FFT.
    // Include owned core copies, projection/backprojection and LU temporaries.
    const double histories=double(f.nq())*nt*d*d;
    const double transform=(nt+2.*nw+8)*d*d;
    const double momentum=12.*f.nk()*r*r+2.*f.nk()*r*n+2.*f.nk()*n*n;
    const double cores=(auxiliary?double(f.nq()):1.)*r*Q+4.*r*d;
    const double bytes=(histories+transform+momentum+cores)*sizeof(thc_complex);
    if(bytes>double(budget))throw std::runtime_error("THC FFT all-q workspace exceeds declared budget; use direct sums or increase budget");
  }
  struct thc_cpu_matrix_ops {
    thc_matrix gemm(const thc_matrix& a,const thc_matrix& b){return a*b;}
    thc_matrix project(const thc_matrix& x,const thc_matrix& g){return x*g*x.adjoint();}
    thc_matrix backproject(const thc_matrix& x,const thc_matrix& g){return x.adjoint()*g*x;}
    thc_matrix solve(const thc_matrix& a,const thc_matrix& b){return a.partialPivLu().solve(b);}
    thc_matrix screen(const thc_matrix& m,const thc_matrix& z,const thc_matrix& response,bool auxiliary){
      return thc_screened_correlation(m,z,response,auxiliary);
    }
    thc_matrix screen_core(const thc_matrix& response){return thc_screened_core(response);}
  };

  /** Hardware-neutral point-space GW algebra. Consumers own MPI scheduling;
   * Ops executes projection/screening/backprojection on CPU or CUDA. Momentum
   * FFTs here run on the host. This module does not reconstruct three-leg V.
   */
  template<class Transformer,class Ops>
  void thc_gw_fft_solve(const integrals::thc_factor_data& f,const Transformer& ft,
                        const thc_tensor5& g,thc_tensor5& sigma,Ops& ops,size_t budget,const std::string& screening="auto",
                        bool reuse_screening=true) {
    const size_t nt=g.shape()[0],ns=g.shape()[1],nk=f.nk(),n=f.nao(),r=f.rank(),nw=ft.sd().repn_bose().nw();
    const bool auxiliary=thc_auxiliary_screening(screening,r,f.naux());
    const size_t d=auxiliary?f.naux():r;
    thc_momentum_fft mesh(f);check_thc_fft_workspace(f,nt,nw,budget,auxiliary);
    std::vector<thc_matrix> cores;
    if(auxiliary){cores.reserve(f.nq());for(size_t q=0;q<f.nq();++q)cores.emplace_back(f.M(q));}
    std::vector<thc_tensor4> chi;
    for(size_t q=0;q<f.nq();++q){chi.emplace_back(nt,1,d,d);chi.back().set_zero();}
    auto map=[&](auto&& a){return Eigen::Map<thc_matrix>(a.data(),d,d);};
    auto green=[&](size_t t,size_t s,size_t k){return Eigen::Map<const thc_matrix>(g(t,s,k).data(),n,n);};
    for(size_t t=0;t<nt/2;++t) {
      std::vector<thc_matrix> accumulated;
      if(auxiliary)for(size_t q=0;q<f.nq();++q)accumulated.emplace_back(thc_matrix::Zero(r,r));
      for(size_t s=0;s<ns;++s) {
        std::vector<thc_matrix> left(nk),right(nk);
        for(size_t k=0;k<nk;++k) {
          left[k]=ops.project(f.X(k),green(nt-t-1,s,k)).transpose();
          right[k]=ops.project(f.X(k),green(t,s,k));
        }
        auto bubble=mesh.correlate(left,right,false,true);
        for(size_t q=0;q<f.nq();++q) {
          if(auxiliary)accumulated[q]-=(ns==2?1.0:2.0)*bubble[q];
          else map(chi[q](t,0))-=(ns==2?1.0:2.0)*bubble[q];
        }
      }
      for(size_t q=0;q<f.nq();++q) {
        // Compression commutes with Hermitian/time symmetrization. Sum spins
        // first so each q is compressed only nt/2 times, independent of ns.
        if(auxiliary)map(chi[q](t,0))=ops.gemm(ops.gemm(cores[q].adjoint(),accumulated[q]),cores[q]);
        thc_matrix bubble=map(chi[q](t,0));
        map(chi[q](t,0))=0.5*(bubble+bubble.adjoint()).eval();
        map(chi[q](nt-t-1,0))=map(chi[q](t,0));
      }
    }
    thc_tensor4 wc_w(nw,1,d,d);
    for(size_t q=0;q<f.nq();++q) {
      thc_matrix m,Z;
      if(!auxiliary){m=f.M(q);Z=ops.gemm(m,m.adjoint());}
      ft.tau_f_to_w_b(chi[q],wc_w,0,nw,true);
      for(size_t w=0;w<nw;++w) {
        if(auxiliary)map(wc_w(w,0))=ops.screen_core(map(wc_w(w,0)));
        else map(wc_w(w,0))=ops.screen(m,Z,map(wc_w(w,0)),false);
      }
      ft.w_b_to_tau_f(wc_w,chi[q],0,nt,true);
    }
    for(size_t t=0;t<nt;++t) {
      std::vector<thc_matrix> wc(f.nq());
      // Keep C(q,tau) in Q space; expand just the point-space momentum slice.
      for(size_t q=0;q<f.nq();++q) {
        if(auxiliary)wc[q]=ops.gemm(ops.gemm(cores[q],map(chi[q](t,0))),cores[q].adjoint());
        else wc[q]=map(chi[q](t,0));
      }
      auto prepared=reuse_screening?mesh.prepare_right(wc):std::vector<thc_matrix>{};
      if(reuse_screening)wc.clear(); // Prepared W replaces the untransformed slice.
      for(size_t s=0;s<ns;++s) {
        std::vector<thc_matrix> projected(nk);
        for(size_t k=0;k<nk;++k)projected[k]=ops.project(f.X(k),green(t,s,k));
        auto point_sigma=reuse_screening?mesh.correlate_prepared(projected,prepared):mesh.correlate(projected,wc,true,false);
        for(size_t k=0;k<nk;++k)
          Eigen::Map<thc_matrix>(sigma(t,s,k).data(),n,n)-=ops.backproject(f.X(k),point_sigma[k]);
      }
    }
    std::cout<<"Native THC GW full-mesh "<<thc_momentum_fft::cpu_backend()<<"; actual shifted Bloch values retained; FFT calls="<<mesh.fft_calls()
      <<"; screening FFT shared="<<reuse_screening<<std::endl;
  }
}
#endif
