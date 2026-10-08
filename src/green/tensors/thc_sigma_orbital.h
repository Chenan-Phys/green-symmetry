#ifndef GREEN_TENSORS_THC_SIGMA_ORBITAL_H
#define GREEN_TENSORS_THC_SIGMA_ORBITAL_H
#include <green/integrals/thc_factor_data.h>

namespace green::tensors {
  using thc_vertex_matrix=integrals::thc_factor_data::matrix;
  // Fixed vertices preserve the original auxiliary dimension and fitted M.
  // Rows are (output orbital, input orbital), columns are auxiliary indices.
  inline thc_vertex_matrix thc_orbital_vertex(const thc_vertex_matrix& x,const thc_vertex_matrix& xp,
                                             const thc_vertex_matrix& m) {
    const size_t n=x.cols(),r=x.rows();
    if(xp.rows()!=r || xp.cols()!=n || m.rows()!=r)throw std::runtime_error("THC orbital vertex shape mismatch");
    thc_vertex_matrix pairs(n*n,r);
    for(size_t a=0;a<n;++a)for(size_t i=0;i<n;++i)
      pairs.row(a*n+i)=x.col(a).conjugate().cwiseProduct(xp.col(i)).transpose();
    return pairs*m;
  }
  inline thc_vertex_matrix thc_orbital_sigma_weighted(const thc_vertex_matrix& v,const thc_vertex_matrix& weighted,
                                                     const thc_vertex_matrix& g) {
    const size_t n=g.rows(),Q=v.cols();
    if(g.cols()!=n || v.rows()!=n*n || weighted.rows()!=v.rows() || weighted.cols()!=Q)
      throw std::runtime_error("THC orbital Sigma shape mismatch");
    // C has already been applied once, shared across spins. Pack so G acts
    // on its own orbital index, then combine the auxiliary/input indices.
    thc_vertex_matrix stacked(n*Q,n);
    for(size_t a=0;a<n;++a)for(size_t A=0;A<Q;++A)for(size_t i=0;i<n;++i)
      stacked(a*Q+A,i)=weighted(a*n+i,A);
    thc_vertex_matrix first=stacked*g,horizontal(n,n*Q);
    for(size_t a=0;a<n;++a)for(size_t i=0;i<n;++i)for(size_t A=0;A<Q;++A)
      horizontal(a,i*Q+A)=first(a*Q+A,i);
    Eigen::Map<const thc_vertex_matrix> right(v.data(),n,n*Q);
    return horizontal*right.adjoint();
  }
  inline bool thc_sigma_orbital(const std::string& choice,size_t n,size_t r,size_t Q,size_t nk,size_t ns,
                                bool auxiliary,bool direct) {
    if(choice!="point" && choice!="orbital" && choice!="auto")throw std::runtime_error("thc_gw_sigma must be point, orbital or auto");
    if(choice=="orbital" && (!auxiliary || !direct))throw std::runtime_error("orbital THC Sigma requires auxiliary screening and direct momentum mode");
    if(!auxiliary || !direct || choice=="point")return false;
    if(choice=="orbital")return true;
    // Conservative cached-point operation estimate. Shape-dependent timings
    // are still required; explicit selections allow a controlled comparison.
    const double nn=n,rr=r,qq=Q,kk=nk,ss=ns;
    const double point=kk*(rr*rr*qq+rr*qq*qq)+ss*kk*(nn*rr*rr+nn*nn*rr+kk*rr*rr);
    const double orbital=kk*kk*(nn*nn*qq*qq+2.*ss*nn*nn*nn*qq);
    return orbital<.9*point;
  }
}
#endif
