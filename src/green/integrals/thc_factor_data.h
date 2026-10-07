#ifndef GREEN_INTEGRALS_THC_FACTOR_DATA_H
#define GREEN_INTEGRALS_THC_FACTOR_DATA_H

#include <Eigen/Dense>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <green/params/params.h>
#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace green::integrals {
  struct thc_reader_options {
    bool enabled = false;
    std::string input_file;
    size_t factor_memory_bytes = 512ul * 1024 * 1024;
    bool preload_all_cores = false;
  };

  inline void define_thc_parameters(params::params& p) {
    p.define<std::string>("interaction_representation", "Interaction storage representation: df or thc", "df");
    p.define<std::string>("thc_mode", "Explicit THC evaluation: reconstruct or native", "");
    p.define<size_t>("thc_factor_memory_mb", "Maximum host X plus cached/preloaded original-Q cores in MiB", 512);
  }

  inline thc_reader_options thc_options(const params::params& p) {
    thc_reader_options options;
    const auto representation=p["interaction_representation"].as<std::string>();
    const auto mode=p["thc_mode"].as<std::string>();
    if(representation!="df" && representation!="thc") throw std::runtime_error("interaction_representation must be df or thc");
    if(representation=="df" && !mode.empty()) throw std::runtime_error("thc_mode requires explicit interaction_representation thc");
    if(representation=="thc" && mode!="reconstruct" && mode!="native") throw std::runtime_error("THC requires explicit thc_mode reconstruct or native");
    options.enabled=representation=="thc";
    options.input_file=p["input_file"].as<std::string>();
    const size_t mb=p["thc_factor_memory_mb"].as<size_t>();
    if(!mb || mb>std::numeric_limits<size_t>::max()/(1024*1024)) throw std::runtime_error("invalid THC host factor budget");
    options.factor_memory_bytes=mb*1024*1024;
    return options;
  }

  /** Hardware-independent host factors; one X and one lazy q core per handle.
   * Original Q is retained. No Gaussian-q symmetry acts on interpolation I.
   */
  class thc_factor_data {
  public:
    using complex = std::complex<double>;
    using matrix = Eigen::Matrix<complex,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    using const_matrix = Eigen::Map<const matrix>;
    static constexpr const char* orientation = "L_Qmn=sum_I conj(X_ki_Im)*X_kj_In*M_q_IQ";

    static bool exists(const std::string& path) { return std::filesystem::exists(path + "/thc_meta.h5"); }

    thc_factor_data(const std::string& path, size_t nk, size_t nao, size_t naux, const thc_reader_options& options) :
      _path(path), _budget(options.factor_memory_bytes), _preload(options.preload_all_cores) {
      if (!options.enabled) throw std::runtime_error("THC requires explicit interaction_representation and thc_mode");
      file meta(path + "/thc_meta.h5");
      if (number(meta.id,"schema_version") != 1 || number(meta.id,"complete") != 1 ||
          string(meta.id,"schema_name") != "green.thc.df_fit" || string(meta.id,"orientation") != orientation)
        throw std::runtime_error("unknown, incompatible, or incomplete THC descriptor");
      if(number(meta.id,"fit_accepted")!=1 || number(meta.id,"original_Q_retained")!=1 || string(meta.id,"construction")!="df_fit")
        throw std::runtime_error("unaccepted or unsupported THC construction/capabilities");
      _nk = positive(meta.id,"nk"); _n = positive(meta.id,"nao");
      _r = positive(meta.id,"n_interp"); _Q = positive(meta.id,"naux_original"); _nq = positive(meta.id,"nq");
      if (_nk != nk || _n != nao || _Q != naux) throw std::runtime_error("THC/source solver dimensions differ");
      _set = string(meta.id,"set_kind");
      if (_set != "hf" && _set != "correlation") throw std::runtime_error("invalid THC interaction set");
      if (options.input_file.empty() || checksum_file(options.input_file) != string(meta.id,"input_fnv64"))
        throw std::runtime_error("THC input/orbital basis fingerprint mismatch");
      if(H5Lexists(meta.id,"correction_present",H5P_DEFAULT)>0 && number(meta.id,"correction_present")) {
        if(checksum_file(path+"/df_ewald.h5")!=string(meta.id,"correction_fnv64")) throw std::runtime_error("THC correction sidecar checksum mismatch");
      } else if(std::filesystem::exists(path+"/df_ewald.h5")) {
        throw std::runtime_error("unfingerprinted THC correction sidecar");
      }
      size_t xcount = multiply(multiply(_nk,_r),_n), mcount = multiply(_r,_Q);
      const size_t core_count=multiply(mcount,_preload?_nq:1);
      if(xcount>std::numeric_limits<size_t>::max()-core_count)throw std::runtime_error("THC factor size overflow");
      if (multiply(xcount + core_count,sizeof(complex)) > _budget)
        throw std::runtime_error("THC factors exceed declared host memory budget");
      _map = longs(meta.id,"pair_to_q",{_nk,_nk});
      for (long q : _map) if (q < 0 || size_t(q) >= _nq) throw std::runtime_error("THC pair-to-q map out of range");
      _pairs = longs(meta.id,"source_pairs",{_nk*(_nk+1)/2,2});
      _conj = longs(meta.id,"source_conj",{_nk*(_nk+1)/2});
      _trans = longs(meta.id,"source_trans",{_nk*(_nk+1)/2});
      file original(options.input_file);
      if(_pairs!=longs(original.id,"symmetry/pairs/kpair_idx",{_nk*(_nk+1)/2,2}) ||
         _conj!=longs(original.id,"symmetry/pairs/conj_pairs_list",{_nk*(_nk+1)/2}) ||
         _trans!=longs(original.id,"symmetry/pairs/trans_pairs_list",{_nk*(_nk+1)/2}))
        throw std::runtime_error("THC/source pair maps differ");
      auto kmesh=doubles(meta.id,"k_mesh_scaled",{_nk,3});
      if(kmesh!=doubles(original.id,"symmetry/k/mesh_scaled",{_nk,3})) throw std::runtime_error("THC/input k ordering differs");
      auto qmesh=doubles(meta.id,"q_mesh_scaled",{_nq,3});
      for(size_t i=0;i<_nk;++i) for(size_t j=0;j<_nk;++j)
        for(size_t axis=0;axis<3;++axis) {
          double diff=kmesh[j*3+axis]-kmesh[i*3+axis]-qmesh[_map[i*_nk+j]*3+axis];
          if(std::abs(diff-std::round(diff))>1e-8) throw std::runtime_error("invalid THC transfer convention/map");
        }
      const size_t np = _nk*(_nk+1)/2;
      for (size_t i=0;i<np;++i) {
        if (_conj[i]<0 || size_t(_conj[i])>=np || _trans[i]<0 || size_t(_trans[i])>=np)
          throw std::runtime_error("THC source pair map out of range");
        if (_pairs[2*i]<0 || size_t(_pairs[2*i])>=_nk || _pairs[2*i+1]<0 || size_t(_pairs[2*i+1])>=_nk)
          throw std::runtime_error("THC source representative out of range");
      }
      _X = factors(path + "/X.h5","X",{_nk,_r,2*_n});
      // Verify every core and its shape before a solver can begin. Keep one in RAM.
      for (size_t q=0;q<_nq;++q) {
        if(_preload) _allM.push_back(read_core(q));
        else _M=read_core(q);
      }
      _cached_q = _nq - 1;
    }

    size_t nk() const { return _nk; }
    size_t nao() const { return _n; }
    size_t rank() const { return _r; }
    size_t naux() const { return _Q; }
    size_t nq() const { return _nq; }
    const std::string& set_kind() const { return _set; }
    size_t transfer(size_t i,size_t j) const { check_pair(i,j); return _map[i*_nk+j]; }
    const_matrix X(size_t k) const { if (k>=_nk) throw std::out_of_range("THC k point"); return const_matrix(_X.data()+k*_r*_n,_r,_n); }
    matrix M(size_t q) const {
      std::lock_guard<std::mutex> guard(_mutex);
      load_core(q);
      return const_matrix(core_data(),_r,_Q);
    }
    matrix Z(size_t q) const { matrix m=M(q); return m*m.adjoint(); }
    std::pair<size_t,size_t> source_representative(size_t i,size_t j) const {
      check_pair(i,j);
      const size_t index=std::max(i,j)*(std::max(i,j)+1)/2+std::min(i,j);
      const size_t rep=_conj[index]!=long(index)?_conj[index]:_trans[index];
      return {size_t(_pairs[2*rep]),size_t(_pairs[2*rep+1])};
    }

    void reconstruct(size_t ki,size_t kj,complex* result,size_t offset=0,size_t count=0) const {
      check_pair(ki,kj);
      if (!count) count=_Q;
      if (offset>_Q || count>_Q-offset || !result) throw std::out_of_range("THC original-Q slice");
      std::lock_guard<std::mutex> guard(_mutex);
      load_core(transfer(ki,kj));
      auto left=X(ki),right=X(kj);
      const_matrix core(core_data(),_r,_Q);
      std::fill(result,result+count*_n*_n,complex(0));
      // Bound design/RHS tiles independently of nao^2 and interpolation rank.
      for(size_t row=0;row<_n*_n;row+=64) {
        size_t nr=std::min(size_t(64),_n*_n-row);
        for(size_t qi=0;qi<count;qi+=64) {
          size_t nc=std::min(size_t(64),count-qi);
          matrix tile=matrix::Zero(nr,nc);
          for(size_t p=0;p<_r;p+=64) {
            size_t ni=std::min(size_t(64),_r-p);
            matrix F(nr,ni);
            for(size_t a=0;a<nr;++a)
              for(size_t b=0;b<ni;++b)
                F(a,b)=std::conj(left(p+b,(row+a)/_n))*right(p+b,(row+a)%_n);
            tile.noalias() += F*core.block(p,offset+qi,ni,nc);
          }
          for(size_t a=0;a<nr;++a) for(size_t b=0;b<nc;++b) result[(qi+b)*_n*_n+row+a]=tile(a,b);
        }
      }
    }

    static std::string checksum_file(const std::string& path) {
      std::ifstream stream(path,std::ios::binary);
      if (!stream) throw std::runtime_error("missing THC fingerprinted input");
      uint64_t h=14695981039346656037ull; char block[65536];
      while(stream) { stream.read(block,sizeof(block)); for(std::streamsize i=0;i<stream.gcount();++i) h=(h^static_cast<unsigned char>(block[i]))*1099511628211ull; }
      return hex(h);
    }

  private:
    struct file {
      hid_t id;
      explicit file(const std::string& path) : id(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT)) {
        if(id<0) throw std::runtime_error("missing THC factor/descriptor: "+path);
      }
      ~file(){H5Fclose(id);}
    };
    static size_t multiply(size_t a,size_t b) {
      if(b && a>std::numeric_limits<size_t>::max()/b) throw std::runtime_error("THC allocation size overflow");
      return a*b;
    }
    static long number(hid_t f,const char* name) {
      long value=0;
      verify_shape(f,name,{});
      if(H5LTread_dataset_long(f,name,&value)<0) throw std::runtime_error("invalid THC integer field");
      return value;
    }
    static size_t positive(hid_t f,const char* name) {
      long value=number(f,name); if(value<=0) throw std::runtime_error("nonpositive THC dimension"); return size_t(value);
    }
    static std::string string(hid_t f,const char* name,bool attribute=false) {
      hid_t object=attribute?H5Aopen(f,name,H5P_DEFAULT):H5Dopen2(f,name,H5P_DEFAULT);
      if(object<0) throw std::runtime_error("missing THC string field");
      hid_t type=attribute?H5Aget_type(object):H5Dget_type(object);
      std::string result;
      if(H5Tget_class(type)!=H5T_STRING) { H5Tclose(type); if(attribute) H5Aclose(object); else H5Dclose(object); throw std::runtime_error("THC string dtype"); }
      if(H5Tis_variable_str(type)) {
        char* ptr=nullptr;
        herr_t status=attribute?H5Aread(object,type,&ptr):H5Dread(object,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,&ptr);
        if(status>=0 && ptr) {result=ptr; H5free_memory(ptr);}
      } else {
        std::vector<char> buffer(H5Tget_size(type)+1,0);
        if(attribute) H5Aread(object,type,buffer.data()); else H5Dread(object,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,buffer.data());
        result=buffer.data();
      }
      H5Tclose(type); if(attribute) H5Aclose(object); else H5Dclose(object);
      return result;
    }
    static void verify_shape(hid_t f,const char* name,const std::vector<size_t>& expected) {
      hid_t d=H5Dopen2(f,name,H5P_DEFAULT); if(d<0) throw std::runtime_error("missing THC dataset");
      hid_t s=H5Dget_space(d); int rank=H5Sget_simple_extent_ndims(s);
      std::vector<hsize_t> dims(std::max(0,rank)); H5Sget_simple_extent_dims(s,dims.data(),nullptr);
      H5Sclose(s); H5Dclose(d);
      if(rank!=int(expected.size()) || !std::equal(dims.begin(),dims.end(),expected.begin())) throw std::runtime_error("THC dataset shape mismatch");
    }
    static std::vector<long> longs(hid_t f,const char* name,const std::vector<size_t>& shape) {
      verify_shape(f,name,shape); size_t count=1; for(size_t d:shape) count=multiply(count,d);
      std::vector<long> value(count);
      if(H5LTread_dataset_long(f,name,value.data())<0) throw std::runtime_error("invalid THC map");
      return value;
    }
    static std::vector<double> doubles(hid_t f,const char* name,const std::vector<size_t>& shape) {
      verify_shape(f,name,shape); size_t count=1; for(size_t d:shape) count=multiply(count,d);
      std::vector<double> value(count);
      if(H5LTread_dataset_double(f,name,value.data())<0) throw std::runtime_error("invalid THC grid");
      for(double x:value) if(!std::isfinite(x)) throw std::runtime_error("nonfinite THC grid");
      return value;
    }
    static std::string hex(uint64_t h) { std::ostringstream s; s<<std::hex<<std::setw(16)<<std::setfill('0')<<h; return s.str(); }
    static std::vector<complex> factors(const std::string& path,const char* name,const std::vector<size_t>& shape) {
      file f(path); verify_shape(f.id,name,shape);
      hid_t dataset=H5Dopen2(f.id,name,H5P_DEFAULT),type=H5Dget_type(dataset);
      bool correct_type=H5Tget_class(type)==H5T_FLOAT && H5Tget_size(type)==sizeof(double) && H5Tget_order(type)==H5T_ORDER_LE;
      H5Tclose(type); H5Dclose(dataset);
      if(!correct_type) throw std::runtime_error("THC factor dtype must be little-endian float64 packed complex128");
      size_t count=1; for(size_t d:shape) count=multiply(count,d);
      std::vector<complex> data(count/2);
      if(H5LTread_dataset_double(f.id,name,reinterpret_cast<double*>(data.data()))<0) throw std::runtime_error("invalid THC factors");
      uint64_t h=14695981039346656037ull;
      auto* bytes=reinterpret_cast<const unsigned char*>(data.data());
      for(size_t i=0;i<count*sizeof(double);++i) h=(h^bytes[i])*1099511628211ull;
      if(hex(h)!=string(f.id,"fnv64",true)) throw std::runtime_error("THC factor checksum mismatch");
      for(auto v:data) if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) throw std::runtime_error("nonfinite THC factors");
      return data;
    }
    std::vector<complex> read_core(size_t q) const { return factors(_path+"/M_q_"+std::to_string(q)+".h5","M",{_r,2*_Q}); }
    void load_core(size_t q) const {
      if(q>=_nq) throw std::out_of_range("THC q core");
      if(q!=_cached_q) { if(!_preload) _M=read_core(q); _cached_q=q; }
    }
    const complex* core_data() const { return _preload?_allM[_cached_q].data():_M.data(); }
    void check_pair(size_t i,size_t j) const { if(i>=_nk || j>=_nk) throw std::out_of_range("THC k pair"); }
    std::string _path,_set;
    size_t _budget,_nk=0,_n=0,_r=0,_Q=0,_nq=0;
    bool _preload;
    std::vector<long> _map,_pairs,_conj,_trans;
    std::vector<complex> _X;
    mutable std::vector<complex> _M;
    std::vector<std::vector<complex>> _allM;
    mutable size_t _cached_q=std::numeric_limits<size_t>::max();
    mutable std::mutex _mutex;
  };
}
#endif
