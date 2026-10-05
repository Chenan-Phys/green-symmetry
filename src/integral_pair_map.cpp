#include "green/symmetry/integral_pair_map.h"
#include <Eigen/Dense>
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace green::symmetry {
namespace {
  using complex = std::complex<double>;
  using matrix = Eigen::Matrix<complex,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  using point = std::array<long,3>;
  constexpr std::size_t metadata_bound = 128ul*1024*1024;
  static_assert(sizeof(complex)==2*sizeof(double));
  [[noreturn]] void invalid(const std::string& message) { throw std::runtime_error("SG integral archive: "+message); }
  struct handle {
    hid_t id;
    herr_t (*close)(hid_t);
    handle(hid_t value,herr_t(*closer)(hid_t)):id(value),close(closer) { if(id<0) invalid("HDF5 open failed"); }
    ~handle() { close(id); }
    operator hid_t() const { return id; }
    handle(const handle&)=delete;
  };
  std::vector<hsize_t> shape(hid_t dataset) {
    handle space(H5Dget_space(dataset),H5Sclose);
    int rank=H5Sget_simple_extent_ndims(space);
    if(rank<0) invalid("invalid dataset rank");
    std::vector<hsize_t> dims(rank);
    H5Sget_simple_extent_dims(space,dims.data(),nullptr);
    return dims;
  }
  std::size_t checked_size(const std::vector<hsize_t>& dims,std::size_t width) {
    std::size_t count=1;
    for(auto d:dims) {
      if(d>metadata_bound/width || (d && count>metadata_bound/width/d)) invalid("descriptor exceeds 128 MiB array bound");
      count*=d;
    }
    return count;
  }
  template<class T> std::vector<T> read(hid_t file,const std::string& name,const std::vector<hsize_t>& expected) {
    handle data(H5Dopen2(file,name.c_str(),H5P_DEFAULT),H5Dclose);
    if(shape(data)!=expected) invalid("bad shape: "+name);
    handle type(H5Dget_type(data),H5Tclose);
    if constexpr(std::is_same_v<T,double>) {
      if(H5Tget_class(type)!=H5T_FLOAT || H5Tget_size(type)!=8) invalid("wrong float64 layout: "+name);
    } else if(H5Tget_class(type)!=H5T_INTEGER) invalid("wrong integer layout: "+name);
    std::vector<T> out(checked_size(expected,sizeof(T)));
    hid_t native=std::is_same_v<T,double>?H5T_NATIVE_DOUBLE:H5T_NATIVE_LONG;
    if(H5Dread(data,native,H5S_ALL,H5S_ALL,H5P_DEFAULT,out.data())<0) invalid("read failed: "+name);
    return out;
  }
  std::string attribute(hid_t file,const std::string& name) {
    if(H5Aexists(file,name.c_str())<=0) invalid("missing attribute: "+name);
    handle attr(H5Aopen(file,name.c_str(),H5P_DEFAULT),H5Aclose);
    handle attr_space(H5Aget_space(attr),H5Sclose);
    if(H5Sget_simple_extent_ndims(attr_space)!=0) invalid("non-scalar attribute: "+name);
    handle type(H5Aget_type(attr),H5Tclose);
    if(H5Tget_class(type)!=H5T_STRING) invalid("non-string attribute: "+name);
    std::string value;
    if(H5Tis_variable_str(type)>0) {
      char* text=nullptr;
      if(H5Aread(attr,type,&text)<0) invalid("attribute read failed");
      if(text) { value=text; H5free_memory(text); }
    } else {
      std::size_t length=H5Tget_size(type);
      if(length>4096) invalid("oversized attribute");
      std::vector<char> text(length+1,0);
      if(H5Aread(attr,type,text.data())<0) invalid("attribute read failed");
      value=text.data();
    }
    if(value.empty()) invalid("empty attribute: "+name);
    return value;
  }
  long floor_div(long x,long d) { long q=x/d; return q-((x%d)<0); }
  std::string string_dataset(hid_t file,const std::string& name) {
    handle data(H5Dopen2(file,name.c_str(),H5P_DEFAULT),H5Dclose);
    handle type(H5Dget_type(data),H5Tclose);
    if(!shape(data).empty() || H5Tget_class(type)!=H5T_STRING) invalid("invalid input identity dataset");
    std::string result;
    if(H5Tis_variable_str(type)>0) {
      char* value=nullptr;
      if(H5Dread(data,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)<0) invalid("input identity read failed");
      if(value) { result=value; H5free_memory(value); }
    } else {
      auto length=H5Tget_size(type); if(length>4096) invalid("oversized input identity");
      std::vector<char> value(length+1,0);
      if(H5Dread(data,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,value.data())<0) invalid("input identity read failed");
      result=value.data();
    }
    return result;
  }
  long modulo(long x,long d) { return x-floor_div(x,d)*d; }
  point wrapped(point value,long denominator) { for(auto& x:value) x=modulo(x,denominator); return value; }
  void close_matrix(const matrix& actual,const matrix& expected,const std::string& what) {
    if(actual.rows()!=expected.rows() || actual.cols()!=expected.cols() || !actual.allFinite() || !expected.allFinite()) invalid(what+" dimensions/nonfinite data");
    for(Eigen::Index i=0;i<actual.size();++i) if(std::abs(actual.data()[i]-expected.data()[i])>1e-10+1e-10*std::abs(expected.data()[i])) invalid(what+" covariance failed");
  }
}

struct integral_pair_map::implementation {
  std::string directory,fingerprint,kind,gauge_id,kernel_id,basis;
  std::size_t nk,nao,naux,nrep,nq,nops,capacity,descriptor=0,budget,resident=0,peak=0;
  long denominator,qdenominator;
  integral_reader_statistics stats;
  std::vector<long> mesh,qmesh,reps,pair_rep,pair_op,tr,ex,pair_q,wraps,rotations,translations,translation_denominators,mult,carries,inverses,starts,counts;
  std::vector<std::vector<long>> point_maps;
  std::map<point,long> k_ids,q_ids;
  struct sparse {
    std::vector<long> offsets,rows,columns;
    std::vector<double> values,phases;
  } orbital,auxiliary;
  std::vector<matrix> stored_x,stored_inverse;
  struct entry { std::shared_ptr<const matrix> value; std::list<std::string>::iterator position; };
  mutable std::list<std::string> recent;
  mutable std::map<std::string,entry> cache;

  implementation(const std::string& path,const std::string& expected,const std::string& expected_kind,std::size_t bytes)
    :directory(path),budget(bytes) {
    if(expected.empty() || (expected_kind!="hf" && expected_kind!="correlation")) invalid("explicit input/set identity is required");
    handle file(H5Fopen((path+"/meta.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
    if(attribute(file,"integral_format")!=format_name) invalid("unknown format");
    if(H5Aexists(file,"format_version")<=0) invalid("missing format version");
    handle version(H5Aopen(file,"format_version",H5P_DEFAULT),H5Aclose);
    handle version_space(H5Aget_space(version),H5Sclose);
    if(H5Sget_simple_extent_ndims(version_space)!=0) invalid("non-scalar format version");
    long number=0;
    if(H5Aread(version,H5T_NATIVE_LONG,&number)<0 || number!=1) invalid("unsupported format version");
    fingerprint=attribute(file,"input_fingerprint"); kind=attribute(file,"set_kind");
    gauge_id=attribute(file,"auxiliary_gauge_id"); kernel_id=attribute(file,"kernel_id"); basis=attribute(file,"orbital_basis");
    const auto finite_size=attribute(file,"finite_size_kind");
    if(finite_size!="none" && finite_size!="ewald") invalid("unsupported special finite-size profile");
    if(std::filesystem::exists(directory+"/df_ewald.h5") || std::filesystem::exists(directory+"/AqQ.h5"))
      invalid("special finite-size sidecars require a separately validated SG profile");
    attribute(file,"finite_size_kind"); attribute(file,"producer_revision");
    if(fingerprint!=expected || kind!=expected_kind) invalid("input fingerprint or HF/correlation set mismatch");
    if(attribute(file,"complex_storage")!="complex128_interleaved_f64" || attribute(file,"transform_direction")!="representative_to_requested" || attribute(file,"operation_order")!="spatial,time_reversal,exchange") invalid("unsupported precision/operation contract");
    if((basis!="ao" && basis!="square_X") || (kernel_id!="ordinary-Coulomb" && kernel_id!="ewald-Coulomb")) invalid("unsupported basis/kernel");
    if(kernel_id!=(finite_size=="ewald"?"ewald-Coulomb":"ordinary-Coulomb")) invalid("finite-size profile differs from captured Coulomb kernel");
    auto dimensions=read<long>(file,"sg/dimensions",{6});
    for(auto d:dimensions) if(d<=0 || d>2147483648l) invalid("invalid dimensions");
    nk=dimensions[0]; nao=dimensions[1]; naux=dimensions[2]; nrep=dimensions[3]; nq=dimensions[4]; nops=dimensions[5];
    if(nk>4096 || nops>4096 || naux>65536 || nao>65536 || naux*nao*nao>metadata_bound/16) invalid("dimensions exceed initial bounded-buffer domain");
    denominator=read<long>(file,"sg/mesh_denominator",{})[0]; qdenominator=read<long>(file,"sg/q_denominator",{})[0];
    if(denominator<=0 || qdenominator<=0 || denominator>2147483648l || qdenominator>2147483648l) invalid("invalid mesh denominator");
    auto ints=[&](const std::string& name,std::vector<hsize_t> dims) { auto data=read<long>(file,"sg/"+name,dims); descriptor+=data.size()*sizeof(long); return data; };
    mesh=ints("mesh",{nk,3}); qmesh=ints("q_mesh",{nq,3}); reps=ints("representative_pairs",{nrep,2});
    pair_rep=ints("pair_to_representative",{nk,nk}); pair_op=ints("pair_operation",{nk,nk});
    tr=ints("time_reversal",{nk,nk}); ex=ints("exchange",{nk,nk}); pair_q=ints("pair_q",{nk,nk}); wraps=ints("reciprocal_wraps",{nk,nk,2,3});
    rotations=ints("rotations",{nops,3,3}); translations=ints("translation_numerator",{nops,3}); translation_denominators=ints("translation_denominator",{nops,3});
    mult=ints("multiplication",{nops,nops}); carries=ints("translation_carries",{nops,nops,3}); inverses=ints("inverses",{nops});
    long chunk=read<long>(file,"sg/chunk_size",{})[0]; if(chunk<=0) invalid("invalid chunk capacity"); capacity=chunk;
    if(capacity>metadata_bound/(16*naux*nao*nao)) invalid("source chunk exceeds 128 MiB bound");
    std::size_t nchunks=(nrep+capacity-1)/capacity;
    starts=ints("chunk_start",{nchunks}); counts=ints("chunk_valid_count",{nchunks});
    for(std::size_t i=0;i<nchunks;++i) if(starts[i]!=long(i*capacity) || counts[i]!=long(std::min(capacity,nrep-i*capacity))) invalid("chunk coverage/final valid count");
    handle cdata(H5Dopen2(file,"sg/captured_C",H5P_DEFAULT),H5Dclose);
    handle ctype(H5Dget_type(cdata),H5Tclose);
    if(shape(cdata)!=std::vector<hsize_t>{nq,naux,2*naux} || H5Tget_class(ctype)!=H5T_FLOAT || H5Tget_size(ctype)!=8) invalid("unsupported retained rank/metric layout");
    auto sparse_load=[&](const std::string& name,std::size_t n) {
      sparse result;
      result.offsets=ints(name+"/offsets",{nops+1});
      long nnz=result.offsets.back(); if(nnz<0 || result.offsets.front()!=0) invalid("invalid sparse offsets");
      result.rows=ints(name+"/rows",{std::size_t(nnz)}); result.columns=ints(name+"/columns",{std::size_t(nnz)});
      result.values=read<double>(file,"sg/"+name+"/values",{std::size_t(2*nnz)});
      result.phases=read<double>(file,"sg/"+name+"/phase_vectors",{nops,n,3});
      descriptor+=(result.values.size()+result.phases.size())*sizeof(double);
      for(std::size_t i=1;i<=nops;++i) if(result.offsets[i]<result.offsets[i-1]) invalid("nonmonotone sparse offsets");
      for(long i=0;i<nnz;++i) if(result.rows[i]<0 || result.rows[i]>=long(n) || result.columns[i]<0 || result.columns[i]>=long(n)) invalid("invalid sparse row/column");
      for(double v:result.values) if(!std::isfinite(v)) invalid("nonfinite angular block");
      for(double v:result.phases) if(!std::isfinite(v) || std::abs(v-std::round(v))>1e-6) invalid("invalid row Bloch lattice phase");
      return result;
    };
    orbital=sparse_load("orbital",nao); auxiliary=sparse_load("auxiliary",naux);
    if(basis=="square_X") {
      for(auto name:{"stored_x","stored_x_inverse"}) {
        auto data=read<double>(file,std::string("sg/")+name,{nk,nao,2*nao});
        descriptor+=data.size()*sizeof(double);
        auto& target=std::string(name)=="stored_x"?stored_x:stored_inverse;
        for(std::size_t k=0;k<nk;++k) target.emplace_back(Eigen::Map<const matrix>(reinterpret_cast<const complex*>(data.data())+k*nao*nao,nao,nao));
      }
      for(std::size_t k=0;k<nk;++k) close_matrix(stored_x[k]*stored_inverse[k],matrix::Identity(nao,nao),"stored basis inverse");
    }
    validate_geometry();
  }

  point coordinate(const std::vector<long>& data,std::size_t k) const { return {data[3*k],data[3*k+1],data[3*k+2]}; }
  std::array<long,9> reciprocal(std::size_t op) const {
    const long* r=rotations.data()+9*op;
    for(int i=0;i<9;++i) if(std::abs(r[i])>10000) invalid("unbounded lattice rotation");
    long det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    if(std::abs(det)!=1) invalid("non-unimodular rotation");
    return {(r[4]*r[8]-r[5]*r[7])/det,(r[5]*r[6]-r[3]*r[8])/det,(r[3]*r[7]-r[4]*r[6])/det,
            (r[2]*r[7]-r[1]*r[8])/det,(r[0]*r[8]-r[2]*r[6])/det,(r[1]*r[6]-r[0]*r[7])/det,
            (r[1]*r[5]-r[2]*r[4])/det,(r[2]*r[3]-r[0]*r[5])/det,(r[0]*r[4]-r[1]*r[3])/det};
  }
  point apply(std::size_t op,point source,bool conjugate=false) const {
    auto r=reciprocal(op); point result{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) result[i]+=r[3*i+j]*source[j];
    if(conjugate) for(auto& v:result) v=-v;
    return result;
  }
  void validate_geometry() {
    for(auto v:mesh) if(std::abs(v)>2147483648l) invalid("unbounded k coordinate");
    for(auto v:qmesh) if(std::abs(v)>2147483648l) invalid("unbounded q coordinate");
    for(std::size_t k=0;k<nk;++k) if(!k_ids.emplace(wrapped(coordinate(mesh,k),denominator),k).second) invalid("duplicate k points");
    for(std::size_t q=0;q<nq;++q) if(!q_ids.emplace(wrapped(coordinate(qmesh,q),qdenominator),q).second) invalid("duplicate q points");
    long td=1;
    for(std::size_t i=0;i<3*nops;++i) {
      long d=translation_denominators[i],v=translations[i];
      if(d<=0 || d>96 || v<0 || v>=d || std::gcd(v,d)!=1) invalid("noncanonical affine translation");
      long reduced=td/std::gcd(td,d);
      if(reduced>2147483648l/d) invalid("translation common denominator overflow");
      td=reduced*d;
    }
    for(int i=0;i<9;++i) if(rotations[i]!=(i%4==0?1:0)) invalid("operation zero is not identity");
    for(int i=0;i<3;++i) if(translations[i]!=0) invalid("operation zero is not affine identity");
    std::set<std::vector<long>> unique_operations;
    for(std::size_t a=0;a<nops;++a) {
      std::vector<long> operation_key(rotations.begin()+9*a,rotations.begin()+9*(a+1));
      for(int i=0;i<3;++i) { operation_key.push_back(translations[3*a+i]); operation_key.push_back(translation_denominators[3*a+i]); }
      if(!unique_operations.insert(operation_key).second) invalid("duplicate affine operation");
      reciprocal(a);
      std::vector<long> mapped;
      for(std::size_t k=0;k<nk;++k) {
        auto found=k_ids.find(wrapped(apply(a,coordinate(mesh,k)),denominator));
        if(found==k_ids.end()) invalid("operation does not preserve actual mesh");
        mapped.push_back(found->second);
      }
      point_maps.push_back(std::move(mapped));
      for(std::size_t b=0;b<nops;++b) {
        long c=mult[a*nops+b]; if(c<0 || c>=long(nops)) invalid("invalid multiplication ID");
        for(int i=0;i<3;++i) {
          long v=translations[3*a+i]*(td/translation_denominators[3*a+i]);
          for(int j=0;j<3;++j) v+=rotations[9*a+3*i+j]*translations[3*b+j]*(td/translation_denominators[3*b+j]);
          if(modulo(v,td)!=translations[3*c+i]*(td/translation_denominators[3*c+i]) || floor_div(v,td)!=carries[3*(a*nops+b)+i]) invalid("affine translation composition/carry");
          for(int j=0;j<3;++j) {
            long r=0; for(int l=0;l<3;++l) r+=rotations[9*a+3*i+l]*rotations[9*b+3*l+j];
            if(r!=rotations[9*c+3*i+j]) invalid("rotation group is not closed");
          }
        }
      }
      long inv=inverses[a]; if(inv<0 || inv>=long(nops) || mult[a*nops+inv]!=0 || mult[inv*nops+a]!=0) invalid("invalid two-sided group inverse");
    }
    std::vector<bool> coverage(nrep,false);
    std::map<std::array<long,2>,bool> distinct;
    for(std::size_t p=0;p<nrep;++p) {
      std::array<long,2> rep{reps[2*p],reps[2*p+1]};
      if(rep[0]<0 || rep[1]<0 || rep[0]>=long(nk) || rep[1]>=long(nk) || !distinct.emplace(rep,true).second) invalid("invalid/duplicate representative pair");
    }
    for(std::size_t i=0;i<nk;++i) for(std::size_t j=0;j<nk;++j) {
      std::size_t p=i*nk+j; long rep=pair_rep[p],op=pair_op[p],q=pair_q[p];
      if(rep<0 || rep>=long(nrep) || op<0 || op>=long(nops) || q<0 || q>=long(nq) || tr[p]<0 || tr[p]>1 || ex[p]<0 || ex[p]>1) invalid("invalid pair/operation/gauge ID");
      coverage[rep]=true;
      for(int leg=0;leg<2;++leg) {
        long source=reps[2*rep+(ex[p]?1-leg:leg)];
        auto transformed=apply(op,coordinate(mesh,source),tr[p]);
        std::size_t target=leg?j:i;
        if(wrapped(transformed,denominator)!=wrapped(coordinate(mesh,target),denominator)) invalid("pair operation does not land on requested pair");
        for(int d=0;d<3;++d) if((transformed[d]-mesh[3*target+d])/denominator!=wraps[(2*p+leg)*3+d]) invalid("invalid reciprocal wrapping");
      }
      for(int d=0;d<3;++d) {
        double delta=double(mesh[3*j+d]-mesh[3*i+d])/denominator-double(qmesh[3*q+d])/qdenominator;
        if(std::abs(delta-std::round(delta))>1e-10) invalid("pair gauge q sign/frame mismatch");
      }
    }
    if(std::find(coverage.begin(),coverage.end(),false)!=coverage.end()) invalid("incomplete representative coverage");
  }

  std::shared_ptr<const matrix> cached(const std::string& key,const std::function<matrix()>& factory) {
    auto found=cache.find(key);
    if(found!=cache.end()) { ++stats.cache_hits; recent.splice(recent.end(),recent,found->second.position); return found->second.value; }
    ++stats.cache_misses;
    auto value=std::make_shared<const matrix>(factory());
    std::size_t bytes=value->size()*sizeof(complex);
    if(bytes<=budget) {
      while(resident+bytes>budget && !recent.empty()) {
        auto old=cache.find(recent.front()); resident-=old->second.value->size()*sizeof(complex); cache.erase(old); recent.pop_front(); ++stats.cache_evictions;
      }
      recent.push_back(key); cache.emplace(key,entry{value,std::prev(recent.end())}); resident+=bytes; peak=std::max(peak,resident);
    }
    return value;
  }
  matrix operation(const sparse& data,std::size_t n,std::size_t op,point target,long den) const {
    matrix out=matrix::Zero(n,n);
    for(long index=data.offsets[op];index<data.offsets[op+1];++index) {
      long row=data.rows[index]; double phase=0;
      for(int d=0;d<3;++d) phase+=double(target[d])/den*data.phases[(op*n+row)*3+d];
      out(row,data.columns[index])=complex(data.values[2*index],data.values[2*index+1])*std::exp(complex(0,2*std::acos(-1.)*phase));
    }
    return out;
  }
  std::shared_ptr<const matrix> metric_factor(std::size_t q) {
    return cached(gauge_id+":C:"+std::to_string(q),[&]() {
      handle file(H5Fopen((directory+"/meta.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
      handle data(H5Dopen2(file,"sg/captured_C",H5P_DEFAULT),H5Dclose);
      handle space(H5Dget_space(data),H5Sclose);
      hsize_t start[3]={q,0,0},count[3]={1,naux,2*naux};
      if(H5Sselect_hyperslab(space,H5S_SELECT_SET,start,nullptr,count,nullptr)<0) invalid("metric hyperslab");
      hsize_t dims[2]={naux,2*naux}; handle memory(H5Screate_simple(2,dims,nullptr),H5Sclose);
      matrix c(naux,naux);
      if(H5Dread(data,H5T_NATIVE_DOUBLE,memory,space,H5P_DEFAULT,reinterpret_cast<double*>(c.data()))<0 || !c.allFinite()) invalid("captured factor read/nonfinite");
      stats.metric_read_bytes+=naux*naux*sizeof(complex);
      double tolerance=1e-12*std::max(1.,c.norm());
      for(std::size_t i=0;i<naux;++i) {
        if(c(i,i).real()<=0 || std::abs(c(i,i).imag())>1e-12) invalid("nonpositive captured Cholesky factor");
        for(std::size_t j=i+1;j<naux;++j) if(std::abs(c(i,j))>tolerance) invalid("captured factor is not lower triangular");
      }
      Eigen::FullPivLU<matrix> lu(c); if(lu.rcond()<1e-12) invalid("ill-conditioned retained metric space");
      return c;
    });
  }
};

bool integral_pair_map::is_space_group(const std::string& directory) {
  handle file(H5Fopen((directory+"/meta.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
  if(H5Aexists(file,"integral_format")<=0) return false;
  if(attribute(file,"integral_format")!=format_name) invalid("unknown integral format");
  return true;
}
integral_pair_map::integral_pair_map(const std::string& path,const std::string& fingerprint,const std::string& kind,std::size_t cache)
  :_impl(std::make_unique<implementation>(path,fingerprint,kind,cache)) {}
integral_pair_map::~integral_pair_map()=default;
std::unique_ptr<integral_pair_map> integral_pair_map::open(const std::string& directory,const integral_reader_options& options) {
  if(options.mode!="legacy" && options.mode!="space_group") invalid("integral_symmetry must be legacy or space_group");
  if(options.cache_bytes>2ul*1024*1024*1024) invalid("dense map cache budget exceeds 2 GiB per reader");
  if(!is_space_group(directory)) return nullptr;
  if(options.mode!="space_group") invalid("space-group archive requires explicit --integral_symmetry space_group");
  if(options.input_file.empty()) invalid("space-group archive requires an identified input file");
  handle file(H5Fopen(options.input_file.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
  auto result=std::make_unique<integral_pair_map>(directory,string_dataset(file,"integral_symmetry/input_fingerprint"),options.set_kind,options.cache_bytes);
  result->validate_input(options.input_file);
  return result;
}
std::size_t integral_pair_map::nk() const { return _impl->nk; }
std::size_t integral_pair_map::nao() const { return _impl->nao; }
std::size_t integral_pair_map::naux() const { return _impl->naux; }
std::size_t integral_pair_map::nrepresentatives() const { return _impl->nrep; }
std::size_t integral_pair_map::chunk_size() const { return _impl->capacity; }
std::size_t integral_pair_map::descriptor_bytes() const { return _impl->descriptor; }
std::size_t integral_pair_map::cache_bytes() const { return _impl->resident; }
std::size_t integral_pair_map::cache_peak_bytes() const { return _impl->peak; }
integral_reader_statistics integral_pair_map::statistics() const { return _impl->stats; }
std::size_t integral_pair_map::representative(std::size_t i,std::size_t j) const {
  if(i>=nk() || j>=nk()) invalid("requested k point out of range");
  return _impl->pair_rep[i*nk()+j];
}
std::size_t integral_pair_map::chunk_valid_count(std::size_t start) const {
  if(start>=nrepresentatives() || start%chunk_size()) invalid("requested chunk out of range");
  return _impl->counts[start/chunk_size()];
}
void integral_pair_map::validate_input(const std::string& path) const {
  handle file(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
  if(string_dataset(file,"integral_symmetry/input_fingerprint")!=_impl->fingerprint) invalid("identified input fingerprint differs from archive");
  auto mesh=read<double>(file,"symmetry/k/mesh_scaled",{nk(),3});
  for(std::size_t i=0;i<mesh.size();++i) if(!std::isfinite(mesh[i]) || std::abs(mesh[i]-double(_impl->mesh[i])/_impl->denominator)>1e-10) invalid("actual input k mesh differs from descriptor");
  if(read<long>(file,"params/nao",{})[0]!=long(nao()) || read<long>(file,"params/nso",{})[0]!=long(nao()) || read<long>(file,"params/NQ",{})[0]!=long(naux())) invalid("input dimension/spinor/retained-space mismatch");
  if(_impl->kind=="correlation" && H5Lexists(file,"symmetry/q/inq",H5P_DEFAULT)>0 &&
     read<long>(file,"symmetry/q/inq",{})[0]<long(_impl->nq)) {
    if(string_dataset(file,"integral_symmetry/correlation_gauge_id")!=_impl->gauge_id)
      invalid("reduced q symmetry requires the actual correlation auxiliary frame");
  }
}
void integral_pair_map::read_chunk(std::size_t start,complex* output,std::size_t capacity) const {
  const auto started=std::chrono::steady_clock::now();
  std::size_t count=chunk_valid_count(start);
  if(!output || capacity<count || capacity>chunk_size()) invalid("invalid chunk buffer capacity");
  handle file(H5Fopen((_impl->directory+"/SGVQ_"+std::to_string(start)+".h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
  auto data=read<double>(file,"factors",{count,naux(),nao(),2*nao()});
  for(auto v:data) if(!std::isfinite(v)) invalid("nonfinite representative factor");
  std::fill(output,output+capacity*naux()*nao()*nao(),complex{});
  std::copy(data.begin(),data.end(),reinterpret_cast<double*>(output));
  _impl->stats.factor_read_bytes+=count*naux()*nao()*nao()*sizeof(complex);
  ++_impl->stats.chunk_reads;
  _impl->stats.read_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
}
void integral_pair_map::reconstruct(std::size_t i,std::size_t j,const complex* source,complex* output,std::size_t offset,std::size_t count) const {
  const auto started=std::chrono::steady_clock::now();
  auto& d=*_impl; representative(i,j);
  if(count==0) count=naux();
  if(!source || !output || source==output || offset>=naux() || count>naux()-offset) invalid("invalid owned output/auxiliary slice");
  std::size_t p=i*nk()+j,position=d.pair_rep[p],op=d.pair_op[p];
  std::size_t r0=d.reps[2*position],r1=d.reps[2*position+1],si=d.point_maps[op][r0],sj=d.point_maps[op][r1];
  bool tr=d.tr[p],ex=d.ex[p],conjugate=tr!=ex;
  point q{};
  for(int axis=0;axis<3;++axis) {
    long numerator=(d.mesh[3*sj+axis]-d.mesh[3*si+axis])*d.qdenominator;
    if(numerator%d.denominator) invalid("incompatible integer q meshes");
    q[axis]=numerator/d.denominator;
  }
  auto qfound=d.q_ids.find(wrapped(q,d.qdenominator));
  if(qfound==d.q_ids.end()) invalid("spatial q is absent");
  std::size_t iq=qfound->second,qs=d.pair_q[r0*nk()+r1],qt=d.pair_q[p];
  std::string key=d.kind+":"+d.gauge_id+":complex128:v1:D:"+std::to_string(op)+":"+std::to_string(iq)+":"+std::to_string(qs)+":"+std::to_string(qt)+":"+std::to_string(conjugate);
  auto auxiliary=d.cached(key,[&]() {
    matrix a=d.operation(d.auxiliary,naux(),op,d.coordinate(d.qmesh,iq),d.qdenominator);
    matrix cs=*d.metric_factor(qs),ct=*d.metric_factor(qt);
    if(conjugate) { a=a.conjugate().eval(); cs=cs.conjugate().eval(); }
    matrix mapped=a*cs;
    close_matrix(mapped*mapped.adjoint(),ct*ct.adjoint(),"captured auxiliary metric");
    return matrix(ct.triangularView<Eigen::Lower>().solve(mapped));
  });
  auto ao=[&](std::size_t k) {
    return d.cached(d.gauge_id+":AO:"+std::to_string(op)+":"+std::to_string(k),[&]() {
      return d.operation(d.orbital,nao(),op,d.coordinate(d.mesh,k),d.denominator);
    });
  };
  matrix left=*ao(ex?sj:si),right=*ao(ex?si:sj);
  if(tr) { left=left.conjugate().eval(); right=right.conjugate().eval(); }
  if(d.basis=="square_X") {
    matrix li=d.stored_inverse[ex?r1:r0],ri=d.stored_inverse[ex?r0:r1];
    if(tr) { li=li.conjugate().eval(); ri=ri.conjugate().eval(); }
    left=(d.stored_x[i]*left*li).eval(); right=(d.stored_x[j]*right*ri).eval();
  }
  matrix rotated(naux(),nao()*nao());
  for(std::size_t b=0;b<naux();++b) {
    matrix v=Eigen::Map<const matrix>(source+b*nao()*nao(),nao(),nao());
    if(conjugate) v=v.conjugate().eval();
    if(ex) v=v.transpose().eval();
    Eigen::Map<matrix> block(rotated.data()+b*nao()*nao(),nao(),nao());
    block.noalias()=left*v*right.adjoint();
  }
  Eigen::Map<matrix> target(output,count,nao()*nao());
  target.noalias()=auxiliary->middleRows(offset,count)*rotated;
  ++d.stats.reconstructions;
  d.stats.reconstruction_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
}
}
