#include <green/symmetry/integral_pair_map.h>
#include <green/symmetry/common_defs.h>
#include <algorithm>
#include <iostream>
#include <vector>

int main(int argc,char** argv) {
  try {
    if(argc<4) throw std::runtime_error("usage: integral_archive_probe SG_DIRECTORY EXPECTED_PACKED_H5 INPUT_H5 [CACHE_BYTES]");
    green::h5pp::archive meta(std::string(argv[1])+"/meta.h5","r");
    green::h5pp::archive input(argv[3],"r");
    std::string fingerprint;
    input["integral_symmetry/input_fingerprint"] >> fingerprint;
    std::string kind=argc>5?argv[5]:"hf";
    if(!green::symmetry::integral_pair_map::is_space_group(argv[1])) throw std::runtime_error("expected SG archive");
    std::size_t budget=argc>4?std::stoull(argv[4]):4096;
    green::symmetry::integral_pair_map map(argv[1],fingerprint,kind,budget);
    map.validate_input(argv[3]);
    green::symmetry::dtensor<5> reference;
    green::h5pp::archive expected(argv[2],"r");
    expected["factors"] >> reference;
    if(reference.shape()!=std::array<std::size_t,5>{map.nk(),map.nk(),map.naux(),map.nao(),2*map.nao()}) throw std::runtime_error("expected factors shape mismatch");
    std::size_t size=map.naux()*map.nao()*map.nao(),offset=map.naux()>2?2:0,count=std::min(std::size_t(3),map.naux()-offset);
    std::vector<std::complex<double>> chunk(map.chunk_size()*size),output(size),slice(count*map.nao()*map.nao());
    auto ref=reinterpret_cast<const std::complex<double>*>(reference.data());
    double maximum=0.,squared=0.,norm=0.;
    long current=-1;
    for(std::size_t i=0;i<map.nk();++i) for(std::size_t j=0;j<map.nk();++j) {
      std::size_t position=map.representative(i,j),start=position/map.chunk_size()*map.chunk_size();
      if(current!=long(start)) { map.read_chunk(start,chunk.data(),map.chunk_size()); current=start; }
      auto before=chunk;
      auto source=chunk.data()+(position-start)*size;
      map.reconstruct(i,j,source,output.data());
      if(chunk!=before) throw std::runtime_error("source representative buffer changed");
      map.reconstruct(i,j,source,slice.data(),offset,count);
      for(std::size_t index=0;index<size;++index) {
        auto expected_value=ref[(i*map.nk()+j)*size+index];
        double error=std::abs(output[index]-expected_value);
        maximum=std::max(maximum,error); squared+=error*error; norm+=std::norm(expected_value);
        if(error>1e-10+1e-10*std::abs(expected_value)) throw std::runtime_error("requested pair differs from direct physical reference");
      }
      for(std::size_t index=0;index<slice.size();++index) if(std::abs(slice[index]-output[offset*map.nao()*map.nao()+index])>1e-12) throw std::runtime_error("auxiliary slice lost source rows");
    }
    if(map.cache_bytes()>budget || map.cache_peak_bytes()>budget) throw std::runtime_error("dense cache exceeded budget");
    auto stats=map.statistics();
    std::cout << "{\"pairs\":" << map.nk()*map.nk() << ",\"representatives\":" << map.nrepresentatives()
      << ",\"max_absolute\":" << maximum << ",\"relative_frobenius\":" << std::sqrt(squared/norm)
      << ",\"descriptor_bytes\":" << map.descriptor_bytes() << ",\"cache_peak_bytes\":" << map.cache_peak_bytes()
      << ",\"factor_read_bytes\":" << stats.factor_read_bytes << ",\"metric_read_bytes\":" << stats.metric_read_bytes
      << ",\"chunk_reads\":" << stats.chunk_reads << ",\"read_seconds\":" << stats.read_seconds
      << ",\"reconstructions\":" << stats.reconstructions << ",\"reconstruction_seconds\":" << stats.reconstruction_seconds
      << ",\"cache_hits\":" << stats.cache_hits << ",\"cache_misses\":" << stats.cache_misses
      << ",\"cache_evictions\":" << stats.cache_evictions << ",\"status\":\"PASS\"}\n";
  } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
