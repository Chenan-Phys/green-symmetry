#ifndef GREEN_SYMMETRY_INTEGRAL_PAIR_MAP_H
#define GREEN_SYMMETRY_INTEGRAL_PAIR_MAP_H

#include <complex>
#include <cstddef>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

namespace green::symmetry {
  struct integral_reader_statistics {
    std::size_t factor_read_bytes=0, metric_read_bytes=0, chunk_reads=0,
                reconstructions=0, cache_hits=0, cache_misses=0, cache_evictions=0;
    double read_seconds=0, reconstruction_seconds=0;
  };
  struct integral_reader_options {
    std::string mode = "legacy";
    std::string input_file;
    std::string set_kind = "hf";
    std::size_t cache_bytes = 64ul * 1024 * 1024;
    template<class Parameters>
    static integral_reader_options from_parameters(const Parameters& p, const std::string& kind) {
      return {p["integral_symmetry"].template as<std::string>(),
              p["input_file"].template as<std::string>(), kind,
              p["integral_symmetry_cache_bytes"].template as<std::size_t>()};
    }
  };
  // Identical registrations are supported by green-params. Both applications
  // and the standalone GPU library register these shared defaults.
  template<class Parameters> void define_integral_parameters(Parameters& p) {
    p.template define<std::string>("integral_symmetry", "Integral storage: legacy or space_group", "legacy");
    p.template define<std::size_t>("integral_symmetry_cache_bytes", "Dense map cache budget per integral reader (bytes)", 64ul*1024*1024);
    p.template define<std::size_t>("integral_symmetry_preload_bytes", "SG representative preload budget per node (bytes)", 1024ul*1024*1024);
  }
  inline std::size_t integral_storage_bytes(std::size_t pairs, std::size_t naux, std::size_t nao,
                                            std::size_t element_bytes, std::size_t budget) {
    std::size_t bytes=element_bytes;
    for(auto factor : {pairs,naux,nao,nao}) {
      if(factor && bytes>std::numeric_limits<std::size_t>::max()/factor)
        throw std::overflow_error("integral storage byte count overflow");
      bytes*=factor;
    }
    if(bytes>budget) throw std::runtime_error("SG representative preload exceeds integral_symmetry_preload_bytes");
    return bytes;
  }
  /** Shared CPU/GPU descriptor and host algebra for green.df.space_group.v1.
   * Source buffers contain immutable representatives in captured CD gauges.
   * Requested outputs are owned by the caller. Target Q slices mix all source Q.
   * Dense maps and metric factors share a bounded cache, private to this object.
   */
  class integral_pair_map {
  public:
    static constexpr const char* format_name = "green.df.space_group.v1";
    static bool is_space_group(const std::string& directory); // rejects unknown formats
    static std::unique_ptr<integral_pair_map> open(const std::string& directory, const integral_reader_options& options);
    integral_pair_map(const std::string& directory, const std::string& input_fingerprint,
                      const std::string& set_kind, std::size_t cache_bytes = 256ul * 1024 * 1024);
    ~integral_pair_map();
    integral_pair_map(const integral_pair_map&) = delete;
    integral_pair_map& operator=(const integral_pair_map&) = delete;
    std::size_t nk() const;
    std::size_t nao() const;
    std::size_t naux() const;
    std::size_t nrepresentatives() const;
    std::size_t chunk_size() const;
    std::size_t representative(std::size_t ki, std::size_t kj) const;
    std::size_t chunk_valid_count(std::size_t start) const;
    std::size_t descriptor_bytes() const;
    std::size_t cache_bytes() const;
    std::size_t cache_peak_bytes() const;
    integral_reader_statistics statistics() const;
    void validate_input(const std::string& input_file) const;
    void read_chunk(std::size_t start, std::complex<double>* buffer, std::size_t capacity) const;
    void reconstruct(std::size_t ki, std::size_t kj, const std::complex<double>* representative_buffer,
                     std::complex<double>* requested_buffer, std::size_t offset = 0, std::size_t count = 0) const;
  private:
    struct implementation;
    std::unique_ptr<implementation> _impl;
  };
}
#endif
