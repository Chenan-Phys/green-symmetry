#ifndef GREEN_TENSORS_THC_HOST_FFT_H
#define GREEN_TENSORS_THC_HOST_FFT_H
#include <array>
#include <complex>
#include <memory>
namespace green::tensors {
  /** Optional batched host FFT, with cached plans hidden from CUDA headers. */
  class thc_host_fft {
    struct implementation;
    std::unique_ptr<implementation> _impl;
  public:
    thc_host_fft();
    ~thc_host_fft();
    thc_host_fft(const thc_host_fft&)=delete;
    thc_host_fft& operator=(const thc_host_fft&)=delete;
    bool transform(std::complex<double>* field,const std::array<size_t,3>& shape,size_t entries,bool inverse);
    static bool enabled();
    static const char* backend();
  };
}
#endif
