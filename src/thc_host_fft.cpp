#include "green/tensors/thc_host_fft.h"
#include <mutex>
#include <limits>
#include <stdexcept>
#ifdef GREEN_THC_FFTW
#include GREEN_THC_FFTW_HEADER
#endif
namespace green::tensors {
namespace {
  std::mutex& planner_mutex(){static std::mutex mutex;return mutex;}
}
struct thc_host_fft::implementation {
#ifdef GREEN_THC_FFTW
  fftw_plan forward=nullptr,inverse=nullptr;
  size_t entries=0;std::array<size_t,3> shape{};
  ~implementation(){std::lock_guard<std::mutex> lock(planner_mutex());if(forward)fftw_destroy_plan(forward);if(inverse)fftw_destroy_plan(inverse);}
#endif
};
thc_host_fft::thc_host_fft():_impl(new implementation){}
thc_host_fft::~thc_host_fft()=default;
bool thc_host_fft::enabled(){
#ifdef GREEN_THC_FFTW
  return true;
#else
  return false;
#endif
}
const char* thc_host_fft::backend(){return enabled()?"batched FFTW":"Eigen fallback";}
bool thc_host_fft::transform(std::complex<double>* field,const std::array<size_t,3>& shape,size_t entries,bool backwards){
#ifdef GREEN_THC_FFTW
  if(entries>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("THC FFT batch dimension too large");
  auto* data=reinterpret_cast<fftw_complex*>(field);
  if(!_impl->forward || _impl->entries!=entries || _impl->shape!=shape){
    std::lock_guard<std::mutex> lock(planner_mutex());
    if(_impl->forward)fftw_destroy_plan(_impl->forward);if(_impl->inverse)fftw_destroy_plan(_impl->inverse);
    _impl->forward=nullptr;_impl->inverse=nullptr;int dims[3];
    for(size_t a=0;a<3;++a){if(shape[a]>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("THC FFT mesh dimension too large");dims[a]=int(shape[a]);}
    _impl->forward=fftw_plan_many_dft(3,dims,int(entries),data,nullptr,int(entries),1,data,nullptr,int(entries),1,FFTW_FORWARD,FFTW_ESTIMATE|FFTW_UNALIGNED);
    _impl->inverse=fftw_plan_many_dft(3,dims,int(entries),data,nullptr,int(entries),1,data,nullptr,int(entries),1,FFTW_BACKWARD,FFTW_ESTIMATE|FFTW_UNALIGNED);
    if(!_impl->forward || !_impl->inverse)throw std::runtime_error("THC FFTW planning failed");
    _impl->entries=entries;_impl->shape=shape;
  }
  fftw_execute_dft(backwards?_impl->inverse:_impl->forward,data,data);
  return true;
#else
  return false;
#endif
}
}
