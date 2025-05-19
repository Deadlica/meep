#ifndef MEEP_CUDA_RESOURCE_MANAGER
#define MEEP_CUDA_RESOURCE_MANAGER

#include "meep.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace meep {
namespace cuda {

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

class CudaResourceManager {
public:
  CudaResourceManager();
  ~CudaResourceManager();

  void init(fields_chunk* chunk);
  void init_async_resources(fields_chunk* chunk);
  void free_resources();

  void sync_to_device(fields_chunk* chunk, component cc, int cmp,
                      direction dsig, direction dsigu, direction d_c,
                      realnum* f_p, realnum* f_m);
  void sync_to_device_async(fields_chunk* chunk, component cc, int cmp,
                           direction dsig, direction dsigu, direction d_c,
                           realnum* f_p, realnum* f_m);
  void sync_from_device(fields_chunk* chunk, component cc, int cmp);
  void sync_from_device_async(fields_chunk* chunk, component cc, int cmp);

  void wait_for_transfers(fields_chunk* chunk, component cc, int cmp);

  bool initialized;

  // Read/Write
  realnum* d_f;
  realnum* d_f_u;
  realnum* d_f_cond;
  realnum* d_f_bfast;

  bool d_f_updated;
  bool d_f_u_updated;
  bool d_f_cond_updated;
  bool d_f_bfast_updated;

  // Readonly
  realnum* d_g1;
  realnum* d_g2;

  realnum* d_sig;
  realnum* d_kap;
  realnum* d_siginv;
  realnum* d_sigu;
  realnum* d_kapu;
  realnum* d_siginvu;

  realnum* d_conductivity;
  realnum* d_condinv;

  // Async resources
  bool async_enabled;

  cudaStream_t compute_stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;

  realnum* h_f_pinned;
  realnum* h_f_u_pinned;
  realnum* h_f_cond_pinned;
  realnum* h_f_bfast_pinned;
  realnum* h_g1_pinned;
  realnum* h_g2_pinned;

  component current_cc;
  int current_cmp;
};

} // namespace cuda
} // namespace meep

#endif // MEEP_CUDA_RESOURCE_MANAGER
