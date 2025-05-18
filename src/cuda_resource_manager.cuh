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
  void free_resources();

  void sync_to_device(fields_chunk* chunk, component cc, int cmp,
                      direction dsig, direction dsigu, direction d_c,
                      realnum* f_p, realnum* f_m);
  void sync_from_device(fields_chunk* chunk, component cc, int cmp);

//private:
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
};

} // namespace cuda
} // namespace meep

#endif // MEEP_CUDA_RESOURCE_MANAGER
