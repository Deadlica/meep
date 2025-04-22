#include "cuda_resource_manager.cuh"

namespace meep {
namespace cuda {

CudaResourceManager::CudaResourceManager():
initialized(false),
d_f_updated(false),
d_f_u_updated(false),
d_f_cond_updated(false),
d_f_bfast_updated(false) {}

CudaResourceManager::~CudaResourceManager() {
  free_resources();
}

void CudaResourceManager::init(fields_chunk* chunk) {
  if (initialized) return;

  size_t ntot = chunk->gv.ntot();

  // Allocate
  checkCudaErrors(cudaMalloc(&d_f,             ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_f_u,           ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_f_cond,        ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_f_bfast,       ntot * sizeof(realnum)));

  checkCudaErrors(cudaMalloc(&d_g1,            ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_g2,            ntot * sizeof(realnum)));

  checkCudaErrors(cudaMalloc(&d_sig,           ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_kap,           ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_siginv,        ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_sigu,          ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_kapu,          ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_siginvu,       ntot * sizeof(realnum)));

  checkCudaErrors(cudaMalloc(&d_conductivity,  ntot * sizeof(realnum)));
  checkCudaErrors(cudaMalloc(&d_condinv,       ntot * sizeof(realnum)));

  // Initialize
  checkCudaErrors(cudaMemset(d_f,             0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_f_u,           0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_f_cond,        0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_f_bfast,       0, ntot * sizeof(realnum)));

  checkCudaErrors(cudaMemset(d_g1,            0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_g2,            0, ntot * sizeof(realnum)));

  checkCudaErrors(cudaMemset(d_sig,           0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_kap,           0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_siginv,        0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_sigu,          0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_kapu,          0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_siginvu,       0, ntot * sizeof(realnum)));

  checkCudaErrors(cudaMemset(d_conductivity,  0, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMemset(d_condinv,       0, ntot * sizeof(realnum)));

  initialized = true;
}

void CudaResourceManager::free_resources() {
  if (!initialized) return;

  cudaFree(d_f);
  cudaFree(d_f_u);
  cudaFree(d_f_cond);
  cudaFree(d_f_bfast);
  cudaFree(d_g1);
  cudaFree(d_g2);
  cudaFree(d_sig);
  cudaFree(d_kap);
  cudaFree(d_siginv);
  cudaFree(d_sigu);
  cudaFree(d_kapu);
  cudaFree(d_siginvu);
  cudaFree(d_conductivity);
  cudaFree(d_condinv);

  initialized = false;
}

void CudaResourceManager::sync_to_device(fields_chunk* chunk, component cc, int cmp,
                                         direction dsig, direction dsigu, direction d_c,
                                         realnum* f_p, realnum* f_m) {
  if (!initialized) return;

  size_t ntot = chunk->gv.ntot();

  if (chunk->f[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(d_f, chunk->f[cc][cmp], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (chunk->f_u[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(d_f_u, chunk->f_u[cc][cmp], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (chunk->f_cond[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(d_f_cond, chunk->f_cond[cc][cmp], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (chunk->f_bfast[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(d_f_bfast, chunk->f_bfast[cc][cmp], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }

  if (f_p) {
    checkCudaErrors(cudaMemcpy(d_g1, f_p, ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (f_m) {
    checkCudaErrors(cudaMemcpy(d_g2, f_m, ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }

  if (dsig != NO_DIRECTION) {
    checkCudaErrors(cudaMemcpy(d_sig,     chunk->s->sig[dsig],     ntot * sizeof(realnum), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kap,     chunk->s->kap[dsig],     ntot * sizeof(realnum), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_siginv,  chunk->s->siginv[dsig],  ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (dsigu != NO_DIRECTION) {
    checkCudaErrors(cudaMemcpy(d_sigu,    chunk->s->sig[dsigu],    ntot * sizeof(realnum), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kapu,    chunk->s->kap[dsigu],    ntot * sizeof(realnum), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_siginvu, chunk->s->siginv[dsigu], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }

  if (chunk->s->conductivity[cc][d_c]) {
    checkCudaErrors(cudaMemcpy(d_conductivity, chunk->s->conductivity[cc][d_c], ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
  if (chunk->s->condinv[cc][d_c]) {
    checkCudaErrors(cudaMemcpy(d_condinv,      chunk->s->condinv[cc][d_c],      ntot * sizeof(realnum), cudaMemcpyHostToDevice));
  }
}

void CudaResourceManager::sync_from_device(fields_chunk* chunk, component cc, int cmp) {
  if (!initialized) return;

  size_t ntot = chunk->gv.ntot();

  if (d_f_updated && chunk->f[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(chunk->f[cc][cmp],       d_f,       ntot * sizeof(realnum), cudaMemcpyDeviceToHost));
    d_f_updated = false;
  }
  if (d_f_u_updated && chunk->f_u[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(chunk->f_u[cc][cmp],     d_f_u,     ntot * sizeof(realnum), cudaMemcpyDeviceToHost));
    d_f_u_updated = false;
  }
  if (d_f_cond_updated && chunk->f_cond[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(chunk->f_cond[cc][cmp],  d_f_cond,  ntot * sizeof(realnum), cudaMemcpyDeviceToHost));
    d_f_cond_updated = false;
  }
  if (d_f_bfast_updated && chunk->f_bfast[cc][cmp]) {
    checkCudaErrors(cudaMemcpy(chunk->f_bfast[cc][cmp], d_f_bfast, ntot * sizeof(realnum), cudaMemcpyDeviceToHost));
    d_f_bfast_updated = false;
  }
}

}
}
