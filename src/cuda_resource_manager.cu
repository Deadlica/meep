#include "cuda_resource_manager.cuh"

namespace meep {
namespace cuda {

CudaResourceManager::CudaResourceManager():
initialized(false),
async_enabled(false),
d_f_updated(false),
d_f_u_updated(false),
d_f_cond_updated(false),
d_f_bfast_updated(false),
current_cc(Ex),
current_cmp(0) {}

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

void CudaResourceManager::init_async_resources(fields_chunk* chunk) {
  if (!initialized || async_enabled) return;

  size_t ntot = chunk->gv.ntot();

  // Create CUDA streams
  checkCudaErrors(cudaStreamCreate(&compute_stream));
  checkCudaErrors(cudaStreamCreate(&h2d_stream));
  checkCudaErrors(cudaStreamCreate(&d2h_stream));

  // Allocate pinned memory
  checkCudaErrors(cudaMallocHost(&h_f_pinned,       ntot * sizeof(realnum)));
  checkCudaErrors(cudaMallocHost(&h_f_u_pinned,     ntot * sizeof(realnum)));
  checkCudaErrors(cudaMallocHost(&h_f_cond_pinned,  ntot * sizeof(realnum)));
  checkCudaErrors(cudaMallocHost(&h_f_bfast_pinned, ntot * sizeof(realnum)));
  checkCudaErrors(cudaMallocHost(&h_g1_pinned,      ntot * sizeof(realnum)));
  checkCudaErrors(cudaMallocHost(&h_g2_pinned,      ntot * sizeof(realnum)));

  async_enabled = true;
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

  if (async_enabled) {
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(h2d_stream);
    cudaStreamDestroy(d2h_stream);

    cudaFreeHost(h_f_pinned);
    cudaFreeHost(h_f_u_pinned);
    cudaFreeHost(h_f_cond_pinned);
    cudaFreeHost(h_f_bfast_pinned);
    cudaFreeHost(h_g1_pinned);
    cudaFreeHost(h_g2_pinned);

    async_enabled = false;
  }

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

void CudaResourceManager::sync_to_device_async(fields_chunk* chunk, component cc, int cmp,
                                              direction dsig, direction dsigu, direction d_c,
                                              realnum* f_p, realnum* f_m) {
  if (!initialized) return;

  current_cc = cc;
  current_cmp = cmp;

  size_t ntot = chunk->gv.ntot();

  if (!async_enabled) {
    sync_to_device(chunk, cc, cmp, dsig, dsigu, d_c, f_p, f_m);
    return;
  }

  if (chunk->f[cc][cmp]) {
    memcpy(h_f_pinned, chunk->f[cc][cmp], ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_f, h_f_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  if (chunk->f_u[cc][cmp]) {
    memcpy(h_f_u_pinned, chunk->f_u[cc][cmp], ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_f_u, h_f_u_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  if (chunk->f_cond[cc][cmp]) {
    memcpy(h_f_cond_pinned, chunk->f_cond[cc][cmp], ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_f_cond, h_f_cond_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  if (chunk->f_bfast[cc][cmp]) {
    memcpy(h_f_bfast_pinned, chunk->f_bfast[cc][cmp], ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_f_bfast, h_f_bfast_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  if (f_p) {
    memcpy(h_g1_pinned, f_p, ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_g1, h_g1_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  if (f_m) {
    memcpy(h_g2_pinned, f_m, ntot * sizeof(realnum));
    checkCudaErrors(cudaMemcpyAsync(d_g2, h_g2_pinned, ntot * sizeof(realnum),
                                  cudaMemcpyHostToDevice, h2d_stream));
  }

  static component last_cc = Ex;
  static int last_cmp = -1;

  if (cc != last_cc || cmp != last_cmp) {
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

    last_cc = cc;
    last_cmp = cmp;
  }

  checkCudaErrors(cudaStreamSynchronize(h2d_stream));
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

void CudaResourceManager::sync_from_device_async(fields_chunk* chunk, component cc, int cmp) {
  if (!initialized) return;

  size_t ntot = chunk->gv.ntot();

  if (!async_enabled) {
    sync_from_device(chunk, cc, cmp);
    return;
  }

  checkCudaErrors(cudaStreamSynchronize(compute_stream));

  if (d_f_updated && chunk->f[cc][cmp]) {
    checkCudaErrors(cudaMemcpyAsync(h_f_pinned, d_f, ntot * sizeof(realnum),
                                  cudaMemcpyDeviceToHost, d2h_stream));
  }

  if (d_f_u_updated && chunk->f_u[cc][cmp]) {
    checkCudaErrors(cudaMemcpyAsync(h_f_u_pinned, d_f_u, ntot * sizeof(realnum),
                                  cudaMemcpyDeviceToHost, d2h_stream));
  }

  if (d_f_cond_updated && chunk->f_cond[cc][cmp]) {
    checkCudaErrors(cudaMemcpyAsync(h_f_cond_pinned, d_f_cond, ntot * sizeof(realnum),
                                  cudaMemcpyDeviceToHost, d2h_stream));
  }

  if (d_f_bfast_updated && chunk->f_bfast[cc][cmp]) {
    checkCudaErrors(cudaMemcpyAsync(h_f_bfast_pinned, d_f_bfast, ntot * sizeof(realnum),
                                  cudaMemcpyDeviceToHost, d2h_stream));
  }
}

void CudaResourceManager::wait_for_transfers(fields_chunk* chunk, component cc, int cmp) {
  if (!initialized || !async_enabled) return;

  checkCudaErrors(cudaStreamSynchronize(d2h_stream));

  size_t ntot = chunk->gv.ntot();

  if (d_f_updated && chunk->f[cc][cmp]) {
    memcpy(chunk->f[cc][cmp], h_f_pinned, ntot * sizeof(realnum));
    d_f_updated = false;
  }

  if (d_f_u_updated && chunk->f_u[cc][cmp]) {
    memcpy(chunk->f_u[cc][cmp], h_f_u_pinned, ntot * sizeof(realnum));
    d_f_u_updated = false;
  }

  if (d_f_cond_updated && chunk->f_cond[cc][cmp]) {
    memcpy(chunk->f_cond[cc][cmp], h_f_cond_pinned, ntot * sizeof(realnum));
    d_f_cond_updated = false;
  }

  if (d_f_bfast_updated && chunk->f_bfast[cc][cmp]) {
    memcpy(chunk->f_bfast[cc][cmp], h_f_bfast_pinned, ntot * sizeof(realnum));
    d_f_bfast_updated = false;
  }
}

}
}
