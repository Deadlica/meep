#include "meep.hpp"
#include "meep_internals.hpp"
#include "cuda_resource_manager.cuh"
#include <cuda_runtime.h>

#define RPR realnum*

namespace meep {

#define SWAP(t, a, b)                                                                            \
{                                                                                                \
  t xxxx = a;                                                                                    \
  a = b;                                                                                         \
  b = xxxx;                                                                                      \
}

namespace cuda {

const int CUDA_X = 8;
const int CUDA_Y = 8;
const int CUDA_Z = 8;

__global__ void check_nan_inf_kernel(const realnum* arr, int size, int* has_error) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (isnan(arr[idx]) || isinf(arr[idx])) {
      atomicAdd(has_error, 1);
    }
  }
}

void check_nan_inf(const char* arr_name, const realnum* arr, int size, dim3 dg, dim3 db) {
  int* d_has_error;
  int has_error = 0;
  checkCudaErrors(cudaMalloc(&d_has_error, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_has_error, &has_error, sizeof(int), cudaMemcpyHostToDevice));

  check_nan_inf_kernel<<<dg, db>>>(arr, size, d_has_error);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&has_error, d_has_error, sizeof(int), cudaMemcpyDeviceToHost));
  if (has_error) {
    printf("WARNING: NaN/Inf detected in %s array (%d instances)\n", arr_name, has_error);
  }
  checkCudaErrors(cudaFree(d_has_error));
}

// step_curl kernels

// Kernel for the case: no PML, no fu, with conductivity, with g2
__global__ void step_curl_kernel1(realnum* f, const realnum* g1, const realnum* g2, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  f[i] = ((1 - dt2 * cnd[i]) * f[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) * cndinv[i];
}

// Kernel for the case: no PML, no fu, with conductivity, no g2
__global__ void step_curl_kernel2(realnum* f, const realnum* g1, ptrdiff_t s1, realnum dtdx,
                                  realnum dt, const realnum* cnd, const realnum* cndinv,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  f[i] = ((1 - dt2 * cnd[i]) * f[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
}

// Kernel for the case: no PML, no fu, no conductivity, with g2
__global__ void step_curl_kernel3(realnum* f, const realnum* g1, const realnum* g2, ptrdiff_t s1, ptrdiff_t s2, realnum dtdx, ptrdiff_t loop_n1,
                                  ptrdiff_t loop_n2, ptrdiff_t loop_n3, ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  f[i] -= dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2]);
}

// Kernel for the case: no PML, no fu, no conductivity, no g2
__global__ void step_curl_kernel4(realnum* f, const realnum* g1, ptrdiff_t s1, realnum dtdx, ptrdiff_t loop_n1, ptrdiff_t loop_n2,
                                  ptrdiff_t loop_n3, ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  f[i] -= dtdx * (g1[i + s1] - g1[i]);
}

// Kernel for the case: no PML, with fu, with conductivity, with g2
__global__ void step_curl_kernel5(realnum* f, realnum* fu, const realnum* g1, const realnum* g2, const realnum* sigu,
                                  const realnum* kapu, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int ku0, int sku1,
                                  int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] =
      ((1 - dt2 * cnd[i]) * fprev - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
      cndinv[i];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: no PML, with fu, with conductivity, no g2
__global__ void step_curl_kernel6(realnum* f, realnum* fu, const realnum* g1, const realnum* sigu,
                                  const realnum* kapu, const realnum* siginvu, ptrdiff_t s1,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int ku0, int sku1,
                                  int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] = ((1 - dt2 * cnd[i]) * fprev - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: no PML, with fu, no conductivity, with g2
__global__ void step_curl_kernel7(realnum* f, realnum* fu, const realnum* g1, const realnum* g2, const realnum* sigu,
                                  const realnum* kapu, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, int ku0, int sku1, int sku2, int sku3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] -= dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2]);
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: no PML, with fu, no conductivity, no g2
__global__ void step_curl_kernel8(realnum* f, realnum* fu, const realnum* g1, const realnum* sigu,
                                  const realnum* kapu, const realnum* siginvu, ptrdiff_t s1,
                                  realnum dtdx, int ku0, int sku1, int sku2, int sku3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] -= dtdx * (g1[i + s1] - g1[i]);
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: with PML, no fu, with conductivity, with g2
__global__ void step_curl_kernel9(realnum* f, realnum* fcnd, const realnum* g1, const realnum* g2, const realnum* sig,
                                  const realnum* kap, const realnum* siginv, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int k0, int sk1,
                                  int sk2, int sk3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum fcnd_prev = fcnd[i];
  fcnd[i] =
      ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
      cndinv[i];
  f[i] = ((kap[k] - sig[k]) * f[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
}

// Kernel for the case: with PML, no fu, with conductivity, no g2
__global__ void step_curl_kernel10(realnum* f, realnum* fcnd, const realnum* g1, const realnum* sig,
                                  const realnum* kap, const realnum* siginv, ptrdiff_t s1,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int k0, int sk1,
                                  int sk2, int sk3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum fcnd_prev = fcnd[i];
  fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
  f[i] = ((kap[k] - sig[k]) * f[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
}

// Kernel for the case: with PML, no fu, no conductivity, with g2
__global__ void step_curl_kernel11(realnum* f, const realnum* g1, const realnum* g2, const realnum* sig,
                                  const realnum* kap, const realnum* siginv, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, int k0, int sk1, int sk2, int sk3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  f[i] = ((kap[k] - sig[k]) * f[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
         siginv[k];
}

// Kernel for the case: with PML, no fu, no conductivity, no g2
__global__ void step_curl_kernel12(realnum* f, const realnum* g1, const realnum* sig,
                                  const realnum* kap, const realnum* siginv, ptrdiff_t s1,
                                  realnum dtdx, int k0, int sk1, int sk2, int sk3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  f[i] = ((kap[k] - sig[k]) * f[i] - dtdx * (g1[i + s1] - g1[i])) * siginv[k];
}

// Kernel for the case: with PML, with fu, with conductivity, with g2
__global__ void step_curl_kernel13(realnum* f, realnum* fu, realnum* fcnd, const realnum* g1, const realnum* g2,
                                  const realnum* sig, const realnum* kap, const realnum* siginv,
                                  const realnum* sigu, const realnum* kapu, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int k0, int sk1,
                                  int sk2, int sk3, int ku0, int sku1, int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  realnum fcnd_prev = fcnd[i];
  fcnd[i] =
      ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
      cndinv[i];
  fu[i] = ((kap[k] - sig[k]) * fu[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: with PML, with fu, with conductivity, no g2
__global__ void step_curl_kernel14(realnum* f, realnum* fu, realnum* fcnd, const realnum* g1,
                                  const realnum* sig, const realnum* kap, const realnum* siginv,
                                  const realnum* sigu, const realnum* kapu, const realnum* siginvu, ptrdiff_t s1,
                                  realnum dtdx, realnum dt, const realnum* cnd, const realnum* cndinv, int k0, int sk1,
                                  int sk2, int sk3, int ku0, int sku1, int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum dt2 = dt * 0.5;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  realnum fcnd_prev = fcnd[i];
  fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
  fu[i] = ((kap[k] - sig[k]) * fu[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: with PML, with fu, no conductivity, with g2
__global__ void step_curl_kernel15(realnum* f, realnum* fu, const realnum* g1, const realnum* g2,
                                  const realnum* sig, const realnum* kap, const realnum* siginv,
                                  const realnum* sigu, const realnum* kapu, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                  realnum dtdx, int k0, int sk1, int sk2, int sk3, int ku0, int sku1, int sku2, int sku3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] = ((kap[k] - sig[k]) * fu[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
          siginv[k];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// Kernel for the case: with PML, with fu, no conductivity, no g2
__global__ void step_curl_kernel16(realnum* f, realnum* fu, const realnum* g1,
                                  const realnum* sig, const realnum* kap, const realnum* siginv,
                                  const realnum* sigu, const realnum* kapu, const realnum* siginvu, ptrdiff_t s1,
                                  realnum dtdx, int k0, int sk1, int sk2, int sk3, int ku0, int sku1, int sku2, int sku3,
                                  ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                  ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum fprev = fu[i];
  fu[i] = ((kap[k] - sig[k]) * fu[i] - dtdx * (g1[i + s1] - g1[i])) * siginv[k];
  f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
}

// step_bfast kernels

// Kernel for the case: no PML, no fu, with conductivity, with g2
__global__ void step_bfast_kernel1(realnum* f, realnum* F, const realnum* g1, const realnum* g2,
                                   ptrdiff_t s1, ptrdiff_t s2, realnum k1, realnum k2, const realnum* cndinv,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  f[i] += (F[i] - F_prev);
}

// Kernel for the case: no PML, no fu, with conductivity, no g2
__global__ void step_bfast_kernel2(realnum* f, realnum* F, const realnum* g1,
                                   ptrdiff_t s1, realnum k1, const realnum* cndinv,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  f[i] += (F[i] - F_prev) * cndinv[i];
}

// Kernel for the case: no PML, no fu, no conductivity, with g2
__global__ void step_bfast_kernel3(realnum* f, realnum* F, const realnum* g1, const realnum* g2,
                                   ptrdiff_t s1, ptrdiff_t s2, realnum k1, realnum k2,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  f[i] += (F[i] - F_prev);
}

// Kernel for the case: no PML, no fu, no conductivity, no g2
__global__ void step_bfast_kernel4(realnum* f, realnum* F, const realnum* g1, ptrdiff_t s1,
                                   realnum k1, ptrdiff_t loop_n1, ptrdiff_t loop_n2,
                                   ptrdiff_t loop_n3, ptrdiff_t loop_s1, ptrdiff_t loop_s2,
                                   ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]);
  f[i] += (F[i] - F_prev);
}

// Kernel for the case: no PML, with fu, with conductivity, with g2
__global__ void step_bfast_kernel5(realnum* f, realnum* fu, realnum* F, const realnum* g1, const realnum* g2,
                                   const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2, realnum k1, realnum k2,
                                   const realnum* cndinv, int ku0, int sku1, int sku2, int sku3,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  fu[i] += (df = (F[i] - F_prev) * cndinv[i]);
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: no PML, with fu, with conductivity, no g2
__global__ void step_bfast_kernel6(realnum* f, realnum* fu, realnum* F, const realnum* g1, const realnum* siginvu,
                                   ptrdiff_t s1, realnum k1, const realnum* cndinv, int ku0, int sku1,
                                   int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  fu[i] += (df = (F[i] - F_prev) * cndinv[i]);
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: no PML, with fu, no conductivity, with g2
__global__ void step_bfast_kernel7(realnum* f, realnum* fu, realnum* F, const realnum* g1, const realnum* g2,
                                   const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2, realnum k1, realnum k2,
                                   int ku0, int sku1, int sku2, int sku3,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  fu[i] += (df = (F[i] - F_prev));
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: no PML, with fu, no conductivity, no g2
__global__ void step_bfast_kernel8(realnum* f, realnum* fu, realnum* F, const realnum* g1,
                                   const realnum* siginvu, ptrdiff_t s1, realnum k1,
                                   int ku0, int sku1, int sku2, int sku3,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  fu[i] += (df = (F[i] - F_prev));
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: with PML, no fu, with conductivity, with g2
__global__ void step_bfast_kernel9(realnum* f, realnum* fcnd, realnum* F, const realnum* g1, const realnum* g2,
                                   const realnum* siginv, ptrdiff_t s1, ptrdiff_t s2, realnum k1, realnum k2,
                                   const realnum* cndinv, int k0, int sk1, int sk2, int sk3,
                                   ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                   ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  realnum dfcnd = (F[i] - F_prev) * cndinv[i];
  fcnd[i] += dfcnd;
  f[i] += dfcnd * siginv[k];
}

// Kernel for the case: with PML, no fu, with conductivity, no g2
__global__ void step_bfast_kernel10(realnum* f, realnum* fcnd, realnum* F, const realnum* g1,
                                    const realnum* siginv, ptrdiff_t s1, realnum k1,
                                    const realnum* cndinv, int k0, int sk1, int sk2, int sk3,
                                    ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  realnum dfcnd = (F[i] - F_prev) * cndinv[i];
  fcnd[i] += dfcnd;
  f[i] += dfcnd * siginv[k];
}

// Kernel for the case: with PML, no fu, no conductivity, with g2
__global__ void step_bfast_kernel11(realnum* f, realnum* F, const realnum* g1, const realnum* g2,
                                    const realnum* siginv, ptrdiff_t s1, ptrdiff_t s2,
                                    realnum k1, realnum k2, int k0, int sk1, int sk2, int sk3,
                                    ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  f[i] += (F[i] - F_prev) * siginv[k];
}

// Kernel for the case: with PML, no fu, no conductivity, no g2
__global__ void step_bfast_kernel12(realnum* f, realnum* F, const realnum* g1, const realnum* siginv,
                                    ptrdiff_t s1, realnum k1, int k0, int sk1, int sk2, int sk3,
                                    ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  f[i] += (F[i] - F_prev) * siginv[k];
}

// Kernel for the case: with PML, with fu, with conductivity, with g2
__global__ void step_bfast_kernel13(realnum* f, realnum* fu, realnum* fcnd, realnum* F, const realnum* g1, const realnum* g2,
                                    const realnum* siginv, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                    realnum k1, realnum k2, const realnum* cndinv, int k0, int sk1,
                                    int sk2, int sk3, int ku0, int sku1, int sku2, int sku3,
                                    ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  realnum dfcnd = (F[i] - F_prev) * cndinv[i];
  fcnd[i] += dfcnd;
  fu[i] += (df = dfcnd * siginv[k]);
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: with PML, with fu, with conductivity, no g2
__global__ void step_bfast_kernel14(realnum* f, realnum* fu, realnum* fcnd, realnum* F, const realnum* g1,
                                    const realnum* siginv, const realnum* siginvu, ptrdiff_t s1, realnum k1,
                                    const realnum* cndinv, int k0, int sk1, int sk2, int sk3, int ku0, int sku1,
                                    int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  realnum dfcnd = (F[i] - F_prev) * cndinv[i];
  fcnd[i] += dfcnd;
  fu[i] += (df = dfcnd * siginv[k]);
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: with PML, with fu, no conductivity, with g2
__global__ void step_bfast_kernel15(realnum* f, realnum* fu, realnum* F, const realnum* g1, const realnum* g2,
                                    const realnum* siginv, const realnum* siginvu, ptrdiff_t s1, ptrdiff_t s2,
                                    realnum k1, realnum k2, int k0, int sk1, int sk2, int sk3, int ku0, int sku1,
                                    int sku2, int sku3, ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - F[i];
  fu[i] += (df = (F[i] - F_prev) * siginv[k]);
  f[i] += siginvu[ku] * df;
}

// Kernel for the case: with PML, with fu, no conductivity, no g2
__global__ void step_bfast_kernel16(realnum* f, realnum* fu, realnum* F, const realnum* g1,
                                    const realnum* siginv, const realnum* siginvu, ptrdiff_t s1,
                                    realnum k1, int k0, int sk1, int sk2, int sk3,
                                    int ku0, int sku1, int sku2, int sku3,
                                    ptrdiff_t loop_n1, ptrdiff_t loop_n2, ptrdiff_t loop_n3,
                                    ptrdiff_t loop_s1, ptrdiff_t loop_s2, ptrdiff_t loop_s3, ptrdiff_t idx0) {
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = blockIdx.z * blockDim.z + threadIdx.z;

  if (i1 >= loop_n1 || i2 >= loop_n2 || i3 >= loop_n3) return;

  ptrdiff_t i = idx0 + i1 * loop_s1 + i2 * loop_s2 + i3 * loop_s3;
  const int k = ((k0 + sk1 * i1) + sk2 * i2) + sk3 * i3;
  const int ku = ((ku0 + sku1 * i1) + sku2 * i2) + sku3 * i3;
  realnum df;
  realnum F_prev = F[i];
  F[i] = k1 * (g1[i + s1] + g1[i]) - F[i];
  fu[i] += (df = (F[i] - F_prev) * siginv[k]);
  f[i] += siginvu[ku] * df;
}

void cuda_step_curl(RPR f, component c, const RPR g1, const RPR g2, ptrdiff_t s1,
                    ptrdiff_t s2, const grid_volume &gv, const ivec is, const ivec ie, realnum dtdx, direction dsig,
                    const RPR sig, const RPR kap, const RPR siginv, RPR fu, direction dsigu,
                    const RPR sigu, const RPR kapu, const RPR siginvu, realnum dt, const RPR cnd,
                    const RPR cndinv, RPR fcnd, CudaResourceManager* cuda_resources) {
  (void)c;   // currently unused
  if (!g1) { // swap g1 and g2
    SWAP(const RPR, g1, g2);
    SWAP(RPR, cuda_resources->d_g1, cuda_resources->d_g2);
    SWAP(ptrdiff_t, s1, s2);
    dtdx = -dtdx; // need to flip derivative sign
  }

  ptrdiff_t loop_is1 = is.yucky_val(0);
  ptrdiff_t loop_is2 = is.yucky_val(1);
  ptrdiff_t loop_is3 = is.yucky_val(2);
  ptrdiff_t loop_n1 = (ie.yucky_val(0) - loop_is1) / 2 + 1;
  ptrdiff_t loop_n2 = (ie.yucky_val(1) - loop_is2) / 2 + 1;
  ptrdiff_t loop_n3 = (ie.yucky_val(2) - loop_is3) / 2 + 1;
  ptrdiff_t loop_d1 = gv.yucky_direction(0);
  ptrdiff_t loop_d2 = gv.yucky_direction(1);
  ptrdiff_t loop_d3 = gv.yucky_direction(2);
  ptrdiff_t loop_s1 = gv.stride((direction)loop_d1);
  ptrdiff_t loop_s2 = gv.stride((direction)loop_d2);
  ptrdiff_t loop_s3 = gv.stride((direction)loop_d3);
  ptrdiff_t idx0 = (is - gv.little_corner()).yucky_val(0) / 2 * loop_s1 +
                    (is - gv.little_corner()).yucky_val(1) / 2 * loop_s2 +
                    (is - gv.little_corner()).yucky_val(2) / 2 * loop_s3;

  dim3 db(CUDA_X, CUDA_Y, CUDA_Z);
  dim3 dg(
    (loop_n1 + db.x - 1) / db.x,
    (loop_n2 + db.y - 1) / db.y,
    (loop_n3 + db.z - 1) / db.z
  );

  /* The following are a bunch of special cases of the "MOST GENERAL CASE"
  loop below.  We make copies of the loop for each special case in
  order to keep the innermost loop efficient.  This is especially
  important because the non-PML cases are actually more common.
  (The "right" way to do this is by partial evaluation of the
  most general case, but that would require a code generator.) */

  if (dsig == NO_DIRECTION) {    // no PML in f update
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        if (g2) {
          step_curl_kernel1<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1, cuda_resources->d_g2,
            s1, s2, dtdx, dt, cuda_resources->d_conductivity, cuda_resources->d_condinv,
            loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel2<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1, s1, dtdx, dt,
            cuda_resources->d_conductivity, cuda_resources->d_condinv,
            loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      else { // no conductivity
        if (g2) {
          step_curl_kernel3<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1, cuda_resources->d_g2,
            s1, s2, dtdx, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel4<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1,
            s1, dtdx, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
    }
    else { // fu update, no PML in f update
      KSTRIDE_DEF(dsigu, ku, is, gv);
      if (cnd) {
        if (g2) {
          step_curl_kernel5<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_g2, cuda_resources->d_sigu, cuda_resources->d_kapu,
            cuda_resources->d_siginvu, s1, s2, dtdx, dt,
            cuda_resources->d_conductivity, cuda_resources->d_condinv, ku0,
            sku1, sku2, sku3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel6<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_sigu, cuda_resources->d_kapu, cuda_resources->d_siginvu,
            s1, dtdx, dt, cuda_resources->d_conductivity, cuda_resources->d_condinv,
            ku0, sku1, sku2, sku3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      else { // no conductivity
        if (g2) {
          step_curl_kernel7<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_g2, cuda_resources->d_sigu, cuda_resources->d_kapu,
            cuda_resources->d_siginvu, s1, s2, dtdx, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel8<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_sigu, cuda_resources->d_kapu, cuda_resources->d_siginvu,
            s1, dtdx, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      cuda_resources->d_f_u_updated = true;
    }
  }
  else { /* PML in f update */
    KSTRIDE_DEF(dsig, k, is, gv);
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        realnum dt2 = dt * 0.5;
        if (g2) {
          step_curl_kernel9<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_cond, cuda_resources->d_g1,
            cuda_resources->d_g2, cuda_resources->d_sig, cuda_resources->d_kap,
            cuda_resources->d_siginv, s1, s2, dtdx, dt,
            cuda_resources->d_conductivity, cuda_resources->d_condinv, k0,
            sk1, sk2, sk3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel10<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_cond, cuda_resources->d_g1,
            cuda_resources->d_sig, cuda_resources->d_kap, cuda_resources->d_siginv,
            s1, dtdx, dt, cuda_resources->d_conductivity, cuda_resources->d_condinv,
            k0, sk1, sk2, sk3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        cuda_resources->d_f_cond_updated = true;
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          step_curl_kernel11<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1, cuda_resources->d_g2,
            cuda_resources->d_sig, cuda_resources->d_kap, cuda_resources->d_siginv,
            s1, s2, dtdx, k0, sk1, sk2, sk3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel12<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_g1,
            cuda_resources->d_sig, cuda_resources->d_kap, cuda_resources->d_siginv,
            s1, dtdx, k0, sk1, sk2, sk3, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
    }
    else { // fu update + PML in f update
      KSTRIDE_DEF(dsigu, ku, is, gv);
      if (cnd) {
        if (g2) {
          //////////////////// MOST GENERAL CASE //////////////////////
          step_curl_kernel13<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_cond,
            cuda_resources->d_g1, cuda_resources->d_g2, cuda_resources->d_sig,
            cuda_resources->d_kap, cuda_resources->d_siginv, cuda_resources->d_sigu,
            cuda_resources->d_kapu, cuda_resources->d_siginvu, s1, s2, dtdx, dt,
            cuda_resources->d_conductivity, cuda_resources->d_condinv,
            k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
          /////////////////////////////////////////////////////////////
        }
        else {
          step_curl_kernel14<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_cond,
            cuda_resources->d_g1, cuda_resources->d_sig, cuda_resources->d_kap,
            cuda_resources->d_siginv, cuda_resources->d_sigu, cuda_resources->d_kapu,
            cuda_resources->d_siginvu, s1, dtdx, dt, cuda_resources->d_conductivity,
            cuda_resources->d_condinv, k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        cuda_resources->d_f_cond_updated = true;
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          step_curl_kernel15<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_g2, cuda_resources->d_sig, cuda_resources->d_kap,
            cuda_resources->d_siginv, cuda_resources->d_sigu, cuda_resources->d_kapu,
            cuda_resources->d_siginvu, s1, s2, dtdx, k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_curl_kernel16<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_g1,
            cuda_resources->d_sig, cuda_resources->d_kap, cuda_resources->d_siginv,
            cuda_resources->d_sigu, cuda_resources->d_kapu, cuda_resources->d_siginvu,
            s1, dtdx, k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      cuda_resources->d_f_u_updated = true;
    }
  }

  cuda_resources->d_f_updated = true;
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void cuda_step_curl_stride1(RPR f, component c, const RPR g1, const RPR g2, ptrdiff_t s1,
                            ptrdiff_t s2, const grid_volume &gv, const ivec is, const ivec ie, realnum dtdx, direction dsig,
                            const RPR sig, const RPR kap, const RPR siginv, RPR fu, direction dsigu,
                            const RPR sigu, const RPR kapu, const RPR siginvu, realnum dt, const RPR cnd,
                            const RPR cndinv, RPR fcnd, CudaResourceManager* cuda_resources) {
  cuda_step_curl(f, c, g1, g2, s1, s2, gv, is, ie, dtdx, dsig, sig, kap, siginv,
                 fu, dsigu, sigu, kapu, siginvu, dt, cnd, cndinv, fcnd, cuda_resources);
}

void cuda_step_bfast(RPR f, component c, const RPR g1, const RPR g2,
                     ptrdiff_t s1, ptrdiff_t s2,
                     const grid_volume &gv, const ivec is, const ivec ie, realnum dtdx, direction dsig,
                     const RPR sig, const RPR kap, const RPR siginv, RPR fu, direction dsigu,
                     const RPR sigu, const RPR kapu, const RPR siginvu, realnum dt, const RPR cnd,
                     const RPR cndinv, RPR fcnd, RPR F, realnum k1, realnum k2,
                     CudaResourceManager* cuda_resources) {
  (void)c;   // currently unused
  if (!g1) { // swap g1 and g2
    SWAP(const RPR, g1, g2);
    SWAP(RPR, cuda_resources->d_g1, cuda_resources->d_g2);
    SWAP(ptrdiff_t, s1, s2);
    SWAP(realnum, k1, k2); // need to swap in cross product
  }

  ptrdiff_t loop_is1 = is.yucky_val(0);
  ptrdiff_t loop_is2 = is.yucky_val(1);
  ptrdiff_t loop_is3 = is.yucky_val(2);
  ptrdiff_t loop_n1 = (ie.yucky_val(0) - loop_is1) / 2 + 1;
  ptrdiff_t loop_n2 = (ie.yucky_val(1) - loop_is2) / 2 + 1;
  ptrdiff_t loop_n3 = (ie.yucky_val(2) - loop_is3) / 2 + 1;
  ptrdiff_t loop_d1 = gv.yucky_direction(0);
  ptrdiff_t loop_d2 = gv.yucky_direction(1);
  ptrdiff_t loop_d3 = gv.yucky_direction(2);
  ptrdiff_t loop_s1 = gv.stride((direction)loop_d1);
  ptrdiff_t loop_s2 = gv.stride((direction)loop_d2);
  ptrdiff_t loop_s3 = gv.stride((direction)loop_d3);
  ptrdiff_t idx0 = (is - gv.little_corner()).yucky_val(0) / 2 * loop_s1 +
                   (is - gv.little_corner()).yucky_val(1) / 2 * loop_s2 +
                   (is - gv.little_corner()).yucky_val(2) / 2 * loop_s3;

  dim3 db(CUDA_X, CUDA_Y, CUDA_Z);
  dim3 dg(
    (loop_n1 + db.x - 1) / db.x,
    (loop_n2 + db.y - 1) / db.y,
    (loop_n3 + db.z - 1) / db.z
  );

  if (dsig == NO_DIRECTION) {    // no PML in f update
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        if (g2) {
          step_bfast_kernel1<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            cuda_resources->d_g2, s1, s2, k1, k2, cuda_resources->d_condinv,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel2<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            s1, k1, cuda_resources->d_condinv, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      else { // no conductivity
        if (g2) {
          step_bfast_kernel3<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            cuda_resources->d_g2, s1, s2, k1, k2, loop_n1, loop_n2, loop_n3,
            loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel4<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            s1, k1, loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
    }
    else { // fu update, no PML in f update
      KSTRIDE_DEF(dsigu, ku, is, gv);
      if (cnd) {
        if (g2) {
          step_bfast_kernel5<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_g2, cuda_resources->d_siginvu,
            s1, s2, k1, k2, cuda_resources->d_condinv, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel6<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_siginvu, s1, k1,
            cuda_resources->d_condinv, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      else { // no conductivity
        if (g2) {
          step_bfast_kernel7<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_g2, cuda_resources->d_siginvu,
            s1, s2, k1, k2, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel8<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_siginvu, s1, k1, ku0, sku1,
            sku2, sku3, loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      cuda_resources->d_f_u_updated = true;
    }
  }
  else { // PML in f update
    KSTRIDE_DEF(dsig, k, is, gv);
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        if (g2) {
          step_bfast_kernel9<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_cond, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_g2, cuda_resources->d_siginv,
            s1, s2, k1, k2, cuda_resources->d_condinv, k0, sk1, sk2, sk3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel10<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_cond, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_siginv,
            s1, k1, cuda_resources->d_condinv, k0, sk1, sk2, sk3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        cuda_resources->d_f_cond_updated = true;
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          step_bfast_kernel11<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            cuda_resources->d_g2, cuda_resources->d_siginv, s1, s2, k1, k2, k0,
            sk1, sk2, sk3, loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel12<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_bfast, cuda_resources->d_g1,
            cuda_resources->d_siginv, s1, k1, k0, sk1, sk2, sk3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
    }
    else { // fu update + PML in f update
      KSTRIDE_DEF(dsigu, ku, is, gv);
      if (cnd) {
        if (g2) {
          //////////////////// MOST GENERAL CASE //////////////////////
          step_bfast_kernel13<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_cond,
            cuda_resources->d_f_bfast, cuda_resources->d_g1, cuda_resources->d_g2,
            cuda_resources->d_siginv, cuda_resources->d_siginvu, s1, s2, k1, k2,
            cuda_resources->d_condinv, k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
          /////////////////////////////////////////////////////////////
        }
        else {
          step_bfast_kernel14<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_cond,
            cuda_resources->d_f_bfast, cuda_resources->d_g1, cuda_resources->d_siginv,
            cuda_resources->d_siginvu, s1, k1, cuda_resources->d_condinv,
            k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        cuda_resources->d_f_cond_updated = true;
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          step_bfast_kernel15<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_g2, cuda_resources->d_siginv,
            cuda_resources->d_siginvu, s1, s2, k1, k2, k0, sk1, sk2, sk3, ku0, sku1,
            sku2, sku3, loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
        else {
          step_bfast_kernel16<<<dg, db, 0, cuda_resources->compute_stream>>>(
            cuda_resources->d_f, cuda_resources->d_f_u, cuda_resources->d_f_bfast,
            cuda_resources->d_g1, cuda_resources->d_siginv, cuda_resources->d_siginvu,
            s1, k1, k0, sk1, sk2, sk3, ku0, sku1, sku2, sku3,
            loop_n1, loop_n2, loop_n3, loop_s1, loop_s2, loop_s3, idx0
          );
        }
      }
      cuda_resources->d_f_u_updated = true;
    }
  }
  cuda_resources->d_f_updated = true;
  cuda_resources->d_f_bfast_updated = true;
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void cuda_step_bfast_stride1(RPR f, component c, const RPR g1, const RPR g2,
                             ptrdiff_t s1, ptrdiff_t s2,
                             const grid_volume &gv, const ivec is, const ivec ie, realnum dtdx, direction dsig,
                             const RPR sig, const RPR kap, const RPR siginv, RPR fu, direction dsigu,
                             const RPR sigu, const RPR kapu, const RPR siginvu, realnum dt, const RPR cnd,
                             const RPR cndinv, RPR fcnd, RPR F, realnum k1, realnum k2,
                             CudaResourceManager* cuda_resources) {
  cuda_step_bfast(f, c, g1, g2, s1, s2, gv, is, ie, dtdx, dsig, sig, kap, siginv,
                  fu, dsigu, sigu, kapu, siginvu, dt, cnd, cndinv, fcnd, F, k1, k2,
                  cuda_resources);
}


} // namespace cuda
} // namespace meep
