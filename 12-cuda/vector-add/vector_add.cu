#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.cuh"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char** argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
  std::vector<float> ha(n, 1.f), hb(n, 2.f), hc(n);
  float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  int threads = 256;
  int blocks = ceildiv(n, threads);
  vector_add_kernel<<<blocks, threads>>>(da, db, dc, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("vector_add n=%d c[0]=%f c[n-1]=%f\n", n, hc[0], hc[n - 1]);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
