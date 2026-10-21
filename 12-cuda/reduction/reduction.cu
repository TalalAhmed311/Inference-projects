#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.cuh"

__global__ void reduce_sum_kernel(const float* x, float* out, int n) {
  __shared__ float smem[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? x[i] : 0.f;
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, smem[0]);
  }
}

int main(int argc, char** argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 1 << 22;
  std::vector<float> h(n, 1.f);
  float *dx, *dout;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  float zero = 0.f;
  CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(float), cudaMemcpyHostToDevice));
  int threads = 256;
  int blocks = ceildiv(n, threads);
  reduce_sum_kernel<<<blocks, threads>>>(dx, dout, n);
  float sum = 0.f;
  CUDA_CHECK(cudaMemcpy(&sum, dout, sizeof(float), cudaMemcpyDeviceToHost));
  printf("reduction n=%d sum=%f (expect %d)\n", n, sum, n);
  cudaFree(dx);
  cudaFree(dout);
  return 0;
}
