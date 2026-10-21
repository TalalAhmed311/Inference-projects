#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.cuh"

__global__ void softmax_row_kernel(const float* x, float* y, int cols) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float maxv = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x) {
    maxv = fmaxf(maxv, x[row * cols + c]);
  }
  smem[tid] = maxv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }
  maxv = smem[0];
  float sum = 0.f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float e = expf(x[row * cols + c] - maxv);
    y[row * cols + c] = e;
    sum += e;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  sum = smem[0];
  for (int c = tid; c < cols; c += blockDim.x) {
    y[row * cols + c] /= sum;
  }
}

int main(int argc, char** argv) {
  int rows = argc > 1 ? std::atoi(argv[1]) : 128;
  int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
  std::vector<float> h(rows * cols), out(rows * cols);
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = static_cast<float>((i % 17) - 8);
  }
  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, out.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  int threads = 256;
  softmax_row_kernel<<<rows, threads, threads * sizeof(float)>>>(dx, dy, cols);
  CUDA_CHECK(cudaMemcpy(out.data(), dy, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float s = 0.f;
  for (int c = 0; c < cols; ++c) {
    s += out[c];
  }
  printf("softmax rows=%d cols=%d row0_sum=%f\n", rows, cols, s);
  cudaFree(dx);
  cudaFree(dy);
  return 0;
}
