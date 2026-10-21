#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.cuh"

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;
  for (int t = 0; t < ceildiv(K, TILE); ++t) {
    int kA = t * TILE + threadIdx.x;
    int kB = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (kB < K && col < N) ? B[kB * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

int main() {
  const int M = 512, N = 512, K = 512;
  std::vector<float> A(M * K, 1.f), B(K * N, 2.f), C(M * N);
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(TILE, TILE);
  dim3 grid(ceildiv(N, TILE), ceildiv(M, TILE));
  matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
  printf("matmul_tiled C[0]=%f (expect %d)\n", C[0], 2 * K);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
