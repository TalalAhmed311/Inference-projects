#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.cuh"

// One head, batch of independent sequences: Q,K,V shape [B, S, D] row-major.
__global__ void naive_attention_kernel(const float* Q, const float* K, const float* V, float* O,
                                       int B, int S, int D, float scale) {
  int b = blockIdx.x;
  int i = threadIdx.x;  // query position
  if (b >= B || i >= S) {
    return;
  }
  extern __shared__ float smem[];
  float* scores = smem;
  float maxv = -1e30f;
  for (int j = 0; j < S; ++j) {
    float dot = 0.f;
    for (int d = 0; d < D; ++d) {
      dot += Q[(b * S + i) * D + d] * K[(b * S + j) * D + d];
    }
    dot *= scale;
    scores[j] = dot;
    maxv = fmaxf(maxv, dot);
  }
  float sum = 0.f;
  for (int j = 0; j < S; ++j) {
    scores[j] = expf(scores[j] - maxv);
    sum += scores[j];
  }
  for (int d = 0; d < D; ++d) {
    float acc = 0.f;
    for (int j = 0; j < S; ++j) {
      acc += (scores[j] / sum) * V[(b * S + j) * D + d];
    }
    O[(b * S + i) * D + d] = acc;
  }
}

int main(int argc, char** argv) {
  int B = argc > 1 ? std::atoi(argv[1]) : 2;
  int S = argc > 2 ? std::atoi(argv[2]) : 64;
  int D = argc > 3 ? std::atoi(argv[3]) : 32;
  size_t n = static_cast<size_t>(B) * S * D;
  std::vector<float> Q(n, 0.01f), K(n, 0.02f), V(n, 0.03f), O(n);
  float *dQ, *dK, *dV, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  float scale = 1.f / sqrtf(static_cast<float>(D));
  size_t smem = static_cast<size_t>(S) * sizeof(float);
  naive_attention_kernel<<<B, S, smem>>>(dQ, dK, dV, dO, B, S, D, scale);
  CUDA_CHECK(cudaMemcpy(O.data(), dO, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("naive_attention B=%d S=%d D=%d O[0]=%f\n", B, S, D, O[0]);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  return 0;
}
