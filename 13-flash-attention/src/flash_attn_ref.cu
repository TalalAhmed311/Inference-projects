#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../12-cuda/common/cuda_utils.cuh"

// Reference: one head, [B, S, D], causal mask optional (full bidirectional here for simplicity).
template <int Br, int Bc>
__global__ void flash_attn_ref_kernel(const float* Q, const float* K, const float* V, float* O, int B, int S,
                                      int D, float scale) {
  int batch = blockIdx.x;
  int q0 = blockIdx.y * Br;
  if (batch >= B) {
    return;
  }

  __shared__ float q_tile[Br][Bc];  // reuse width for K tile loads
  __shared__ float k_tile[Bc][Bc];
  __shared__ float v_tile[Bc][Bc];

  float m[Br];
  float l[Br];
  float acc[Br][Bc];  // D limited to Bc in this teaching kernel
  for (int i = 0; i < Br; ++i) {
    m[i] = -1e30f;
    l[i] = 0.f;
    for (int d = 0; d < Bc; ++d) {
      acc[i][d] = 0.f;
    }
  }

  for (int k0 = 0; k0 < S; k0 += Bc) {
    // Load K/V tile into shared (only first Bc dims when D > Bc — teaching simplification)
    for (int ty = threadIdx.y; ty < Bc; ty += blockDim.y) {
      for (int tx = threadIdx.x; tx < Bc; tx += blockDim.x) {
        int k = k0 + ty;
        int d = tx;
        k_tile[ty][tx] = (k < S && d < D) ? K[(batch * S + k) * D + d] * scale : 0.f;
        v_tile[ty][tx] = (k < S && d < D) ? V[(batch * S + k) * D + d] : 0.f;
      }
    }
    __syncthreads();

    for (int i = 0; i < Br; ++i) {
      int q = q0 + i;
      if (q >= S) {
        continue;
      }
      float row_m = m[i];
      float row_l = l[i];
      float scores[Bc];
      float block_m = -1e30f;
      for (int j = 0; j < Bc; ++j) {
        int k = k0 + j;
        if (k >= S) {
          scores[j] = -1e30f;
          continue;
        }
        float dot = 0.f;
        for (int d = 0; d < D && d < Bc; ++d) {
          float qv = Q[(batch * S + q) * D + d];
          dot += qv * k_tile[j][d];
        }
        scores[j] = dot;
        block_m = fmaxf(block_m, dot);
      }
      float new_m = fmaxf(row_m, block_m);
      float alpha = expf(row_m - new_m);
      row_l *= alpha;
      for (int d = 0; d < D && d < Bc; ++d) {
        acc[i][d] *= alpha;
      }
      for (int j = 0; j < Bc; ++j) {
        int k = k0 + j;
        if (k >= S) {
          continue;
        }
        float p = expf(scores[j] - new_m);
        row_l += p;
        for (int d = 0; d < D && d < Bc; ++d) {
          acc[i][d] += p * v_tile[j][d];
        }
      }
      m[i] = new_m;
      l[i] = row_l;
    }
    __syncthreads();
  }

  for (int i = 0; i < Br; ++i) {
    int q = q0 + i;
    if (q >= S) {
      continue;
    }
    for (int d = threadIdx.x; d < D && d < Bc; d += blockDim.x) {
      O[(batch * S + q) * D + d] = acc[i][d] / l[i];
    }
  }
}

int main(int argc, char** argv) {
  int B = argc > 1 ? std::atoi(argv[1]) : 1;
  int S = argc > 2 ? std::atoi(argv[2]) : 256;
  int D = argc > 3 ? std::atoi(argv[3]) : 64;
  if (D > 64) {
    fprintf(stderr, "flash_attn_ref teaching kernel supports D <= 64\n");
    return 1;
  }
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
  constexpr int Br = 32;
  constexpr int Bc = 32;
  dim3 block(32, 4);
  dim3 grid(B, ceildiv(S, Br));
  float scale = 1.f / sqrtf(static_cast<float>(D));
  flash_attn_ref_kernel<Br, Bc><<<grid, block>>>(dQ, dK, dV, dO, B, S, D, scale);
  CUDA_CHECK(cudaMemcpy(O.data(), dO, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("flash_attn_ref B=%d S=%d D=%d O[0]=%f\n", B, S, D, O[0]);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  return 0;
}
