#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

#define CUDA_CHECK(call) cuda_check((call), #call)

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
