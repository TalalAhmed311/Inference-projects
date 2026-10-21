cmake_minimum_required(VERSION 3.18)
project(tiny_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(vector_add vector-add/vector_add.cu)
add_executable(reduction reduction/reduction.cu)
add_executable(matmul_tiled matmul/matmul_tiled.cu)
add_executable(softmax softmax/softmax.cu)
add_executable(naive_attention attention/naive_attention.cu)

foreach(tgt vector_add reduction matmul_tiled softmax naive_attention)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endforeach()
