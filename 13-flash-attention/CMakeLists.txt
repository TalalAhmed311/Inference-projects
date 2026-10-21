cmake_minimum_required(VERSION 3.18)
project(flash_attn_ref LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(flash_attn_ref src/flash_attn_ref.cu)
target_compile_options(flash_attn_ref PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
