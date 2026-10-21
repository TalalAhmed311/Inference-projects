cmake_minimum_required(VERSION 3.16)
project(tiny_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tiny_cpp STATIC
  src/kv_block_table.cpp
)
target_include_directories(tiny_cpp PUBLIC include)

add_executable(kv_block_table_demo src/kv_block_table_demo.cpp)
target_link_libraries(kv_block_table_demo PRIVATE tiny_cpp)

add_executable(layout_bench src/layout_bench.cpp)
target_link_libraries(layout_bench PRIVATE tiny_cpp)

enable_testing()
add_test(NAME kv_block_table_demo COMMAND kv_block_table_demo)
