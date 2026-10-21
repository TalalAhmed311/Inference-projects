#include <chrono>
#include <iostream>
#include <vector>

#include "tiny_cpp/tensor_view.hpp"

int main() {
  constexpr std::size_t layers = 4;
  constexpr std::size_t slots = 4096;
  constexpr std::size_t heads = 8;
  constexpr std::size_t dim = 128;
  tiny_cpp::OwnedTensor tensor(layers, slots, heads, dim);
  auto view = tensor.view();

  const int iters = 200;
  volatile float sink = 0.f;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (std::size_t layer = 0; layer < layers; ++layer) {
      for (std::size_t slot = 0; slot < slots; slot += 64) {
        float* row = view.layer_slot(layer, slot);
        sink += row[0];
      }
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "layout_bench " << iters << " passes, " << ms << " ms, sink=" << sink << "\n";
  return 0;
}
