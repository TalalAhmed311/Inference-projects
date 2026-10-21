#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tiny_cpp {

/// Non-owning view into a flattened K or V cache: [num_layers, num_slots, num_heads, head_dim]
struct TensorView {
  float* data = nullptr;
  std::size_t num_layers = 0;
  std::size_t num_slots = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;

  std::size_t slot_stride() const {
    return num_heads * head_dim;
  }

  std::size_t layer_stride() const {
    return num_slots * slot_stride();
  }

  float* layer_slot(std::size_t layer, std::size_t slot) {
    if (layer >= num_layers || slot >= num_slots) {
      throw std::out_of_range("layer_slot index");
    }
    return data + layer * layer_stride() + slot * slot_stride();
  }

  const float* layer_slot(std::size_t layer, std::size_t slot) const {
    return const_cast<TensorView*>(this)->layer_slot(layer, slot);
  }
};

/// Owns a simple contiguous buffer for demos / tests.
class OwnedTensor {
 public:
  OwnedTensor(std::size_t layers, std::size_t slots, std::size_t heads, std::size_t head_dim)
      : dims_{layers, slots, heads, head_dim}, storage_(layers * slots * heads * head_dim, 0.f) {}

  TensorView view() {
    TensorView v;
    v.data = storage_.data();
    v.num_layers = dims_[0];
    v.num_slots = dims_[1];
    v.num_heads = dims_[2];
    v.head_dim = dims_[3];
    return v;
  }

 private:
  std::array<std::size_t, 4> dims_;
  std::vector<float> storage_;
};

}  // namespace tiny_cpp
