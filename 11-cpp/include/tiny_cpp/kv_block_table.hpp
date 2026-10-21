#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tiny_cpp {

/// Fixed-size token blocks (block_size tokens share one physical block id).
class BlockAllocator {
 public:
  explicit BlockAllocator(std::size_t num_blocks) : ref_(num_blocks, 0) {
    free_.reserve(num_blocks);
    for (std::size_t i = 0; i < num_blocks; ++i) {
      free_.push_back(static_cast<std::int32_t>(num_blocks - 1 - i));
    }
  }

  std::int32_t allocate() {
    if (free_.empty()) {
      throw std::runtime_error("BlockAllocator: OOM");
    }
    std::int32_t id = free_.back();
    free_.pop_back();
    ref_[static_cast<std::size_t>(id)] = 1;
    return id;
  }

  void retain(std::int32_t block_id) {
    ref_[static_cast<std::size_t>(block_id)] += 1;
  }

  void release(std::int32_t block_id) {
    auto& r = ref_[static_cast<std::size_t>(block_id)];
    if (r == 0) {
      throw std::runtime_error("BlockAllocator: double free");
    }
    r -= 1;
    if (r == 0) {
      free_.push_back(block_id);
    }
  }

  std::size_t free_count() const { return free_.size(); }

 private:
  std::vector<std::int32_t> free_;
  std::vector<std::uint32_t> ref_;
};

class BlockTable {
 public:
  explicit BlockTable(std::size_t block_size) : block_size_(block_size) {}

  std::size_t block_size() const { return block_size_; }

  void append_block(std::int32_t physical_block_id) {
    blocks_.push_back(physical_block_id);
  }

  /// Map logical token index → physical block id (for slot = token index in paged layout).
  std::int32_t block_for_token(std::size_t token_idx) const {
    std::size_t logical_block = token_idx / block_size_;
    if (logical_block >= blocks_.size()) {
      throw std::out_of_range("block_for_token");
    }
    return blocks_[logical_block];
  }

  const std::vector<std::int32_t>& blocks() const { return blocks_; }

 private:
  std::size_t block_size_;
  std::vector<std::int32_t> blocks_;
};

}  // namespace tiny_cpp
