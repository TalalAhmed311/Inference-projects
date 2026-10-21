#include <iostream>

#include "tiny_cpp/kv_block_table.hpp"

int main() {
  constexpr std::size_t kBlockSize = 16;
  tiny_cpp::BlockAllocator alloc(8);
  tiny_cpp::BlockTable table(kBlockSize);

  auto b0 = alloc.allocate();
  auto b1 = alloc.allocate();
  table.append_block(b0);
  table.append_block(b1);

  std::cout << "token 0 block " << table.block_for_token(0) << "\n";
  std::cout << "token 17 block " << table.block_for_token(17) << "\n";

  alloc.release(b0);
  alloc.release(b1);
  std::cout << "free blocks " << alloc.free_count() << "\n";
  return 0;
}
