#pragma once
#include <vector>

namespace highway {

struct CSR {
  int n = 0;
  std::vector<int> indptr;   // size n+1
  std::vector<int> indices;  // size m (two-directed)
};

// Build CSR from two-directed edges (src/dst) in contiguous space [0..n-1]
CSR build_csr_from_edges(int n, const std::vector<int>& src, const std::vector<int>& dst);

// Degrees from CSR
std::vector<int> degrees_from_csr(const CSR& csr);

} // namespace highway