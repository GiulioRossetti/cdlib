#include "csr.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace highway {

CSR build_csr_from_edges(int n, const std::vector<int>& src, const std::vector<int>& dst) {
  if (src.size() != dst.size()) throw std::runtime_error("src/dst size mismatch");
  CSR csr;
  csr.n = n;
  const size_t m = src.size();

  csr.indptr.assign(n + 1, 0);

  // count
  for (size_t i = 0; i < m; ++i) {
    int u = src[i];
    if (u < 0 || u >= n) throw std::runtime_error("src out of range");
    csr.indptr[u + 1] += 1;
  }

  // prefix sum
  for (int i = 0; i < n; ++i) csr.indptr[i + 1] += csr.indptr[i];

  csr.indices.assign(m, -1);
  std::vector<int> cur = csr.indptr; // copy

  // fill
  for (size_t i = 0; i < m; ++i) {
    int u = src[i];
    int v = dst[i];
    int p = cur[u]++;
    csr.indices[p] = v;
  }

  // sort neighbor lists (helps deterministic behavior)
  for (int u = 0; u < n; ++u) {
    int s0 = csr.indptr[u];
    int s1 = csr.indptr[u + 1];
    std::sort(csr.indices.begin() + s0, csr.indices.begin() + s1);
  }

  return csr;
}

std::vector<int> degrees_from_csr(const CSR& csr) {
  std::vector<int> deg(csr.n, 0);
  for (int u = 0; u < csr.n; ++u) {
    deg[u] = csr.indptr[u + 1] - csr.indptr[u];
  }
  return deg;
}

} // namespace highway