#include "anchors.hpp"
#include "csr.hpp"
#include <algorithm>
#include <numeric>

namespace highway {

std::vector<int> select_anchors_greedy_dedup_with_trace(
    const CSR& full,
    int max_anchors,
    std::vector<AnchorSelectionStep>& trace) {
  trace.clear();

  const int n = full.n;
  max_anchors = std::max(1, max_anchors);
  if (n <= 0) return {};

  std::vector<int> deg = degrees_from_csr(full);
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });

  std::vector<char> covered(n, 0);
  std::vector<int> anchors;
  anchors.reserve(std::min(max_anchors, n));

  int total_covered = 0;
  int step_id = 0;

  for (int u : order) {
    if ((int)anchors.size() >= max_anchors) break;
    if (covered[u]) continue;

    anchors.push_back(u);

    int newly = 0;
    if (!covered[u]) {
      covered[u] = 1;
      ++newly;
      ++total_covered;
    }

    int s0 = full.indptr[u];
    int s1 = full.indptr[u + 1];
    for (int p = s0; p < s1; ++p) {
      int v = full.indices[p];
      if (v >= 0 && v < n && !covered[v]) {
        covered[v] = 1;
        ++newly;
        ++total_covered;
      }
    }

    AnchorSelectionStep rec;
    rec.step = step_id++;
    rec.anchor = u;
    rec.degree = deg[u];
    rec.newly_covered = newly;
    rec.total_covered = total_covered;
    trace.push_back(rec);
  }

  if (anchors.empty()) {
    anchors.push_back(order[0]);
    AnchorSelectionStep rec;
    rec.step = 0;
    rec.anchor = order[0];
    rec.degree = deg[order[0]];
    rec.newly_covered = 1;
    rec.total_covered = 1;
    trace.push_back(rec);
  }

  return anchors;
}

std::vector<int> select_anchors_greedy_dedup(const CSR& full, int max_anchors) {
  std::vector<AnchorSelectionStep> ignored;
  return select_anchors_greedy_dedup_with_trace(full, max_anchors, ignored);
}

} // namespace highway