#pragma once
#include "csr.hpp"
#include <vector>

namespace highway {

struct AnchorSelectionStep {
  int step = -1;
  int anchor = -1;         // contig id
  int degree = 0;          // in the graph passed into selector
  int newly_covered = 0;   // how many uncovered nodes became covered due to this anchor
  int total_covered = 0;   // cumulative covered nodes after this step
};

// Original API kept for compatibility.
std::vector<int> select_anchors_greedy_dedup(const CSR& full, int max_anchors);

// New explainable API.
std::vector<int> select_anchors_greedy_dedup_with_trace(
    const CSR& full,
    int max_anchors,
    std::vector<AnchorSelectionStep>& trace);

} // namespace highway