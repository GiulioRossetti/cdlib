#pragma once
#include "config.hpp"
#include "csr.hpp"
#include <vector>

namespace highway {

struct TopRState {
  int n = 0;
  int r = 0;
  int k = 0;
  std::vector<int> idx;   // size n*r, -1 if empty
  std::vector<float> val; // size n*r
};

TopRState propagate_other_assignment_topr_cpu(
    const CSR& highway_csr,
    const CSR& full_csr,
    const std::vector<int>& anchors, // contig node ids
    const PropConfig& cfg);

} // namespace highway