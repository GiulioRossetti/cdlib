#pragma once
#include "config.hpp"
#include "csr.hpp"
#include <vector>

namespace highway {

struct HighwayEdgeDecision {
  int src = -1;                 // contig id
  int dst = -1;                 // contig id
  double modularity_score = 0;  // 1 - d_u d_v / (2m)
  double jaccard_score = 0;
  double hybrid_score = 0;
  bool kept_by_src_topr = false; // selected when processing src
  bool kept_final = false;       // after symmetrize/final retention
};

// Original API kept for compatibility.
void build_highway_edges(
    const CSR& full,
    const HighwayBuildConfig& cfg,
    std::vector<int>& out_src,
    std::vector<int>& out_dst);

// New explainable API.
void build_highway_edges_with_trace(
    const CSR& full,
    const HighwayBuildConfig& cfg,
    std::vector<int>& out_src,
    std::vector<int>& out_dst,
    std::vector<HighwayEdgeDecision>& trace);

} // namespace highway