#pragma once
#include "propagation.hpp"
#include <vector>

namespace highway {

// node -> list of anchor-node-id (contig space)
std::vector<std::vector<int>> decode_overlapping_communities(
    const TopRState& st,
    const std::vector<int>& anchors,
    float theta = 0.30f,
    int max_memberships = 3);

} // namespace highway