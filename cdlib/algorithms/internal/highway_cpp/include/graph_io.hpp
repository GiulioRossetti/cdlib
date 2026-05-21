#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace highway {

struct EdgeList {
  // undirected edge list stored as two-directed (u->v and v->u)
  int n = 0; // number of nodes in contiguous space [0..n-1]
  std::vector<int> src;
  std::vector<int> dst;

  // mapping: contig_id -> original_id (for writing outputs)
  std::vector<long long> inv_map;
};

// Read edge list file: each line "u v" (original ids can be int64)
EdgeList read_edgelist_to_undirected_two_directed_contig(const std::string& path);

// If you already have edges in memory (original ids), convert to contig + two-directed
EdgeList to_undirected_two_directed_contig(const std::vector<std::pair<long long, long long>>& edges);

} // namespace highway