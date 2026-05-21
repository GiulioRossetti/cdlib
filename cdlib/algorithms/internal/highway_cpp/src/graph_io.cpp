#include "graph_io.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace highway {

static EdgeList build_from_edges(const std::vector<std::pair<long long, long long>>& edges) {
  EdgeList out;
  std::unordered_map<long long, int> map;
  map.reserve(edges.size() * 2);

  auto get_id = [&](long long x) -> int {
    auto it = map.find(x);
    if (it != map.end()) return it->second;
    int nid = static_cast<int>(map.size());
    map.emplace(x, nid);
    return nid;
  };

  // first pass: assign ids
  for (auto [u, v] : edges) {
    (void)get_id(u);
    (void)get_id(v);
  }
  out.n = static_cast<int>(map.size());
  out.inv_map.resize(out.n);

  // invert map
  for (auto& kv : map) {
    out.inv_map[kv.second] = kv.first;
  }

  // build two-directed
  out.src.reserve(edges.size() * 2);
  out.dst.reserve(edges.size() * 2);
  for (auto [u0, v0] : edges) {
    int u = map[u0];
    int v = map[v0];
    if (u == v) continue; // drop self-loop (keep consistent w/ most pipelines)
    out.src.push_back(u); out.dst.push_back(v);
    out.src.push_back(v); out.dst.push_back(u);
  }

  return out;
}

EdgeList read_edgelist_to_undirected_two_directed_contig(const std::string& path) {
  std::ifstream fin(path);
  if (!fin) throw std::runtime_error("Failed to open file: " + path);

  std::vector<std::pair<long long, long long>> edges;
  edges.reserve(1 << 20);

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') continue;
    std::istringstream iss(line);
    long long u, v;
    if (!(iss >> u >> v)) continue;
    edges.emplace_back(u, v);
  }
  return build_from_edges(edges);
}

EdgeList to_undirected_two_directed_contig(const std::vector<std::pair<long long, long long>>& edges) {
  return build_from_edges(edges);
}

} // namespace highway