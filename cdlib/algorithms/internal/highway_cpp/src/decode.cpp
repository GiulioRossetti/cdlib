#include "decode.hpp"
#include <algorithm>
#include <utility>

namespace highway {

std::vector<std::vector<int>> decode_overlapping_communities(
    const TopRState& st,
    const std::vector<int>& anchors,
    float theta,
    int max_memberships) {

  max_memberships = std::max(1, max_memberships);
  const int n = st.n;
  const int r = st.r;

  std::vector<std::vector<int>> out;
  out.reserve(n);

  for (int v = 0; v < n; ++v) {
    std::vector<std::pair<int, float>> pairs;
    pairs.reserve(r);

    for (int j = 0; j < r; ++j) {
      int cid = st.idx[v * r + j];
      float pv = st.val[v * r + j];
      if (cid >= 0 && pv >= theta) {
        int anchor_node = anchors[cid];
        pairs.emplace_back(anchor_node, pv);
      }
    }

    std::sort(pairs.begin(), pairs.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    std::vector<int> labs;
    for (int i = 0; i < (int)pairs.size() && i < max_memberships; ++i) {
      labs.push_back(pairs[i].first);
    }
    out.push_back(std::move(labs));
  }

  return out;
}

} // namespace highway