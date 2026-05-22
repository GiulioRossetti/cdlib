#include "highway.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highway {

static inline int intersection_size_sorted(
    const CSR& csr,
    int u,
    int v) {
  int pu = csr.indptr[u];
  int pv = csr.indptr[v];
  const int eu = csr.indptr[u + 1];
  const int ev = csr.indptr[v + 1];

  int inter = 0;
  while (pu < eu && pv < ev) {
    const int a = csr.indices[pu];
    const int b = csr.indices[pv];
    if (a == b) {
      ++inter;
      ++pu;
      ++pv;
    } else if (a < b) {
      ++pu;
    } else {
      ++pv;
    }
  }
  return inter;
}

static inline double jaccard_score_from_csr(
    const CSR& csr,
    const std::vector<int>& deg,
    int u,
    int v) {
  const int du = std::max(0, deg[u]);
  const int dv = std::max(0, deg[v]);
  if (du == 0 && dv == 0) return 0.0;

  const int inter = intersection_size_sorted(csr, u, v);
  const int uni = du + dv - inter;
  if (uni <= 0) return 0.0;
  return static_cast<double>(inter) / static_cast<double>(uni);
}

void build_highway_edges_with_trace(
    const CSR& full,
    const HighwayBuildConfig& cfg,
    std::vector<int>& out_src,
    std::vector<int>& out_dst,
    std::vector<HighwayEdgeDecision>& trace) {

  const int n = full.n;
  out_src.clear();
  out_dst.clear();
  trace.clear();

  if (n <= 0) return;

  const std::vector<int> deg_i = degrees_from_csr(full);

  // full stores an undirected graph as two-directed edges
  const long long m_two_directed = static_cast<long long>(full.indices.size());
  const double m_undirected = std::max(1.0, 0.5 * static_cast<double>(m_two_directed));

  const int r = std::max(1, cfg.top_r);
  const double alpha = std::max(0.0, std::min(1.0, cfg.mod_jaccard_alpha));

  std::vector<std::vector<int>> picked(n);
  std::vector<std::vector<HighwayEdgeDecision>> local_trace(n);

  #if defined(HIGHWAY_USE_OPENMP)
  #pragma omp parallel for schedule(dynamic, 256)
  #endif
  for (int v = 0; v < n; ++v) {
    const int s0 = full.indptr[v];
    const int s1 = full.indptr[v + 1];
    const int dv_i = s1 - s0;
    if (dv_i <= 0) continue;

    std::vector<int> neigh;
    neigh.reserve(dv_i);
    for (int p = s0; p < s1; ++p) {
      neigh.push_back(full.indices[p]);
    }

    std::vector<double> scores(neigh.size(), -std::numeric_limits<double>::infinity());
    std::vector<double> modularity_scores(neigh.size(), 0.0);
    std::vector<double> jaccard_scores(neigh.size(), 0.0);

    for (size_t i = 0; i < neigh.size(); ++i) {
      const int u = neigh[i];
      if (u < 0 || u >= n) continue;

      const double du = static_cast<double>(std::max(0, deg_i[u]));
      const double dv = static_cast<double>(std::max(0, deg_i[v]));

      const double modularity_score = 1.0 - (du * dv) / (2.0 * m_undirected);
      const double jaccard_score = jaccard_score_from_csr(full, deg_i, u, v);
      const double score =
          alpha * modularity_score +
          (1.0 - alpha) * jaccard_score;

      modularity_scores[i] = modularity_score;
      jaccard_scores[i] = jaccard_score;
      scores[i] = score;
    }

    std::vector<int> topi;
    std::vector<double> topv;
    topk_indices_values(
        scores,
        std::min<int>(r, static_cast<int>(scores.size())),
        topi,
        topv);

    std::vector<char> chosen_mask(neigh.size(), 0);
    for (int ii : topi) {
      if (ii >= 0 && ii < static_cast<int>(neigh.size())) chosen_mask[ii] = 1;
    }

    std::vector<int> chosen;
    chosen.reserve(topi.size());
    for (int ii : topi) {
      if (ii >= 0 && ii < static_cast<int>(neigh.size())) {
        chosen.push_back(neigh[ii]);
      }
    }

    if (cfg.ensure_min1_per_node && chosen.empty() && !neigh.empty()) {
      int best_idx = 0;
      double best_score = scores[0];
      for (int i = 1; i < static_cast<int>(scores.size()); ++i) {
        if (scores[i] > best_score) {
          best_score = scores[i];
          best_idx = i;
        }
      }
      chosen.push_back(neigh[best_idx]);
      chosen_mask[best_idx] = 1;
    }

    std::vector<HighwayEdgeDecision> my_trace;
    my_trace.reserve(neigh.size());
    for (size_t i = 0; i < neigh.size(); ++i) {
      HighwayEdgeDecision rec;
      rec.src = v;
      rec.dst = neigh[i];
      rec.modularity_score = modularity_scores[i];
      rec.jaccard_score = jaccard_scores[i];
      rec.hybrid_score = scores[i];
      rec.kept_by_src_topr = (chosen_mask[i] != 0);
      rec.kept_final = false; // finalized later
      my_trace.push_back(rec);
    }

    picked[v] = std::move(chosen);
    local_trace[v] = std::move(my_trace);
  }

  std::unordered_set<long long> final_kept_dir;
  auto make_key = [&](int a, int b) -> long long {
    return static_cast<long long>(a) * static_cast<long long>(n) + static_cast<long long>(b);
  };

  if (cfg.symmetrize) {
    std::unordered_set<long long> dir;
    dir.reserve(static_cast<size_t>(n) * static_cast<size_t>(r) * 2);

    for (int v = 0; v < n; ++v) {
      for (int u : picked[v]) {
        dir.insert(make_key(v, u));
      }
    }

    std::unordered_set<long long> und_seen;
    und_seen.reserve(dir.size());

    out_src.reserve(dir.size() * 2);
    out_dst.reserve(dir.size() * 2);

    for (int v = 0; v < n; ++v) {
      for (int u : picked[v]) {
        if (u < 0 || u >= n || u == v) continue;

        const bool keep = dir.count(make_key(v, u)) || dir.count(make_key(u, v));
        if (!keep) continue;

        const int a = std::min(u, v);
        const int b = std::max(u, v);
        const long long und_key = make_key(a, b);

        if (und_seen.insert(und_key).second) {
          out_src.push_back(a);
          out_dst.push_back(b);
          out_src.push_back(b);
          out_dst.push_back(a);

          final_kept_dir.insert(make_key(a, b));
          final_kept_dir.insert(make_key(b, a));
        }
      }
    }
  } else {
    size_t est = 0;
    for (int v = 0; v < n; ++v) est += picked[v].size();

    out_src.reserve(est);
    out_dst.reserve(est);

    for (int v = 0; v < n; ++v) {
      for (int u : picked[v]) {
        if (u < 0 || u >= n || u == v) continue;
        out_src.push_back(v);
        out_dst.push_back(u);
        final_kept_dir.insert(make_key(v, u));
      }
    }
  }

  for (int v = 0; v < n; ++v) {
    for (auto& rec : local_trace[v]) {
      rec.kept_final = final_kept_dir.count(make_key(rec.src, rec.dst)) > 0;
      trace.push_back(rec);
    }
  }
}

void build_highway_edges(
    const CSR& full,
    const HighwayBuildConfig& cfg,
    std::vector<int>& out_src,
    std::vector<int>& out_dst) {
  std::vector<HighwayEdgeDecision> ignored;
  build_highway_edges_with_trace(full, cfg, out_src, out_dst, ignored);
}

} // namespace highway