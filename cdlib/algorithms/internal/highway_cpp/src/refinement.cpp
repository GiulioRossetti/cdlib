#include "refinement.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace highway {

static inline int idx_at(const TopRState& s, int v, int j) {
  return s.idx[v * s.r + j];
}

static inline float val_at(const TopRState& s, int v, int j) {
  return s.val[v * s.r + j];
}

static inline float clamp01(float x) {
  return std::max(0.0f, std::min(1.0f, x));
}

static inline float clamp_range(float x, float lo, float hi) {
  if (lo > hi) std::swap(lo, hi);
  return std::max(lo, std::min(hi, x));
}

struct PatternKey {
  std::vector<int> ids;
  bool operator==(const PatternKey& o) const { return ids == o.ids; }
};

struct PatternHash {
  std::size_t operator()(PatternKey const& k) const noexcept {
    std::size_t h = 1469598103934665603ULL;
    for (int x : k.ids) {
      h ^= static_cast<std::size_t>(x) +
           0x9e3779b97f4a7c15ULL +
           (h << 6) +
           (h >> 2);
    }
    return h;
  }
};

struct PatternInfo {
  PatternKey key;
  std::vector<int> nodes;

  int internal_edges = 0;
  int external_edges = 0;
  std::unordered_map<int, int> ext_counts;

  float self_fraction = 0.0f;
  float neighbor_entropy = 0.0f;
  float confidence = 0.0f;

  // Anchor-indexed target distribution, not pattern-id distribution.
  std::unordered_map<int, float> target;
};

static PatternKey support_key_of_node(const TopRState& st, int v) {
  PatternKey key;
  key.ids.reserve(st.r);

  for (int j = 0; j < st.r; ++j) {
    const int cid = idx_at(st, v, j);
    const float pv = val_at(st, v, j);
    if (cid >= 0 && pv > 0.0f) key.ids.push_back(cid);
  }

  std::sort(key.ids.begin(), key.ids.end());
  key.ids.erase(std::unique(key.ids.begin(), key.ids.end()), key.ids.end());
  return key;
}

static float normalized_entropy_counts(const std::unordered_map<int, int>& counts) {
  double total = 0.0;
  int k = 0;

  for (const auto& kv : counts) {
    if (kv.second > 0) {
      total += kv.second;
      ++k;
    }
  }

  if (total <= 0.0 || k <= 1) return 0.0f;

  double h = 0.0;
  for (const auto& kv : counts) {
    if (kv.second <= 0) continue;
    const double p = static_cast<double>(kv.second) / total;
    h -= p * std::log(p + 1e-12);
  }

  return static_cast<float>(h / std::log(static_cast<double>(k)));
}

static void normalize_distribution(std::unordered_map<int, float>& dist) {
  double sum = 0.0;
  for (const auto& kv : dist) {
    if (kv.second > 0.0f) sum += kv.second;
  }

  if (sum <= 1e-20) return;

  for (auto& kv : dist) {
    if (kv.second > 0.0f) kv.second = static_cast<float>(kv.second / sum);
    else kv.second = 0.0f;
  }
}

static void sharpen_distribution(std::unordered_map<int, float>& dist, float gamma) {
  gamma = std::max(1.0f, gamma);

  if (std::abs(gamma - 1.0f) <= 1e-6f) {
    normalize_distribution(dist);
    return;
  }

  for (auto& kv : dist) {
    kv.second = std::pow(std::max(0.0f, kv.second), gamma);
  }
  normalize_distribution(dist);
}

static float compute_pattern_confidence(
    const PatternInfo& p,
    const LocalRefineConfig& cfg) {

  const float w_self = std::max(0.0f, cfg.confidence_self_fraction_weight);
  const float w_entropy = std::max(0.0f, cfg.confidence_low_entropy_weight);
  const float w_sum = w_self + w_entropy;

  float score = 0.0f;
  if (w_sum > 1e-20f) {
    score =
        w_self * clamp01(p.self_fraction) +
        w_entropy * clamp01(1.0f - p.neighbor_entropy);
    score /= w_sum;
  }

  const float lo = clamp01(cfg.pattern_confidence_floor);
  const float hi = clamp01(cfg.pattern_confidence_ceiling);
  return clamp_range(score, lo, hi);
}

static float same_pattern_neighbor_ratio(
    const CSR& csr,
    const std::vector<int>& node_pid,
    int v) {

  if (v < 0 || v >= csr.n || v >= static_cast<int>(node_pid.size())) return 0.0f;

  const int pid = node_pid[v];
  if (pid < 0) return 0.0f;

  int deg = 0;
  int same = 0;

  const int s0 = csr.indptr[v];
  const int s1 = csr.indptr[v + 1];

  for (int p = s0; p < s1; ++p) {
    const int u = csr.indices[p];
    if (u < 0 || u >= static_cast<int>(node_pid.size())) continue;

    const int qid = node_pid[u];
    if (qid < 0) continue;

    ++deg;
    if (qid == pid) ++same;
  }

  if (deg <= 0) return 0.0f;
  return static_cast<float>(same) / static_cast<float>(deg);
}

static std::unordered_map<int, float> node_neighbor_anchor_consensus(
    const TopRState& st,
    const CSR& csr,
    int v) {

  std::unordered_map<int, float> dist;
  if (v < 0 || v >= csr.n || v >= st.n) return dist;

  const int s0 = csr.indptr[v];
  const int s1 = csr.indptr[v + 1];

  for (int p = s0; p < s1; ++p) {
    const int u = csr.indices[p];
    if (u < 0 || u >= st.n) continue;

    for (int j = 0; j < st.r; ++j) {
      const int cid = idx_at(st, u, j);
      const float pv = val_at(st, u, j);
      if (cid >= 0 && pv > 0.0f) dist[cid] += pv;
    }
  }

  normalize_distribution(dist);
  return dist;
}

static void write_topr_from_distribution(
    TopRState& out,
    int v,
    const std::unordered_map<int, float>& dist,
    float min_abs_mass_to_keep,
    bool renormalize) {

  std::vector<std::pair<int, float>> items;
  items.reserve(dist.size());

  for (const auto& kv : dist) {
    if (kv.first >= 0 && kv.second > min_abs_mass_to_keep) {
      items.push_back(kv);
    }
  }

  if (items.empty()) return;

  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  const int keep = std::min(out.r, static_cast<int>(items.size()));

  for (int j = 0; j < out.r; ++j) {
    out.idx[v * out.r + j] = -1;
    out.val[v * out.r + j] = 0.0f;
  }

  float kept_sum = 0.0f;
  for (int j = 0; j < keep; ++j) {
    out.idx[v * out.r + j] = items[j].first;
    out.val[v * out.r + j] = items[j].second;
    kept_sum += items[j].second;
  }

  if (renormalize && kept_sum > 1e-20f) {
    for (int j = 0; j < keep; ++j) {
      out.val[v * out.r + j] /= kept_sum;
    }
  }
}

TopRState refine_uncertain_nodes_set_search_cpu(
    const TopRState& in,
    const CSR& full_csr,
    const CSR& backbone_csr,
    const LocalRefineConfig& cfg) {

  if (!cfg.enable_pattern_refinement) return in;

  const int n = in.n;
  const int r = in.r;
  if (n <= 0 || r <= 0) return in;

  // ------------------------------------------------------------
  // 1. Extract support-set patterns from propagated memberships.
  // ------------------------------------------------------------
  std::unordered_map<PatternKey, int, PatternHash> key_to_pid;
  key_to_pid.reserve(static_cast<std::size_t>(n) * 2U);

  std::vector<PatternInfo> patterns;
  std::vector<int> node_pid(n, -1);

  for (int v = 0; v < n; ++v) {
    PatternKey key = support_key_of_node(in, v);

    auto it = key_to_pid.find(key);
    int pid = -1;

    if (it == key_to_pid.end()) {
      pid = static_cast<int>(patterns.size());
      key_to_pid.emplace(key, pid);

      PatternInfo info;
      info.key = std::move(key);
      patterns.push_back(std::move(info));
    } else {
      pid = it->second;
    }

    node_pid[v] = pid;
    patterns[pid].nodes.push_back(v);
  }

  if (patterns.empty()) return in;

  // ------------------------------------------------------------
  // 2. Estimate pattern structural confidence on the full graph.
  // ------------------------------------------------------------
  for (int u = 0; u < full_csr.n && u < n; ++u) {
    const int pu = node_pid[u];
    if (pu < 0) continue;

    const int s0 = full_csr.indptr[u];
    const int s1 = full_csr.indptr[u + 1];

    for (int p = s0; p < s1; ++p) {
      const int v = full_csr.indices[p];
      if (v <= u || v < 0 || v >= n) continue; // undirected once

      const int pv = node_pid[v];
      if (pv < 0) continue;

      if (pu == pv) {
        patterns[pu].internal_edges += 1;
      } else {
        patterns[pu].external_edges += 1;
        patterns[pv].external_edges += 1;
        patterns[pu].ext_counts[pv] += 1;
        patterns[pv].ext_counts[pu] += 1;
      }
    }
  }

  for (auto& p : patterns) {
    const int incident_units = 2 * p.internal_edges + p.external_edges;
    if (incident_units > 0) {
      p.self_fraction = static_cast<float>(2 * p.internal_edges) /
                        static_cast<float>(incident_units);
    }

    p.neighbor_entropy = normalized_entropy_counts(p.ext_counts);
    p.confidence = compute_pattern_confidence(p, cfg);
  }

  // ------------------------------------------------------------
  // 3. Build anchor-space pattern targets by averaging propagated memberships.
  // ------------------------------------------------------------
  for (auto& p : patterns) {
    for (int v : p.nodes) {
      for (int j = 0; j < r; ++j) {
        const int cid = idx_at(in, v, j);
        const float pv = val_at(in, v, j);
        if (cid >= 0 && pv > 0.0f) p.target[cid] += pv;
      }
    }

    normalize_distribution(p.target);
    sharpen_distribution(p.target, cfg.target_sharpen_gamma);
  }

  // ------------------------------------------------------------
  // 4. Anchor-preserving soft decoding.
  //
  // Pattern is only a calibration unit. The output remains in the original
  // anchor community space:
  //   q_v = mix * q_pattern + (1 - mix) * q_neighbor
  //   lambda_v = update_strength * confidence(P_v) * same_pattern_ratio(v)^gamma
  //   alpha_new = (1 - lambda_v) * alpha_old + lambda_v * q_v
  // ------------------------------------------------------------
  TopRState out = in;
  out.k = in.k;

  const CSR& local_csr = (backbone_csr.n == n ? backbone_csr : full_csr);
  const float mix = clamp01(cfg.pattern_target_mix);
  const float update_strength = clamp01(cfg.update_strength);
  const float mode_power = std::max(0.0f, cfg.node_mode_power);

  for (int v = 0; v < n; ++v) {
    const int pid = node_pid[v];
    if (pid < 0 || pid >= static_cast<int>(patterns.size())) continue;

    const PatternInfo& p = patterns[pid];
    if (p.target.empty()) continue;

    std::unordered_map<int, float> q;

    for (const auto& kv : p.target) {
      if (kv.first >= 0 && kv.second > 0.0f) q[kv.first] += mix * kv.second;
    }

    std::unordered_map<int, float> q_neighbor =
        node_neighbor_anchor_consensus(in, local_csr, v);

    for (const auto& kv : q_neighbor) {
      if (kv.first >= 0 && kv.second > 0.0f) {
        q[kv.first] += (1.0f - mix) * kv.second;
      }
    }

    normalize_distribution(q);
    if (q.empty()) continue;

    const float same_ratio = same_pattern_neighbor_ratio(local_csr, node_pid, v);
    const float node_factor = std::pow(clamp01(same_ratio), mode_power);
    const float lambda_v = clamp01(update_strength * clamp01(p.confidence) * node_factor);

    if (lambda_v <= 1e-7f) continue;

    std::unordered_map<int, float> blended;

    for (int j = 0; j < r; ++j) {
      const int cid = idx_at(in, v, j);
      const float pv = val_at(in, v, j);
      if (cid >= 0 && pv > 0.0f) {
        blended[cid] += (1.0f - lambda_v) * pv;
      }
    }

    for (const auto& kv : q) {
      if (kv.first >= 0 && kv.second > 0.0f) {
        blended[kv.first] += lambda_v * kv.second;
      }
    }

    normalize_distribution(blended);
    sharpen_distribution(blended, cfg.target_sharpen_gamma);

    write_topr_from_distribution(
        out,
        v,
        blended,
        cfg.min_abs_mass_to_keep,
        cfg.renormalize);
  }

  return out;
}

} // namespace highway
