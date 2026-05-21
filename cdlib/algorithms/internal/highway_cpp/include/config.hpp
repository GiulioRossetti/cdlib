#pragma once
#include <cstdint>

namespace highway {

struct HighwayBuildConfig {
  int top_r = 3;
  bool ensure_min1_per_node = true;
  bool symmetrize = true;

  // Hybrid backbone score:
  // score(u,v) = alpha * modularity_score(u,v) + (1 - alpha) * jaccard(u,v)
  // alpha in [0, 1]. Larger alpha -> more modularity-driven.
  double mod_jaccard_alpha = 0.70;
};

struct PropConfig {
  int top_r = 3;
  int T = 15;
  double damping = 0.9;
  double eta_leak = 0.0;
  double tau = 0.01;
  double eps = 1e-12;
};

} // namespace highway