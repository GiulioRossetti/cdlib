#pragma once

#include "csr.hpp"
#include "propagation.hpp"

namespace highway {

// Anchor-preserving pattern decoding configuration.
//
// Design principle:
//   1) Extract support-set patterns from propagated anchor-indexed memberships.
//   2) Use each pattern only as a calibration unit, not as a new community id.
//   3) Build a pattern-level target distribution in the original anchor space.
//   4) Blend this target with local backbone-neighbor consensus.
//   5) Apply conservative shrinkage:
//        alpha_new = (1 - lambda_v) * alpha_old + lambda_v * q_v
//
// Therefore, the returned TopRState preserves the original anchor-indexed
// community ids. This avoids the previous pattern-id reassignment problem.
struct LocalRefineConfig {
  // Enable/disable anchor-preserving pattern decoding.
  bool enable_pattern_refinement = true;

  // Pattern-confidence weights.
  // confidence(P) is computed from:
  //   self_fraction(P): internal cohesion of the support pattern;
  //   1 - neighbor_entropy(P): concentration of neighboring patterns.
  // The weighted score is normalized by the total positive weight.
  float confidence_self_fraction_weight = 0.75f;
  float confidence_low_entropy_weight = 0.25f;

  // Bounds for pattern confidence.
  float pattern_confidence_floor = 0.00f;
  float pattern_confidence_ceiling = 1.00f;

  // Maximum shrinkage/update strength.
  // Actual lambda_v is:
  //   update_strength * confidence(P_v) * same_pattern_ratio(v)^node_mode_power
  float update_strength = 0.45f;
  float node_mode_power = 2.00f;

  // Blend between pattern target and node-neighbor target:
  //   q_v = pattern_target_mix * q_pattern +
  //         (1 - pattern_target_mix) * q_neighbor
  float pattern_target_mix = 0.65f;

  // Optional sharpening applied to pattern-level and final target distributions.
  // gamma=1 keeps the distribution unchanged; gamma>1 makes it more peaked.
  float target_sharpen_gamma = 1.00f;

  // Output sparsification.
  bool renormalize = true;
  float min_abs_mass_to_keep = 1e-8f;
};

// Anchor-preserving pattern decoding refinement.
// The returned TopRState keeps the original anchor ids as community ids.
TopRState refine_uncertain_nodes_set_search_cpu(
    const TopRState& in,
    const CSR& full_csr,
    const CSR& backbone_csr,
    const LocalRefineConfig& cfg);

} // namespace highway
