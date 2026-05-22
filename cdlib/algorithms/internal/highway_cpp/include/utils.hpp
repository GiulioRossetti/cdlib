#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace highway {

inline bool is_finite(double x) {
  return std::isfinite(x);
}

inline void stable_softmax_inplace(std::vector<float>& x, float tau, float eps = 1e-12f) {
  if (x.empty()) return;
  if (tau <= 0.0f) tau = 1e-6f;

  float mx = -std::numeric_limits<float>::infinity();
  for (float v : x) mx = std::max(mx, v / tau);

  double sum = 0.0;
  for (float& v : x) {
    float z = (v / tau) - mx;
    float e = std::exp(z);
    v = e;
    sum += e;
  }
  if (sum <= eps) {
    // fallback uniform
    float u = 1.0f / std::max<size_t>(1, x.size());
    for (float& v : x) v = u;
    return;
  }
  float inv = static_cast<float>(1.0 / sum);
  for (float& v : x) v *= inv;
}

template <class T>
inline void topk_indices_values(
    const std::vector<T>& a,
    int k,
    std::vector<int>& out_idx,
    std::vector<T>& out_val) {
  out_idx.clear();
  out_val.clear();
  if (k <= 0 || a.empty()) return;
  k = std::min<int>(k, static_cast<int>(a.size()));

  std::vector<int> idx(a.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                   [&](int i, int j) { return a[i] > a[j]; });
  idx.resize(k);

  std::sort(idx.begin(), idx.end(),
            [&](int i, int j) { return a[i] > a[j]; });

  out_idx = idx;
  out_val.reserve(k);
  for (int i : out_idx) out_val.push_back(a[i]);
}

} // namespace highway