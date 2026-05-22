#include "propagation.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace highway {

static inline int at(const std::vector<int>& a, int n, int r, int v, int j) {
  return a[v * r + j];
}
static inline float atf(const std::vector<float>& a, int n, int r, int v, int j) {
  return a[v * r + j];
}
static inline void set(std::vector<int>& a, int n, int r, int v, int j, int x) {
  a[v * r + j] = x;
}
static inline void setf(std::vector<float>& a, int n, int r, int v, int j, float x) {
  a[v * r + j] = x;
}

TopRState propagate_other_assignment_topr_cpu(
    const CSR& highway_csr,
    const CSR& full_csr,
    const std::vector<int>& anchors,
    const PropConfig& cfg) {

  TopRState st;
  const int n = highway_csr.n;
  const int r = std::max(1, cfg.top_r);
  const int k = (int)anchors.size();
  st.n = n; st.r = r; st.k = k;

  st.idx.assign(n * r, -1);
  st.val.assign(n * r, 0.0f);

  if (n <= 0 || k <= 0) return st;

  // init anchors: community ids aligned with anchors list
  for (int cid = 0; cid < k; ++cid) {
    int v = anchors[cid];
    if (v < 0 || v >= n) continue;
    set(st.idx, n, r, v, 0, cid);
    setf(st.val, n, r, v, 0, 1.0f);
  }

  // degrees
  std::vector<int> deg_h_i = degrees_from_csr(highway_csr);
  std::vector<int> deg_f_i = degrees_from_csr(full_csr);
  std::vector<double> deg_h(n, 1.0), deg_f(n, 1.0);
  for (int i = 0; i < n; ++i) {
    deg_h[i] = std::max(1, deg_h_i[i]);
    deg_f[i] = std::max(1, deg_f_i[i]);
  }

  const int T = std::max(0, cfg.T);
  const double damping = cfg.damping;
  const double eta_leak = cfg.eta_leak;
  const float tau = (float)cfg.tau;
  const float eps = (float)cfg.eps;

  std::vector<int> idx_new(n * r, -1);
  std::vector<float> val_new(n * r, 0.0f);

  std::vector<float> acc_k; acc_k.reserve(k);

  for (int it = 0; it < T; ++it) {
    std::fill(idx_new.begin(), idx_new.end(), -1);
    std::fill(val_new.begin(), val_new.end(), 0.0f);

    // ---- per node update ----
    #if defined(HIGHWAY_USE_OPENMP)
    #pragma omp parallel
    #endif
    {
      std::vector<float> acc_local;
      acc_local.resize(k, 0.0f);

      #if defined(HIGHWAY_USE_OPENMP)
      #pragma omp for schedule(dynamic, 128)
      #endif
      for (int v = 0; v < n; ++v) {
        std::fill(acc_local.begin(), acc_local.end(), 0.0f);

        // highway neighbors
        int s0 = highway_csr.indptr[v];
        int s1 = highway_csr.indptr[v + 1];
        if (s1 > s0) {
          for (int p = s0; p < s1; ++p) {
            int u = highway_csr.indices[p];
            if (u < 0 || u >= n) continue;
            double w = 1.0 / std::sqrt(deg_h[u] * deg_h[v]);

            for (int j = 0; j < r; ++j) {
              int cid = at(st.idx, n, r, u, j);
              float pv = atf(st.val, n, r, u, j);
              if (cid >= 0 && pv > 0.0f) {
                acc_local[cid] += (float)(pv * w);
              }
            }
          }
        }

        // optional leak from full graph
        if (eta_leak > 0.0) {
          int f0 = full_csr.indptr[v];
          int f1 = full_csr.indptr[v + 1];
          if (f1 > f0) {
            for (int p = f0; p < f1; ++p) {
              int u = full_csr.indices[p];
              if (u < 0 || u >= n) continue;
              double w = eta_leak / std::sqrt(deg_f[u] * deg_f[v]);
              for (int j = 0; j < r; ++j) {
                int cid = at(st.idx, n, r, u, j);
                float pv = atf(st.val, n, r, u, j);
                if (cid >= 0 && pv > 0.0f) {
                  acc_local[cid] += (float)(pv * w);
                }
              }
            }
          }
        }

        // fallback copy
        bool any_pos = false;
        for (int c = 0; c < k; ++c) {
          if (acc_local[c] > 0.0f) { any_pos = true; break; }
        }
        if (!any_pos) {
          for (int j = 0; j < r; ++j) {
            set(idx_new, n, r, v, j, at(st.idx, n, r, v, j));
            setf(val_new, n, r, v, j, atf(st.val, n, r, v, j));
          }
          continue;
        }

        // top-r + softmax
        std::vector<int> topi;
        std::vector<float> topv;
        topk_indices_values(acc_local, std::min(r, k), topi, topv);

        stable_softmax_inplace(topv, tau, eps);

        for (int j = 0; j < r; ++j) {
          set(idx_new, n, r, v, j, -1);
          setf(val_new, n, r, v, j, 0.0f);
        }
        int rr = (int)topi.size();
        for (int j = 0; j < rr; ++j) {
          set(idx_new, n, r, v, j, topi[j]);
          setf(val_new, n, r, v, j, topv[j]);
        }
      }
    }

    // ---- damping mix (same as python: scatter into dense acc_k then top-r+softmax) ----
    if (damping < 1.0) {
      const float d = (float)damping;
      const float one_minus = 1.0f - d;

      #if defined(HIGHWAY_USE_OPENMP)
      #pragma omp parallel
      #endif
      {
        std::vector<float> acc_local;
        acc_local.resize(k, 0.0f);

        #if defined(HIGHWAY_USE_OPENMP)
        #pragma omp for schedule(dynamic, 128)
        #endif
        for (int v = 0; v < n; ++v) {
          float sum_new = 0.0f;
          for (int j = 0; j < r; ++j) sum_new += atf(val_new, n, r, v, j);
          if (sum_new <= 0.0f) continue;

          std::fill(acc_local.begin(), acc_local.end(), 0.0f);

          // add new
          for (int j = 0; j < r; ++j) {
            int cid = at(idx_new, n, r, v, j);
            float pv = atf(val_new, n, r, v, j);
            if (cid >= 0 && pv > 0.0f) acc_local[cid] += d * pv;
          }
          // add old
          for (int j = 0; j < r; ++j) {
            int cid = at(st.idx, n, r, v, j);
            float pv = atf(st.val, n, r, v, j);
            if (cid >= 0 && pv > 0.0f) acc_local[cid] += one_minus * pv;
          }

          std::vector<int> topi;
          std::vector<float> topv;
          topk_indices_values(acc_local, std::min(r, k), topi, topv);
          stable_softmax_inplace(topv, tau, eps);

          for (int j = 0; j < r; ++j) {
            set(idx_new, n, r, v, j, -1);
            setf(val_new, n, r, v, j, 0.0f);
          }
          int rr = (int)topi.size();
          for (int j = 0; j < rr; ++j) {
            set(idx_new, n, r, v, j, topi[j]);
            setf(val_new, n, r, v, j, topv[j]);
          }
        }
      }
    }

    // swap into state
    st.idx.swap(idx_new);
    st.val.swap(val_new);
  }

  return st;
}

} // namespace highway