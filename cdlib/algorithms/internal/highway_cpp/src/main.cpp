#include "anchors.hpp"
#include "config.hpp"
#include "csr.hpp"
#include "graph_io.hpp"
#include "highway.hpp"
#include "propagation.hpp"
#include "refinement.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static inline double now_sec() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static highway::TopRState make_init_state(
    int n,
    int r,
    const std::vector<int>& anchors) {
  highway::TopRState st;
  st.n = n;
  st.r = r;
  st.k = static_cast<int>(anchors.size());
  st.idx.assign(n * r, -1);
  st.val.assign(n * r, 0.0f);

  for (int cid = 0; cid < static_cast<int>(anchors.size()); ++cid) {
    int v = anchors[cid];
    if (v < 0 || v >= n) continue;
    st.idx[v * r + 0] = cid;
    st.val[v * r + 0] = 1.0f;
  }

  return st;
}

static std::vector<std::vector<int>> decode_state_labels(
    const highway::TopRState& st,
    float theta,
    int max_memberships) {
  const int keep_max = std::max(1, max_memberships);
  std::vector<std::vector<int>> out(st.n);

  for (int v = 0; v < st.n; ++v) {
    std::vector<std::pair<int, float>> items;
    items.reserve(st.r);

    for (int j = 0; j < st.r; ++j) {
      int cid = st.idx[v * st.r + j];
      float pv = st.val[v * st.r + j];

      if (cid >= 0 && pv > 0.0f) {
        items.emplace_back(cid, pv);
      }
    }

    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });

    for (const auto& kv : items) {
      if (static_cast<int>(out[v].size()) >= keep_max) break;
      if (kv.second >= theta) out[v].push_back(kv.first);
    }

    if (out[v].empty() && !items.empty()) {
      out[v].push_back(items.front().first);
    }
  }

  return out;
}

static long long count_undirected_edges_from_csr(const highway::CSR& csr) {
  long long m = 0;

  for (int u = 0; u < csr.n; ++u) {
    int s0 = csr.indptr[u];
    int s1 = csr.indptr[u + 1];

    for (int p = s0; p < s1; ++p) {
      int v = csr.indices[p];
      if (u < v) ++m;
    }
  }

  return m;
}

static long long count_total_memberships(
    const std::vector<std::vector<int>>& node_memberships) {
  long long total = 0;

  for (const auto& labs : node_memberships) {
    total += static_cast<long long>(labs.size());
  }

  return total;
}

static std::vector<std::vector<int>> build_communities_from_node_memberships(
    const std::vector<std::vector<int>>& node_memberships) {
  std::map<int, std::vector<int>> comm_to_nodes;

  for (int v = 0; v < static_cast<int>(node_memberships.size()); ++v) {
    for (int lab : node_memberships[v]) {
      if (lab >= 0) {
        comm_to_nodes[lab].push_back(v);
      }
    }
  }

  std::vector<std::vector<int>> communities;
  communities.reserve(comm_to_nodes.size());

  for (auto& kv : comm_to_nodes) {
    if (!kv.second.empty()) {
      communities.push_back(std::move(kv.second));
    }
  }

  return communities;
}

static void write_communities_json(
    const std::string& path,
    const std::vector<long long>& inv_map,
    const std::vector<std::vector<int>>& communities_contig) {
  std::ofstream fout(path);

  if (!fout) {
    throw std::runtime_error("Cannot write communities file: " + path);
  }

  fout << "[\n";

  for (int ci = 0; ci < static_cast<int>(communities_contig.size()); ++ci) {
    fout << "  [";

    const auto& comm = communities_contig[ci];
    bool first = true;

    for (int i = 0; i < static_cast<int>(comm.size()); ++i) {
      int v = comm[i];

      if (v < 0 || v >= static_cast<int>(inv_map.size())) {
        continue;
      }

      if (!first) {
        fout << ", ";
      }

      fout << inv_map[v];
      first = false;
    }

    fout << "]";

    if (ci + 1 < static_cast<int>(communities_contig.size())) {
      fout << ",";
    }

    fout << "\n";
  }

  fout << "]\n";
}

int main(int argc, char** argv) {
  std::string input;

  highway::HighwayBuildConfig hcfg;
  int max_anchors = -1;

  highway::PropConfig pcfg;
  pcfg.top_r = 3;
  pcfg.T = 10;
  pcfg.damping = 0.9;
  pcfg.eta_leak = 0.0;
  pcfg.tau = 0.85;

  highway::LocalRefineConfig lrcfg;

  float decode_theta = 0.30f;
  int max_memberships = 3;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];

    auto need = [&](const std::string& key) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + key);
      }
      return std::string(argv[++i]);
    };

    if (a == "--input") {
      input = need(a);
    }

    else if (a == "--highway_top_r") {
      hcfg.top_r = std::stoi(need(a));
    }
    else if (a == "--ensure_min1") {
      hcfg.ensure_min1_per_node = (need(a) == "1");
    }
    else if (a == "--symmetrize") {
      hcfg.symmetrize = (need(a) == "1");
    }
    else if (a == "--mod_jaccard_alpha") {
      hcfg.mod_jaccard_alpha = std::stod(need(a));
    }

    else if (a == "--max_anchors") {
      max_anchors = std::stoi(need(a));
    }

    else if (a == "--prop_top_r") {
      pcfg.top_r = std::stoi(need(a));
    }
    else if (a == "--prop_T") {
      pcfg.T = std::stoi(need(a));
    }
    else if (a == "--prop_damping") {
      pcfg.damping = std::stod(need(a));
    }
    else if (a == "--prop_eta_leak") {
      pcfg.eta_leak = std::stod(need(a));
    }
    else if (a == "--prop_tau") {
      pcfg.tau = std::stod(need(a));
    }

    else if (a == "--local_enable_pattern_refinement") {
      lrcfg.enable_pattern_refinement = (need(a) == "1");
    }
    else if (a == "--local_confidence_self_fraction_weight") {
      lrcfg.confidence_self_fraction_weight = std::stof(need(a));
    }
    else if (a == "--local_confidence_low_entropy_weight") {
      lrcfg.confidence_low_entropy_weight = std::stof(need(a));
    }
    else if (a == "--local_pattern_confidence_floor") {
      lrcfg.pattern_confidence_floor = std::stof(need(a));
    }
    else if (a == "--local_pattern_confidence_ceiling") {
      lrcfg.pattern_confidence_ceiling = std::stof(need(a));
    }
    else if (a == "--local_update_strength") {
      lrcfg.update_strength = std::stof(need(a));
    }
    else if (a == "--local_node_mode_power") {
      lrcfg.node_mode_power = std::stof(need(a));
    }
    else if (a == "--local_pattern_target_mix") {
      lrcfg.pattern_target_mix = std::stof(need(a));
    }
    else if (a == "--local_target_sharpen_gamma") {
      lrcfg.target_sharpen_gamma = std::stof(need(a));
    }
    else if (a == "--local_min_abs_mass_to_keep") {
      lrcfg.min_abs_mass_to_keep = std::stof(need(a));
    }
    else if (a == "--local_renormalize") {
      lrcfg.renormalize = (need(a) == "1");
    }

    else if (a == "--decode_theta") {
      decode_theta = std::stof(need(a));
    }
    else if (a == "--max_memberships") {
      max_memberships = std::stoi(need(a));
    }

    else if (a == "--help") {
      std::cout
          << "Usage: highway_algorithm_only --input graph.edgelist\n"
          << "\n"
          << "Core Highway parameters:\n"
          << "  --highway_top_r 3\n"
          << "  --ensure_min1 1\n"
          << "  --symmetrize 1\n"
          << "  --mod_jaccard_alpha 0.70\n"
          << "  --max_anchors <int>\n"
          << "\n"
          << "Propagation parameters:\n"
          << "  --prop_top_r 3\n"
          << "  --prop_T 10\n"
          << "  --prop_damping 0.9\n"
          << "  --prop_eta_leak 0\n"
          << "  --prop_tau 0.85\n"
          << "\n"
          << "Anchor-preserving pattern decoding parameters:\n"
          << "  --local_enable_pattern_refinement 1\n"
          << "  --local_confidence_self_fraction_weight 0.85\n"
          << "  --local_confidence_low_entropy_weight 0.15\n"
          << "  --local_pattern_confidence_floor 0.05\n"
          << "  --local_pattern_confidence_ceiling 1.00\n"
          << "  --local_update_strength 0.50\n"
          << "  --local_node_mode_power 1.50\n"
          << "  --local_pattern_target_mix 0.75\n"
          << "  --local_target_sharpen_gamma 1.20\n"
          << "  --local_min_abs_mass_to_keep 1e-8\n"
          << "  --local_renormalize 1\n"
          << "\n"
          << "Final decode parameters:\n"
          << "  --decode_theta 0.30\n"
          << "  --max_memberships 3\n";
      return 0;
    }

    else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (input.empty()) {
    std::cerr << "Error: --input is required. Use --help.\n";
    return 1;
  }

  std::unordered_map<std::string, double> timers;
  const double t_global0 = now_sec();

  try {
    double t0 = now_sec();
    highway::EdgeList el =
        highway::read_edgelist_to_undirected_two_directed_contig(input);
    timers["Read+Relabel"] = now_sec() - t0;

    const int n = el.n;
    if (n <= 0) {
      std::cerr << "Empty graph.\n";
      return 0;
    }

    t0 = now_sec();
    highway::CSR full = highway::build_csr_from_edges(n, el.src, el.dst);
    timers["Full_CSR"] = now_sec() - t0;

    t0 = now_sec();
    std::vector<int> hsrc;
    std::vector<int> hdst;
    highway::build_highway_edges(full, hcfg, hsrc, hdst);
    highway::CSR hcsr = highway::build_csr_from_edges(n, hsrc, hdst);
    timers["Highway_build"] = now_sec() - t0;

    t0 = now_sec();
    if (max_anchors < 0) {
      max_anchors = std::max(8, std::min(30, n / 5));
    }
    std::vector<int> anchors =
        highway::select_anchors_greedy_dedup(full, max_anchors);
    timers["Anchor_select"] = now_sec() - t0;

    t0 = now_sec();
    highway::TopRState st_init =
        make_init_state(n, std::max(1, pcfg.top_r), anchors);
    timers["State_init"] = now_sec() - t0;

    t0 = now_sec();
    highway::TopRState st =
        highway::propagate_other_assignment_topr_cpu(hcsr, full, anchors, pcfg);
    timers["Propagation"] = now_sec() - t0;

    t0 = now_sec();
    highway::TopRState st_ref =
        highway::refine_uncertain_nodes_set_search_cpu(st, full, hcsr, lrcfg);
    timers["Anchor_preserving_pattern_decoding"] = now_sec() - t0;

    t0 = now_sec();
    auto node_memberships_contig =
        decode_state_labels(st_ref, decode_theta, max_memberships);
    timers["Decode"] = now_sec() - t0;

    t0 = now_sec();
    auto communities_contig =
        build_communities_from_node_memberships(node_memberships_contig);
    write_communities_json("communities.json", el.inv_map, communities_contig);
    timers["WriteCommunitiesJson"] = now_sec() - t0;

    timers["TOTAL"] = now_sec() - t_global0;

    const long long m_undirected = count_undirected_edges_from_csr(full);
    const long long m_backbone_undirected = count_undirected_edges_from_csr(hcsr);
    const long long total_memberships =
        count_total_memberships(node_memberships_contig);

    std::cout << "Summary:\n";
    std::cout << "  n=" << n << "\n";
    std::cout << "  m_undirected=" << m_undirected << "\n";
    std::cout << "  m_backbone_undirected=" << m_backbone_undirected << "\n";
    std::cout << "  backbone_ratio="
              << std::fixed << std::setprecision(6)
              << static_cast<double>(m_backbone_undirected) /
                     std::max(1.0, static_cast<double>(m_undirected))
              << "\n";
    std::cout << "  anchors=" << anchors.size() << "\n";
    std::cout << "  total_memberships=" << total_memberships << "\n";
    std::cout << "  avg_memberships_per_node="
              << std::fixed << std::setprecision(6)
              << static_cast<double>(total_memberships) /
                     std::max(1.0, static_cast<double>(n))
              << "\n";

    std::cout << "Timers:\n";
    for (const auto& kv : timers) {
      std::cout << "  " << kv.first << " = "
                << std::fixed << std::setprecision(6)
                << kv.second << "s\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }

  return 0;
}