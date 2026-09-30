// GANIT GNN branching model (inference), written from scratch.
//
// Bipartite graph per branch-and-bound node:
//   variable nodes  (n, FV features)  -- LP state, pseudocosts, index-aware features
//   constraint nodes(m, FC features)  -- row type, duals, tightness, index-aware features
//   edges           normalised matrix coefficients a_ij / max_k |a_ik|
//
// Architecture (Gasse et al. 2019 style, one round each way):
//   Hv  = relu(Fv Wv1 + bv1)                      embed variables
//   Hc  = relu(Fc Wc1 + bc1)                      embed constraints
//   Hc2 = relu(Hc Wc2 + (E Hv) Wm1 + bc2)         variables -> constraints (mean over row)
//   Hv2 = relu(Hv Wv2 + (G Hc2) Wm2 + bv2)        constraints -> variables (mean over column)
//   s   = relu(Hv2 Wo1 + bo1) wo2                 branching score per variable
// Trained offline (bench/gnn_train.py, NumPy) to imitate strong branching.
#pragma once
#include <string>
#include <vector>

namespace ganit {

constexpr int kGnnFV = 26;  // variable features
constexpr int kGnnFC = 17;  // constraint features

struct GnnGraph {
    int n = 0, m = 0;
    std::vector<int> er, ec;    // edge row / column
    std::vector<float> ew;      // normalised coefficient
    std::vector<float> row_deg, col_deg;
};

class GnnModel {
   public:
    bool load(const std::string& path, std::string* err = nullptr);
    bool loaded() const { return D_ > 0; }
    // Fv: n x FV, Fc: m x FC (raw features; standardised inside). Returns score per variable.
    std::vector<double> forward(const GnnGraph& g, const std::vector<float>& Fv, const std::vector<float>& Fc) const;
    int dim() const { return D_; }

   private:
    int D_ = 0;
    std::vector<double> mean_v_, std_v_, mean_c_, std_c_;
    std::vector<double> Wv1_, bv1_, Wc1_, bc1_, Wc2_, Wm1_, bc2_, Wv2_, Wm2_, bv2_, Wo1_, bo1_, wo2_;
};

}  // namespace ganit
