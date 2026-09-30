#include "gnn.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace ganit {

namespace {

bool read_array(std::istream& in, const std::string& name, int rows, int cols, std::vector<double>& out,
                std::string* err) {
    std::string tag;
    int r = 0, c = 0;
    if (!(in >> tag >> r >> c) || tag != name || r != rows || c != cols) {
        if (err) *err = "model file: expected array '" + name + "' (" + std::to_string(rows) + "x" +
                        std::to_string(cols) + "), found '" + tag + "'";
        return false;
    }
    out.resize(static_cast<size_t>(rows) * cols);
    for (auto& v : out)
        if (!(in >> v)) { if (err) *err = "model file: truncated array " + name; return false; }
    return true;
}

// Y (rows x out) = X (rows x in) * W (in x out) + b ; optional relu
void dense(const std::vector<double>& X, int rows, int in, const std::vector<double>& W, int out,
           const std::vector<double>* b, std::vector<double>& Y, bool accumulate = false) {
    if (!accumulate) Y.assign(static_cast<size_t>(rows) * out, 0.0);
    for (int i = 0; i < rows; ++i) {
        const double* x = &X[static_cast<size_t>(i) * in];
        double* y = &Y[static_cast<size_t>(i) * out];
        for (int k = 0; k < in; ++k) {
            const double xv = x[k];
            if (xv == 0.0) continue;
            const double* w = &W[static_cast<size_t>(k) * out];
            for (int o = 0; o < out; ++o) y[o] += xv * w[o];
        }
        if (b) for (int o = 0; o < out; ++o) y[o] += (*b)[o];
    }
}

void relu(std::vector<double>& v) {
    for (auto& x : v) x = x > 0 ? x : 0.0;
}

}  // namespace

bool GnnModel::load(const std::string& path, std::string* err) {
    std::ifstream in(path);
    if (!in) { if (err) *err = "cannot open model " + path; return false; }
    std::string magic;
    int ver = 0, fv = 0, fc = 0, d = 0;
    in >> magic >> ver >> fv >> fc >> d;
    if (magic != "GANIT_GNN" || ver != 1) { if (err) *err = "not a GANIT GNN model file"; return false; }
    if (fv != kGnnFV || fc != kGnnFC) { if (err) *err = "feature dimension mismatch (retrain the model)"; return false; }
    bool ok = read_array(in, "mean_v", 1, fv, mean_v_, err) && read_array(in, "std_v", 1, fv, std_v_, err) &&
              read_array(in, "mean_c", 1, fc, mean_c_, err) && read_array(in, "std_c", 1, fc, std_c_, err) &&
              read_array(in, "Wv1", fv, d, Wv1_, err) && read_array(in, "bv1", 1, d, bv1_, err) &&
              read_array(in, "Wc1", fc, d, Wc1_, err) && read_array(in, "bc1", 1, d, bc1_, err) &&
              read_array(in, "Wc2", d, d, Wc2_, err) && read_array(in, "Wm1", d, d, Wm1_, err) &&
              read_array(in, "bc2", 1, d, bc2_, err) && read_array(in, "Wv2", d, d, Wv2_, err) &&
              read_array(in, "Wm2", d, d, Wm2_, err) && read_array(in, "bv2", 1, d, bv2_, err) &&
              read_array(in, "Wo1", d, d, Wo1_, err) && read_array(in, "bo1", 1, d, bo1_, err) &&
              read_array(in, "wo2", d, 1, wo2_, err);
    if (ok) D_ = d;
    return ok;
}

std::vector<double> GnnModel::forward(const GnnGraph& g, const std::vector<float>& Fv,
                                      const std::vector<float>& Fc) const {
    const int n = g.n, m = g.m, D = D_;
    std::vector<double> fv(static_cast<size_t>(n) * kGnnFV), fc(static_cast<size_t>(m) * kGnnFC);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < kGnnFV; ++k)
            fv[static_cast<size_t>(i) * kGnnFV + k] = (Fv[static_cast<size_t>(i) * kGnnFV + k] - mean_v_[k]) / std_v_[k];
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < kGnnFC; ++k)
            fc[static_cast<size_t>(i) * kGnnFC + k] = (Fc[static_cast<size_t>(i) * kGnnFC + k] - mean_c_[k]) / std_c_[k];

    std::vector<double> Hv, Hc, Mc(static_cast<size_t>(m) * D, 0.0), Mv(static_cast<size_t>(n) * D, 0.0), Hc2, Hv2, Ho;
    dense(fv, n, kGnnFV, Wv1_, D, &bv1_, Hv);
    relu(Hv);
    dense(fc, m, kGnnFC, Wc1_, D, &bc1_, Hc);
    relu(Hc);
    // Mc = E Hv  (mean over the row's variables, signed by coefficient)
    for (size_t e = 0; e < g.ew.size(); ++e) {
        const int r = g.er[e], c = g.ec[e];
        const double w = g.ew[e] / g.row_deg[r];
        const double* h = &Hv[static_cast<size_t>(c) * D];
        double* o = &Mc[static_cast<size_t>(r) * D];
        for (int k = 0; k < D; ++k) o[k] += w * h[k];
    }
    dense(Hc, m, D, Wc2_, D, &bc2_, Hc2);
    dense(Mc, m, D, Wm1_, D, nullptr, Hc2, true);
    relu(Hc2);
    // Mv = G Hc2 (mean over the column's constraints)
    for (size_t e = 0; e < g.ew.size(); ++e) {
        const int r = g.er[e], c = g.ec[e];
        const double w = g.ew[e] / g.col_deg[c];
        const double* h = &Hc2[static_cast<size_t>(r) * D];
        double* o = &Mv[static_cast<size_t>(c) * D];
        for (int k = 0; k < D; ++k) o[k] += w * h[k];
    }
    dense(Hv, n, D, Wv2_, D, &bv2_, Hv2);
    dense(Mv, n, D, Wm2_, D, nullptr, Hv2, true);
    relu(Hv2);
    dense(Hv2, n, D, Wo1_, D, &bo1_, Ho);
    relu(Ho);
    std::vector<double> s(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* h = &Ho[static_cast<size_t>(i) * D];
        double t = 0.0;
        for (int k = 0; k < D; ++k) t += h[k] * wo2_[k];
        s[i] = t;
    }
    return s;
}

}  // namespace ganit
