// Checks that the C++ GNN forward pass reproduces the Python training model:
// reads the first sample of a collection file, prints scores of its candidates.
#include <cstdint>
#include <cstdio>
#include <vector>
#include "gnn.hpp"
using namespace ganit;
int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    char mg[4]; std::fread(mg, 1, 4, f);
    int32_t h[5]; std::fread(h, 4, 5, f);
    GnnGraph g; g.n = h[0]; g.m = h[1]; int nnz = h[4];
    g.er.resize(nnz); g.ec.resize(nnz); g.ew.resize(nnz);
    std::fread(g.er.data(), 4, nnz, f); std::fread(g.ec.data(), 4, nnz, f); std::fread(g.ew.data(), 4, nnz, f);
    g.row_deg.assign(g.m, 0); g.col_deg.assign(g.n, 0);
    for (int e = 0; e < nnz; ++e) { g.row_deg[g.er[e]] += 1; g.col_deg[g.ec[e]] += 1; }
    for (auto& d : g.row_deg) d = d < 1 ? 1 : d;
    for (auto& d : g.col_deg) d = d < 1 ? 1 : d;
    int32_t kl[2]; std::fread(kl, 4, 2, f);
    std::vector<float> Fv(static_cast<size_t>(g.n) * kGnnFV), Fc(static_cast<size_t>(g.m) * kGnnFC);
    std::fread(Fv.data(), 4, Fv.size(), f); std::fread(Fc.data(), 4, Fc.size(), f);
    std::vector<int32_t> cand(kl[0]); std::fread(cand.data(), 4, kl[0], f);
    GnnModel model; std::string err;
    if (!model.load(argv[2], &err)) { std::printf("%s\n", err.c_str()); return 1; }
    auto s = model.forward(g, Fv, Fc);
    for (int k = 0; k < kl[0] && k < 6; ++k) std::printf("%.10f\n", s[cand[k]]);
}
