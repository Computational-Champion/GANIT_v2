#include "mip.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>

#include "simplex.hpp"
#include "gnn.hpp"
#include <cstdint>
#include <cstring>
#include <map>
#include <string>

namespace ganit {

const char* to_string(MipStatus s) {
    switch (s) {
        case MipStatus::Optimal: return "OPTIMAL";
        case MipStatus::Infeasible: return "INFEASIBLE";
        case MipStatus::Unbounded: return "UNBOUNDED";
        case MipStatus::TimeLimit: return "TIME_LIMIT";
        case MipStatus::NodeLimit: return "NODE_LIMIT";
        default: return "NUMERICAL_ERROR";
    }
}

namespace {

using Clock = std::chrono::steady_clock;

struct BoundChange { int j; double lo, hi; };

// Outward rounding: round-to-nearest has error <= 0.5 ulp, so stepping one ulp
// outward gives a rigorous bound on the exact result of the operation.
inline double dn(double v) { return std::nextafter(v, -kInf); }
inline double up(double v) { return std::nextafter(v, kInf); }

struct StoredCut { std::vector<int> idx; std::vector<double> val; double lo, hi; };

struct Node {
    double bound;      // certified bound (used for pruning and for the reported global bound)
    double est = -kInf;  // search-order estimate (may use uncertified strong-branching values)
    int depth;
    std::vector<BoundChange> changes;
    std::shared_ptr<DualSimplex::Basis> basis;
    // for pseudocost updates
    double parent_obj = -kInf;
    int branch_var = -1;
    int branch_dir = 0;  // 0 down, 1 up
    double branch_frac = 0;
};

struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        double ka = std::max(a.bound, a.est), kb = std::max(b.bound, b.est);
        if (ka != kb) return ka > kb;  // min-heap on the ordering key
        return a.depth < b.depth;
    }
};

class Mip {
   public:
    Mip(const LP& lp, const MipOptions& opt) : lp_(lp), opt_(opt), S_(lp) {
        t0_ = Clock::now();
        const int n = lp.n;
        for (int j = 0; j < n; ++j)
            if (j < static_cast<int>(lp.is_int.size()) && lp.is_int[j]) ints_.push_back(j);
        is_int_.assign(n, 0);
        for (int j : ints_) is_int_[j] = 1;
        // integral bounds for integer columns
        for (int j : ints_) {
            double lo = lp.l[j], hi = lp.u[j];
            if (std::isfinite(lo)) lo = std::ceil(lo - 1e-9);
            if (std::isfinite(hi)) hi = std::floor(hi + 1e-9);
            S_.set_col_bounds(j, lo, hi);
        }
        exact_lb_ = lp.l;
        exact_ub_ = lp.u;
        for (int j : ints_) {
            if (std::isfinite(exact_lb_[j])) exact_lb_[j] = std::ceil(exact_lb_[j] - 1e-9);
            if (std::isfinite(exact_ub_[j])) exact_ub_[j] = std::floor(exact_ub_[j] + 1e-9);
        }
        root_lb_.resize(n);
        root_ub_.resize(n);
        for (int j = 0; j < n; ++j) { root_lb_[j] = S_.col_lower(j); root_ub_[j] = S_.col_upper(j); }
        compute_implied_bounds();
        cmax_ = 0.0;
        for (double v : lp.c) cmax_ = std::max(cmax_, std::fabs(v));
        if (cmax_ == 0.0) cmax_ = 1.0;
        index_features(lp.col_names, var_class_, var_pos_);
        index_features(lp.row_names, row_class_, row_pos_);
        if (opt.branching == Branching::Gnn || opt.branching == Branching::GnnStrong) {
            std::string err;
            if (!gnn_.load(opt.gnn_model, &err)) throw std::runtime_error("GNN model: " + err);
        }
        if (!opt.collect_path.empty()) {
            collect_fp_ = std::fopen(opt.collect_path.c_str(), "wb");
            if (!collect_fp_) throw std::runtime_error("cannot write " + opt.collect_path);
        }
        pc_sum_[0].assign(n, 0.0); pc_sum_[1].assign(n, 0.0);
        pc_n_[0].assign(n, 0); pc_n_[1].assign(n, 0);
        sign_ = lp.maximize ? -1.0 : 1.0;
    }

    MipResult run();
    ~Mip() { if (collect_fp_) std::fclose(collect_fp_); }

   private:
    const LP& lp_;
    MipOptions opt_;
    DualSimplex S_;
    Clock::time_point t0_;
    std::vector<int> ints_;
    std::vector<char> is_int_;
    std::vector<double> root_lb_, root_ub_;
    double inc_ = kInf;  // min-sense
    std::vector<double> inc_x_;
    std::vector<double> pc_sum_[2];
    std::vector<int> pc_n_[2];
    double sign_ = 1.0;
    long nodes_ = 0;
    int cuts_ = 0;
    std::vector<StoredCut> cutpool_;          // exact cut data as added to the LP
    std::vector<double> exact_lb_, exact_ub_; // exact root column bounds (original units)
    long certified_prunes_ = 0, uncertifiable_ = 0, uncertified_prunes_ = 0;
    std::vector<double> implied_lb_, implied_ub_;  // bounds implied by the rows (valid for all nodes)

    // ---------------- GNN branching support
    GnnModel gnn_;
    GnnGraph graph_;
    bool graph_built_ = false;
    std::vector<int> var_class_, row_class_;
    std::vector<float> var_pos_, row_pos_;   // 3 relative index positions per entity
    std::vector<double> rownorm_;
    double cmax_ = 1.0;
    std::FILE* collect_fp_ = nullptr;
    int samples_ = 0;
    int cur_depth_ = 0;
    double branch_time_ = 0.0;
    struct SbRecord { int j; double score; };
    std::vector<SbRecord> sb_record_;
    bool best_score_ignore_rest_ = false;

    static uint32_t fnv(const std::string& s) {
        uint32_t h = 2166136261u;
        for (char ch : s) { h ^= static_cast<unsigned char>(ch); h *= 16777619u; }
        return h;
    }
    // "X_3_1_7" -> class hash of "X", numeric indices {3,1,7}
    static void parse_name(const std::string& nm, std::string& prefix, std::vector<int>& idx) {
        idx.clear();
        size_t start = 0;
        bool first = true;
        while (start <= nm.size()) {
            size_t e = nm.find('_', start);
            std::string tok = nm.substr(start, e == std::string::npos ? std::string::npos : e - start);
            if (first) { prefix = tok; first = false; }
            else if (!tok.empty() && tok.find_first_not_of("0123456789") == std::string::npos) idx.push_back(std::stoi(tok));
            if (e == std::string::npos) break;
            start = e + 1;
        }
    }
    static void index_features(const std::vector<std::string>& names, std::vector<int>& cls, std::vector<float>& pos) {
        const size_t N = names.size();
        cls.assign(N, 0);
        pos.assign(N * 3, 0.0f);
        std::vector<std::string> pre(N);
        std::vector<std::vector<int>> ids(N);
        std::map<std::string, std::vector<int>> maxi;
        for (size_t i = 0; i < N; ++i) {
            parse_name(names[i], pre[i], ids[i]);
            auto& mx = maxi[pre[i]];
            if (mx.size() < ids[i].size()) mx.resize(ids[i].size(), 0);
            for (size_t k = 0; k < ids[i].size(); ++k) mx[k] = std::max(mx[k], ids[i][k]);
        }
        for (size_t i = 0; i < N; ++i) {
            cls[i] = static_cast<int>(fnv(pre[i]) % 8u);
            const auto& mx = maxi[pre[i]];
            for (size_t k = 0; k < std::min<size_t>(3, ids[i].size()); ++k)
                pos[i * 3 + k] = mx[k] > 0 ? static_cast<float>(ids[i][k]) / mx[k] : 0.0f;
        }
    }
    void build_graph();
    void compute_features(const std::vector<double>& x, const std::vector<int>& cand, std::vector<float>& Fv,
                          std::vector<float>& Fc);
    void write_sample(const std::vector<double>& x);

    // Activity-based bound propagation over the original rows, with outward
    // rounding and a small safety margin. The results only enter the safe bound.
    void compute_implied_bounds() {
        const int n = lp_.n;
        implied_lb_ = exact_lb_;
        implied_ub_ = exact_ub_;
        for (int round = 0; round < 8; ++round) {
            bool changed = false;
            for (int i = 0; i < lp_.m; ++i) {
                const int b = lp_.A.ptr[i], e = lp_.A.ptr[i + 1];
                double minact = 0, maxact = 0;
                int ninf_min = 0, ninf_max = 0;
                for (int p = b; p < e; ++p) {
                    double a = lp_.A.val[p];
                    int j = lp_.A.idx[p];
                    double lo = a > 0 ? implied_lb_[j] : implied_ub_[j];
                    double hi = a > 0 ? implied_ub_[j] : implied_lb_[j];
                    if (std::isfinite(lo)) minact = dn(minact + dn(a * lo)); else ++ninf_min;
                    if (std::isfinite(hi)) maxact = up(maxact + up(a * hi)); else ++ninf_max;
                }
                for (int p = b; p < e; ++p) {
                    double a = lp_.A.val[p];
                    int j = lp_.A.idx[p];
                    double lo_c = a > 0 ? implied_lb_[j] : implied_ub_[j];
                    double hi_c = a > 0 ? implied_ub_[j] : implied_lb_[j];
                    // residual activities without column j
                    bool minf = std::isfinite(lo_c) ? ninf_min == 0 : ninf_min == 1;
                    bool maxf = std::isfinite(hi_c) ? ninf_max == 0 : ninf_max == 1;
                    double rmin = std::isfinite(lo_c) ? dn(minact - up(a * lo_c)) : minact;
                    double rmax = std::isfinite(hi_c) ? up(maxact - dn(a * hi_c)) : maxact;
                    double nl = -kInf, nu = kInf;
                    // a x_j <= hi - rmin   and   a x_j >= lo - rmax
                    if (std::isfinite(lp_.hi[i]) && minf) {
                        double t = up(lp_.hi[i] - rmin);
                        if (a > 0) nu = up(t / a); else nl = dn(t / a);
                    }
                    if (std::isfinite(lp_.lo[i]) && maxf) {
                        double t = dn(lp_.lo[i] - rmax);
                        if (a > 0) nl = std::max(nl, dn(t / a)); else nu = std::min(nu, up(t / a));
                    }
                    auto widen_up = [](double v) { return up(v + 1e-9 * (1.0 + std::fabs(v))); };
                    auto widen_dn = [](double v) { return dn(v - 1e-9 * (1.0 + std::fabs(v))); };
                    if (std::isfinite(nu)) { nu = widen_up(nu); if (nu < implied_ub_[j] && !std::isfinite(implied_ub_[j])) { implied_ub_[j] = nu; changed = true; } }
                    if (std::isfinite(nl)) { nl = widen_dn(nl); if (nl > implied_lb_[j] && !std::isfinite(implied_lb_[j])) { implied_lb_[j] = nl; changed = true; } }
                }
            }
            if (!changed) break;
        }
        int fin = 0;
        for (int j = 0; j < n; ++j) fin += std::isfinite(implied_lb_[j]) && std::isfinite(implied_ub_[j]);
        if (opt_.verbose && opt_.safe_bounds)
            std::printf("Safe-bound box: %d of %d columns have finite (original or implied) bounds\n", fin, n);
    }
    double last_log_ = -1e9;

    double elapsed() const { return std::chrono::duration<double>(Clock::now() - t0_).count(); }
    double zmin() const { return sign_ * S_.objective(); }
    double cutoff() const {
        if (!std::isfinite(inc_)) return kInf;
        return inc_ - std::max(opt_.abs_gap, opt_.rel_gap * std::max(1.0, std::fabs(inc_)));
    }
    static double frac(double v) { return v - std::floor(v); }

    int fractional(const std::vector<double>& x, std::vector<int>* out) const {
        int cnt = 0;
        for (int j : ints_) {
            double f = frac(x[j]);
            if (f > opt_.int_tol && f < 1.0 - opt_.int_tol) { ++cnt; if (out) out->push_back(j); }
        }
        return cnt;
    }

    // Verify a candidate against the ORIGINAL model (not the cut-extended LP).
    bool try_incumbent(std::vector<double> x, const char* source) {
        for (int j : ints_) x[j] = std::round(x[j]);
        const double ft = opt_.feas_tol;
        for (int j = 0; j < lp_.n; ++j) {
            double lo = lp_.l[j], hi = lp_.u[j];
            if (x[j] < lo - ft * (1 + std::fabs(lo)) || x[j] > hi + ft * (1 + std::fabs(hi))) return false;
            x[j] = std::min(std::max(x[j], lo), hi);
        }
        for (int i = 0; i < lp_.m; ++i) {
            double a = 0;
            for (int p = lp_.A.ptr[i]; p < lp_.A.ptr[i + 1]; ++p) a += lp_.A.val[p] * x[lp_.A.idx[p]];
            if (a < lp_.lo[i] - ft * (1 + std::fabs(lp_.lo[i])) || a > lp_.hi[i] + ft * (1 + std::fabs(lp_.hi[i])))
                return false;
        }
        double obj = lp_.obj_const;
        for (int j = 0; j < lp_.n; ++j) obj += lp_.c[j] * x[j];
        obj *= sign_;
        if (!std::isfinite(inc_) || obj < inc_ - 1e-9 * (1 + std::fabs(inc_))) {
            inc_ = obj;
            inc_x_ = x;
            if (opt_.verbose)
                std::printf("  * new incumbent %.10g (%s) at node %ld, %.2fs\n", sign_ * inc_, source, nodes_, elapsed());
            return true;
        }
        return false;
    }

    void pc_update(int j, int dir, double gain, double f) {
        if (!(f > 1e-9) || !std::isfinite(gain)) return;
        pc_sum_[dir][j] += std::max(gain, 0.0) / f;
        pc_n_[dir][j] += 1;
    }
    double pc(int j, int dir) const {
        if (pc_n_[dir][j] > 0) return pc_sum_[dir][j] / pc_n_[dir][j];
        double s = 0; long c = 0;
        for (int k : ints_) if (pc_n_[dir][k] > 0) { s += pc_sum_[dir][k] / pc_n_[dir][k]; ++c; }
        return c ? s / c : 1.0;
    }

    void apply_bounds(const std::vector<BoundChange>& ch) {
        for (int j : ints_) S_.set_col_bounds(j, root_lb_[j], root_ub_[j]);
        for (auto& b : ch) S_.set_col_bounds(b.j, b.lo, b.hi);
    }

    // Solve the current node LP. A Cutoff returned by the simplex is only accepted
    // when a certified bound confirms it; otherwise the LP is solved to optimality.
    LpStatus solve_node(const std::vector<BoundChange>& ch) {
        LpStatus st = S_.solve(-1, cutoff(), opt_.time_limit - elapsed());
        if (st == LpStatus::Cutoff && opt_.safe_bounds) {
            double sb = safe_bound(ch);
            if (sb >= cutoff()) { ++certified_prunes_; return st; }
            if (!std::isfinite(sb)) { ++uncertified_prunes_; return st; }
            st = S_.solve(-1, kInf, opt_.time_limit - elapsed());
        }
        return st;
    }
    long infeasible_prunes_ = 0;
    int gomory_round();
    double safe_bound(const std::vector<BoundChange>& changes);
    // prune test: approximate bound first (cheap), then certify
    bool prunable(double z_approx, const std::vector<BoundChange>& changes, double* sb_out = nullptr) {
        if (!(z_approx >= cutoff())) return false;
        if (!opt_.safe_bounds) return true;
        double sb = safe_bound(changes);
        if (sb_out) *sb_out = sb;
        if (sb >= cutoff()) { ++certified_prunes_; return true; }
        if (!std::isfinite(sb)) { ++uncertified_prunes_; return true; }  // cannot certify: fall back, but count it
        return false;
    }
    void dive();

    // Branching decision. Returns: -2 node infeasible, -1 bound tightened (re-solve), else var.
    int select_branch(const std::vector<double>& x, double z, std::vector<BoundChange>& changes,
                      double& est_down, double& est_up, double& cert_down, double& cert_up);

    void log(size_t open, double bound, bool force = false) {
        if (!opt_.verbose) return;
        double t = elapsed();
        if (!force && t - last_log_ < opt_.log_interval) return;
        last_log_ = t;
        double gap = std::isfinite(inc_) ? (inc_ - bound) / std::max(1.0, std::fabs(inc_)) : kInf;
        std::printf("%9ld nodes %8zu open | incumbent %16.8g | bound %16.8g | gap %8.3f%% | %7.1fs\n", nodes_, open,
                    std::isfinite(inc_) ? sign_ * inc_ : NAN, sign_ * bound, 100 * gap, t);
    }
};

// Rigorous Lagrangian (Neumaier-Shcherbina) lower bound for the node LP defined by
// the original rows, the cut pool and the node's column bounds, from ANY duals y:
//   c'x = y'(Ax) + (c - A'y)'x  >=  sum_i min(y_i lo_i, y_i hi_i) + sum_j min_{x_j in [l_j,u_j]} d_j x_j
// Every floating-point operation is rounded outward, d_j is carried as an interval.
double Mip::safe_bound(const std::vector<BoundChange>& changes) {
    const int n = lp_.n;
    std::vector<double> y = S_.row_duals_orig();
    std::vector<double> lb = exact_lb_, ub = exact_ub_;
    for (auto& b : changes) { lb[b.j] = b.lo; ub[b.j] = b.hi; }
    for (int j = 0; j < n; ++j) {
        if (!std::isfinite(lb[j])) lb[j] = implied_lb_[j];
        if (!std::isfinite(ub[j])) ub[j] = implied_ub_[j];
    }
    std::vector<double> dlo(n), dhi(n);
    for (int j = 0; j < n; ++j) dlo[j] = dhi[j] = sign_ * lp_.c[j];
    double rowsum = 0.0;
    auto process_row = [&](double yi, double lo, double hi, const int* idx, const double* val, int len) -> bool {
        if (!std::isfinite(lo)) yi = std::min(yi, 0.0);
        if (!std::isfinite(hi)) yi = std::max(yi, 0.0);
        if (yi == 0.0) return true;
        double term = yi > 0 ? dn(yi * lo) : dn(yi * hi);
        if (!std::isfinite(term)) return false;
        rowsum = dn(rowsum + term);
        for (int k = 0; k < len; ++k) {
            double t = yi * val[k];
            double tl = dn(t), th = up(t);
            int j = idx[k];
            dlo[j] = dn(dlo[j] - th);
            dhi[j] = up(dhi[j] - tl);
        }
        return true;
    };
    for (int i = 0; i < lp_.m; ++i) {
        int b = lp_.A.ptr[i], e = lp_.A.ptr[i + 1];
        if (!process_row(y[i], lp_.lo[i], lp_.hi[i], &lp_.A.idx[b], &lp_.A.val[b], e - b)) return -kInf;
    }
    for (size_t c = 0; c < cutpool_.size(); ++c) {
        const StoredCut& ct = cutpool_[c];
        if (!process_row(y[lp_.m + c], ct.lo, ct.hi, ct.idx.data(), ct.val.data(), static_cast<int>(ct.idx.size())))
            return -kInf;
    }
    double colsum = 0.0;
    for (int j = 0; j < n; ++j) {
        const double a = dlo[j], b = dhi[j], l = lb[j], u = ub[j];
        if ((!std::isfinite(l) && b > 0) || (!std::isfinite(u) && a < 0)) {
            ++uncertifiable_;
            return -kInf;
        }
        double best = kInf;
        const double ds[2] = {a, b}, xs[2] = {l, u};
        for (double dv : ds)
            for (double xv : xs) {
                if (!std::isfinite(xv)) continue;
                best = std::min(best, dn(dv * xv));
            }
        if (!std::isfinite(best)) best = 0.0;  // both bounds infinite and d == [0,0]
        colsum = dn(colsum + best);
    }
    return dn(dn(rowsum + colsum) + sign_ * lp_.obj_const);
}

int Mip::gomory_round() {
    const int n = S_.num_cols();
    const int m = S_.num_rows();
    std::vector<double> x = S_.primal();
    struct Cand { int p; double score; };
    std::vector<Cand> cand;
    for (int p = 0; p < m; ++p) {
        int h = S_.head(p);
        if (h >= n || !is_int_[h]) continue;
        double f = frac(x[h]);
        if (f < 0.005 || f > 0.995) continue;
        cand.push_back({p, std::fabs(f - 0.5)});
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });
    if (static_cast<int>(cand.size()) > opt_.max_cuts_per_round) cand.resize(opt_.max_cuts_per_round);

    int added = 0;
    std::vector<double> coef, g(n);
    std::vector<int> ridx;
    std::vector<double> rval;
    for (auto& c : cand) {
        int h = S_.tableau_row_orig(c.p, coef);
        double b = S_.var_value_orig(h);
        double f0 = frac(b);
        if (f0 < 0.005 || f0 > 0.995) continue;
        std::fill(g.begin(), g.end(), 0.0);
        double rhs = 1.0;
        bool ok = true;
        const int NN = static_cast<int>(coef.size());
        std::vector<std::pair<int, double>> logical_terms;
        for (int j = 0; j < NN && ok; ++j) {
            double a = coef[j];
            if (std::fabs(a) < 1e-11) continue;
            auto st = S_.status(j);
            if (st == DualSimplex::AT_FREE || S_.is_artificial_bound(j)) { ok = false; break; }
            bool at_lb = st == DualSimplex::AT_LB;
            double bnd = at_lb ? S_.var_lower_orig(j) : S_.var_upper_orig(j);
            if (!std::isfinite(bnd)) { ok = false; break; }
            double ap = at_lb ? a : -a;
            double pi;
            if (j < n && is_int_[j]) {
                double fj = frac(ap);
                pi = fj <= f0 ? fj / f0 : (1.0 - fj) / (1.0 - f0);
            } else {
                pi = ap >= 0 ? ap / f0 : -ap / (1.0 - f0);
            }
            if (pi == 0.0) continue;
            // pi * xt, xt = x - lb (at lb) or ub - x (at ub)
            double gj = at_lb ? pi : -pi;
            rhs += at_lb ? pi * bnd : -pi * bnd;
            if (j < n) g[j] += gj;
            else logical_terms.emplace_back(j - n, gj);
        }
        if (!ok) continue;
        for (auto& lt : logical_terms) {
            S_.row_orig(lt.first, ridx, rval);
            for (size_t k = 0; k < ridx.size(); ++k) g[ridx[k]] += lt.second * rval[k];
        }
        // clean tiny coefficients using bounds; check dynamism
        double gmax = 0;
        for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax <= 0) continue;
        std::vector<int> idx;
        std::vector<double> val;
        double gmin = kInf;
        for (int j = 0; j < n && ok; ++j) {
            double v = g[j];
            if (v == 0.0) continue;
            if (std::fabs(v) < 1e-9 * gmax) {
                // drop term keeping validity of  sum g x >= rhs
                double lo = S_.col_lower(j), hi = S_.col_upper(j);
                double worst = v > 0 ? v * hi : v * lo;  // max of v*x over the box
                if (!std::isfinite(worst)) { ok = false; break; }
                rhs -= worst;
                continue;
            }
            idx.push_back(j);
            val.push_back(v);
            gmin = std::min(gmin, std::fabs(v));
        }
        if (!ok || idx.empty() || gmax / gmin > 1e7) continue;
        double act = 0, nrm = 0;
        for (size_t k = 0; k < idx.size(); ++k) { act += val[k] * x[idx[k]]; nrm += val[k] * val[k]; }
        double viol = rhs - act;
        if (viol <= 1e-6 * (1 + std::fabs(rhs)) || viol / std::sqrt(nrm) < 1e-5) continue;
        S_.add_row(idx, val, rhs, kInf);
        cutpool_.push_back({idx, val, rhs, kInf});
        ++added;
    }
    return added;
}

void Mip::dive() {
    auto saved = S_.save_state();
    std::vector<double> x = S_.primal();
    for (int depth = 0; depth < 100; ++depth) {
        if (elapsed() > opt_.time_limit) break;
        std::vector<int> fr;
        if (fractional(x, &fr) == 0) { try_incumbent(x, "diving"); break; }
        int best = -1;
        double bf = 2;
        for (int j : fr) {
            double f = frac(x[j]);
            double d = std::min(f, 1 - f);
            if (d < bf) { bf = d; best = j; }
        }
        double v = x[best];
        bool up = frac(v) >= 0.5;
        double lo = S_.col_lower(best), hi = S_.col_upper(best);
        if (up) S_.set_col_bounds(best, std::ceil(v), hi); else S_.set_col_bounds(best, lo, std::floor(v));
        LpStatus st = S_.solve(2000, cutoff());
        if (st != LpStatus::Optimal) {
            S_.set_col_bounds(best, lo, hi);
            if (up) S_.set_col_bounds(best, lo, std::floor(v)); else S_.set_col_bounds(best, std::ceil(v), hi);
            st = S_.solve(2000, cutoff());
            if (st != LpStatus::Optimal) break;
        }
        x = S_.primal();
    }
    S_.restore_state(*saved);
}

// ---------------------------------------------------------------- GNN features
void Mip::build_graph() {
    const int n = lp_.n;
    const int m = S_.num_rows();  // original rows + root cuts (fixed after the root)
    graph_ = GnnGraph();
    graph_.n = n;
    graph_.m = m;
    rownorm_.assign(m, 0.0);
    auto add_row = [&](int r, const int* idx, const double* val, int len) {
        double mx = 0.0;
        for (int k = 0; k < len; ++k) mx = std::max(mx, std::fabs(val[k]));
        rownorm_[r] = mx > 0 ? mx : 1.0;
        for (int k = 0; k < len; ++k) {
            graph_.er.push_back(r);
            graph_.ec.push_back(idx[k]);
            graph_.ew.push_back(static_cast<float>(val[k] / rownorm_[r]));
        }
    };
    for (int i = 0; i < lp_.m; ++i) {
        int b = lp_.A.ptr[i], e = lp_.A.ptr[i + 1];
        add_row(i, &lp_.A.idx[b], &lp_.A.val[b], e - b);
    }
    for (size_t c = 0; c < cutpool_.size(); ++c)
        add_row(lp_.m + static_cast<int>(c), cutpool_[c].idx.data(), cutpool_[c].val.data(),
                static_cast<int>(cutpool_[c].idx.size()));
    graph_.row_deg.assign(m, 0.0f);
    graph_.col_deg.assign(n, 0.0f);
    for (size_t e = 0; e < graph_.er.size(); ++e) { graph_.row_deg[graph_.er[e]] += 1; graph_.col_deg[graph_.ec[e]] += 1; }
    for (auto& d : graph_.row_deg) d = std::max(d, 1.0f);
    for (auto& d : graph_.col_deg) d = std::max(d, 1.0f);
    graph_built_ = true;
}

static inline float slog(double v) {  // signed log compression
    return static_cast<float>(v >= 0 ? std::log1p(v) : -std::log1p(-v));
}

void Mip::compute_features(const std::vector<double>& x, const std::vector<int>& cand, std::vector<float>& Fv,
                           std::vector<float>& Fc) {
    const int n = lp_.n, m = graph_.m;
    Fv.assign(static_cast<size_t>(n) * kGnnFV, 0.0f);
    Fc.assign(static_cast<size_t>(m) * kGnnFC, 0.0f);
    std::vector<char> is_cand(n, 0);
    for (int j : cand) is_cand[j] = 1;
    double pcavg = 0.0;
    int pcc = 0;
    for (int k : ints_) { pcavg += pc(k, 0) + pc(k, 1); pcc += 2; }
    pcavg = pcc ? std::max(pcavg / pcc, 1e-9) : 1.0;
    const float depth = static_cast<float>(cur_depth_) / (cur_depth_ + 10.0f);
    const float has_inc = std::isfinite(inc_) ? 1.0f : 0.0f;
    for (int j = 0; j < n; ++j) {
        float* f = &Fv[static_cast<size_t>(j) * kGnnFV];
        const double lo = S_.col_lower(j), hi = S_.col_upper(j);
        const auto st = S_.status(j);
        f[0] = static_cast<float>(sign_ * lp_.c[j] / cmax_);
        f[1] = std::max(-10.0f, std::min(10.0f, static_cast<float>(S_.reduced_cost_scaled(j))));
        if (is_int_[j]) {
            double fr = frac(x[j]);
            f[2] = static_cast<float>(fr);
            f[3] = static_cast<float>(std::min(fr, 1.0 - fr));
        }
        f[4] = is_cand[j];
        f[5] = st == DualSimplex::AT_LB;
        f[6] = st == DualSimplex::AT_UB;
        f[7] = st == DualSimplex::BASIC;
        f[8] = is_int_[j];
        if (std::isfinite(lo) && std::isfinite(hi) && hi > lo) f[9] = static_cast<float>((x[j] - lo) / (hi - lo));
        if (is_int_[j]) {
            f[10] = slog(pc(j, 0) / pcavg);
            f[11] = slog(pc(j, 1) / pcavg);
            f[12] = std::min(1.0f, std::min(pc_n_[0][j], pc_n_[1][j]) / static_cast<float>(std::max(1, opt_.reliability)));
        }
        f[13] = depth;
        f[14] = has_inc;
        f[15 + var_class_[j]] = 1.0f;  // 15..22 class one-hot
        for (int k = 0; k < 3; ++k) f[23 + k] = var_pos_[static_cast<size_t>(j) * 3 + k];
    }
    std::vector<double> y = S_.row_duals_orig();
    for (int i = 0; i < m; ++i) {
        float* f = &Fc[static_cast<size_t>(i) * kGnnFC];
        const double lo = S_.var_lower_orig(n + i), hi = S_.var_upper_orig(n + i);
        const double act = S_.var_value_orig(n + i);
        f[0] = (std::isfinite(lo) && std::isfinite(hi) && std::fabs(hi - lo) <= 1e-12 * (1 + std::fabs(lo))) ? 1.0f : 0.0f;
        double rhs = std::isfinite(hi) ? hi : (std::isfinite(lo) ? lo : 0.0);
        f[1] = slog(rhs / rownorm_[i]);
        f[2] = slog(y[i] * rownorm_[i] / cmax_);
        bool tight = (std::isfinite(lo) && std::fabs(act - lo) <= 1e-6 * (1 + std::fabs(lo))) ||
                     (std::isfinite(hi) && std::fabs(act - hi) <= 1e-6 * (1 + std::fabs(hi)));
        f[3] = tight;
        f[4] = i >= lp_.m;
        f[5] = static_cast<float>(std::log1p(graph_.row_deg[i]) / 10.0);
        if (i < lp_.m) {
            f[6 + row_class_[i]] = 1.0f;  // 6..13
            for (int k = 0; k < 3; ++k) f[14 + k] = row_pos_[static_cast<size_t>(i) * 3 + k];
        } else {
            f[6 + (fnv("CUT") % 8u)] = 1.0f;
        }
    }
}

// Sample file layout (little-endian):
//   header (once): "GNS1", int32 n, m, FV, FC, nnz, int32 er[nnz], int32 ec[nnz], float32 ew[nnz]
//   sample:        int32 K, label, float32 Fv[n*FV], float32 Fc[m*FC], int32 cand[K], float32 score[K]
void Mip::write_sample(const std::vector<double>& x) {
    if (!collect_fp_ || sb_record_.size() < 2 || samples_ >= opt_.collect_max) return;
    if (!graph_built_) build_graph();
    if (samples_ == 0) {
        std::fwrite("GNS1", 1, 4, collect_fp_);
        int32_t hdr[5] = {graph_.n, graph_.m, kGnnFV, kGnnFC, static_cast<int32_t>(graph_.ew.size())};
        std::fwrite(hdr, sizeof(int32_t), 5, collect_fp_);
        std::fwrite(graph_.er.data(), sizeof(int32_t), graph_.er.size(), collect_fp_);
        std::fwrite(graph_.ec.data(), sizeof(int32_t), graph_.ec.size(), collect_fp_);
        std::fwrite(graph_.ew.data(), sizeof(float), graph_.ew.size(), collect_fp_);
    }
    std::vector<int> cand;
    std::vector<float> score;
    int label = 0;
    for (size_t k = 0; k < sb_record_.size(); ++k) {
        cand.push_back(sb_record_[k].j);
        score.push_back(static_cast<float>(sb_record_[k].score));
        if (sb_record_[k].score > sb_record_[label].score) label = static_cast<int>(k);
    }
    std::vector<float> Fv, Fc;
    compute_features(x, cand, Fv, Fc);
    int32_t kl[2] = {static_cast<int32_t>(cand.size()), label};
    std::fwrite(kl, sizeof(int32_t), 2, collect_fp_);
    std::fwrite(Fv.data(), sizeof(float), Fv.size(), collect_fp_);
    std::fwrite(Fc.data(), sizeof(float), Fc.size(), collect_fp_);
    std::fwrite(cand.data(), sizeof(int32_t), cand.size(), collect_fp_);
    std::fwrite(score.data(), sizeof(float), score.size(), collect_fp_);
    ++samples_;
}

int Mip::select_branch(const std::vector<double>& x, double z, std::vector<BoundChange>& changes,
                       double& est_down, double& est_up, double& cert_down, double& cert_up) {
    cert_down = cert_up = -kInf;
    std::vector<int> fr;
    fractional(x, &fr);
    const double eps = 1e-6;
    sb_record_.clear();
    const auto tb = Clock::now();
    struct Timer { const Clock::time_point& t; double& acc;
        ~Timer() { acc += std::chrono::duration<double>(Clock::now() - t).count(); } } timer{tb, branch_time_};

    if (opt_.branching == Branching::Gnn && collect_fp_ == nullptr) {
        if (!graph_built_) build_graph();
        std::vector<float> Fv, Fc;
        compute_features(x, fr, Fv, Fc);
        std::vector<double> sc = gnn_.forward(graph_, Fv, Fc);
        int best = fr[0];
        for (int j : fr) if (sc[j] > sc[best]) best = j;
        const double f = frac(x[best]);
        est_down = z + pc(best, 0) * f;
        est_up = z + pc(best, 1) * (1 - f);
        return best;
    }
    std::vector<double> gnn_rank;  // GnnStrong: GNN scores used to rank strong-branching candidates
    if (opt_.branching == Branching::GnnStrong && collect_fp_ == nullptr) {
        if (!graph_built_) build_graph();
        std::vector<float> Fv, Fc;
        compute_features(x, fr, Fv, Fc);
        gnn_rank = gnn_.forward(graph_, Fv, Fc);
    }
    const bool full_sb = opt_.branching == Branching::FullStrong || collect_fp_ != nullptr;
    const bool no_sb = opt_.branching == Branching::Pseudocost && collect_fp_ == nullptr;
    struct C { int j; double score; bool reliable; };
    std::vector<C> cs;
    for (int j : fr) {
        double f = frac(x[j]);
        double sd = pc(j, 0) * f, su = pc(j, 1) * (1 - f);
        bool rel = std::min(pc_n_[0][j], pc_n_[1][j]) >= opt_.reliability;
        cs.push_back({j, std::max(sd, eps) * std::max(su, eps), rel});
    }
    std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.score > b.score; });
    if (full_sb) {
        // expert policy: strong-branch the most fractional candidates, ignoring reliability
        std::sort(cs.begin(), cs.end(), [&](const C& a, const C& b) {
            double fa = frac(x[a.j]), fb = frac(x[b.j]);
            return std::min(fa, 1 - fa) > std::min(fb, 1 - fb);
        });
        for (auto& c : cs) c.reliable = false;
    }
    if (no_sb) for (auto& c : cs) c.reliable = true;
    int sb_cap = full_sb ? opt_.full_sb_candidates : opt_.sb_candidates;
    if (!gnn_rank.empty()) {
        std::sort(cs.begin(), cs.end(), [&](const C& a, const C& b) { return gnn_rank[a.j] > gnn_rank[b.j]; });
        for (auto& c : cs) c.reliable = false;
        sb_cap = opt_.gnn_sb_k;
        best_score_ignore_rest_ = true;
    } else {
        best_score_ignore_rest_ = false;
    }

    int best = cs[0].j;
    double best_score = -1;
    est_down = est_up = z;
    std::unique_ptr<DualSimplex::State> saved;
    int sb_done = 0;
    for (auto& c : cs) {
        if (best_score_ignore_rest_ && sb_done >= sb_cap) break;  // GnnStrong: only the top-k are considered
        if (c.reliable || sb_done >= sb_cap) {
            if (c.score > best_score) {
                best_score = c.score; best = c.j;
                double f = frac(x[c.j]);
                est_down = z + pc(c.j, 0) * f; est_up = z + pc(c.j, 1) * (1 - f);
            }
            continue;
        }
        if (!saved) saved = S_.save_state();
        const int j = c.j;
        const double v = x[j], f = frac(v);
        const double lo = S_.col_lower(j), hi = S_.col_upper(j);
        double zz[2], cz[2] = {-kInf, -kInf};
        for (int dir = 0; dir < 2; ++dir) {
            BoundChange child = dir == 0 ? BoundChange{j, lo, std::floor(v)} : BoundChange{j, std::ceil(v), hi};
            S_.set_col_bounds(child.j, child.lo, child.hi);
            LpStatus st = S_.solve(opt_.sb_iter_limit, cutoff());
            if (opt_.safe_bounds && st != LpStatus::Infeasible) {
                // certify the child's bound from the strong-branching duals (any duals are valid)
                std::vector<BoundChange> ch = changes;
                ch.push_back(child);
                cz[dir] = safe_bound(ch);
            }
            if (st == LpStatus::Infeasible || st == LpStatus::Cutoff) zz[dir] = kInf;
            else if (st == LpStatus::Optimal) { zz[dir] = zmin(); pc_update(j, dir, zz[dir] - z, dir ? 1 - f : f); }
            else { double db = S_.dual_bound_min(); zz[dir] = std::isfinite(db) ? std::max(z, db) : z; }
            S_.restore_state(*saved);
        }
        ++sb_done;
        if (!std::isfinite(zz[0]) && !std::isfinite(zz[1])) return -2;
        if (!std::isfinite(zz[0]) || !std::isfinite(zz[1])) {
            // one side infeasible: tighten this node and re-solve
            BoundChange bc = !std::isfinite(zz[0]) ? BoundChange{j, std::ceil(v), hi} : BoundChange{j, lo, std::floor(v)};
            changes.push_back(bc);
            S_.set_col_bounds(bc.j, bc.lo, bc.hi);
            return -1;
        }
        double s = std::max(zz[0] - z, eps) * std::max(zz[1] - z, eps);
        if (full_sb) sb_record_.push_back({j, s});
        if (s > best_score) {
            best_score = s; best = j; est_down = zz[0]; est_up = zz[1];
            cert_down = cz[0]; cert_up = cz[1];
        }
    }
    return best;
}

MipResult Mip::run() {
    MipResult res;
    const double inf = kInf;
    if (opt_.verbose)
        std::printf("GANIT MILP | rows %d cols %d (%zu integer) nnz %lld\n", lp_.m, lp_.n, ints_.size(),
                    (long long)lp_.A.nnz());

    LpStatus st = S_.solve(-1, inf, opt_.time_limit);
    if (st == LpStatus::Infeasible) { res.status = MipStatus::Infeasible; res.objective = NAN; res.seconds = elapsed(); return res; }
    if (st == LpStatus::Unbounded) { res.status = MipStatus::Unbounded; res.objective = NAN; res.seconds = elapsed(); return res; }
    if (st != LpStatus::Optimal) { res.status = st == LpStatus::TimeLimit ? MipStatus::TimeLimit : MipStatus::Numerical; res.seconds = elapsed(); return res; }
    res.root_lp = sign_ * zmin();
    if (opt_.verbose) std::printf("Root LP %.10g (%ld simplex iterations, %.2fs)\n", res.root_lp, S_.iterations(), elapsed());

    try_incumbent(S_.primal(), "rounding");
    // ---- root cutting planes
    double zprev = zmin();
    int stall = 0;
    for (int round = 0; round < opt_.cut_rounds && !ints_.empty(); ++round) {
        if (fractional(S_.primal(), nullptr) == 0) break;
        int added = gomory_round();
        if (!added) break;
        cuts_ += added;
        st = solve_node({});
        if (st == LpStatus::Infeasible || st == LpStatus::Cutoff) break;
        if (st != LpStatus::Optimal) break;
        double z = zmin();
        if (opt_.verbose) std::printf("  cut round %d: +%d GMI cuts, bound %.10g\n", round + 1, added, sign_ * z);
        if (z - zprev < 1e-4 * std::max(1.0, std::fabs(z))) { if (++stall >= 2) break; } else stall = 0;
        zprev = z;
        try_incumbent(S_.primal(), "rounding");
    }
    if (st != LpStatus::Optimal) {
        // cuts proved the incumbent optimal (cutoff) or the problem infeasible
        if (st == LpStatus::Infeasible || st == LpStatus::Cutoff)
            res.status = std::isfinite(inc_) ? MipStatus::Optimal : MipStatus::Infeasible;
        else
            res.status = st == LpStatus::TimeLimit ? MipStatus::TimeLimit : MipStatus::Numerical;
        res.objective = std::isfinite(inc_) ? sign_ * inc_ : NAN;
        res.bound = res.objective;
        res.gap = 0;
        res.x = inc_x_;
        res.cuts = cuts_;
        res.lp_iterations = S_.iterations();
        res.seconds = elapsed();
        res.certified = opt_.safe_bounds && uncertified_prunes_ == 0;
        if (opt_.verbose)
            std::printf("Status %s | objective %.10g | proven at the root by cuts | %.2fs%s\n", to_string(res.status),
                        res.objective, res.seconds, res.certified ? " | CERTIFIED" : "");
        return res;
    }
    res.root_bound = sign_ * zmin();
    if (st == LpStatus::Optimal && !ints_.empty()) dive();

    // ---- branch and bound
    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    std::multiset<double> open_bounds;  // certified bounds of open nodes (global bound = min)
    auto push = [&](const Node& nd) { open.push(nd); open_bounds.insert(nd.bound); };
    auto pop = [&]() {
        Node nd = open.top();
        open.pop();
        auto it = open_bounds.find(nd.bound);
        if (it != open_bounds.end()) open_bounds.erase(it);
        return nd;
    };
    Node root;
    root.bound = zmin();
    root.depth = 0;
    root.basis = std::make_shared<DualSimplex::Basis>(S_.get_basis());
    bool have_dive = true;  // root state already loaded in the simplex
    Node cur = root;
    bool warm = true;       // current simplex state matches parent of `cur` (or cur itself for root)
    bool first = true;
    MipStatus stop = MipStatus::Optimal;
    double global_bound = root.bound;

    while (true) {
        if (!have_dive) {
            if (open.empty()) break;
            cur = pop();
            if (cur.bound >= cutoff()) { ++certified_prunes_; continue; }
            apply_bounds(cur.changes);
            S_.set_basis(*cur.basis);
            warm = false;
        }
        have_dive = false;
        if (elapsed() > opt_.time_limit) { stop = MipStatus::TimeLimit; push(cur); break; }
        if (nodes_ >= opt_.node_limit || (collect_fp_ && samples_ >= opt_.collect_max)) {
            stop = MipStatus::NodeLimit; push(cur); break;
        }
        ++nodes_;

        global_bound = open_bounds.empty() ? cur.bound : std::min(cur.bound, *open_bounds.begin());
        log(open.size(), global_bound);

        if (!first) {
            st = solve_node(cur.changes);
            if (st == LpStatus::Infeasible) ++infeasible_prunes_;
        } else {
            st = LpStatus::Optimal;  // root already solved
        }
        (void)warm;
        first = false;
        if (st == LpStatus::TimeLimit) { stop = MipStatus::TimeLimit; push(cur); break; }
        if (st != LpStatus::Optimal) continue;  // infeasible / cutoff / numerical -> prune

        // resolve loop for bound tightening from strong branching
        int j = -1;
        double z = 0, ed = 0, eu = 0, cd = -kInf, cu = -kInf;
        std::vector<double> x;
        bool pruned = false;
        for (size_t rep = 0; rep < ints_.size() + 2; ++rep) {
            z = zmin();
            if (prunable(z, cur.changes)) { pruned = true; break; }
            if (cur.branch_var >= 0 && rep == 0) pc_update(cur.branch_var, cur.branch_dir, z - cur.parent_obj, cur.branch_frac);
            x = S_.primal();
            if (fractional(x, nullptr) == 0) { try_incumbent(x, "LP solution"); pruned = true; break; }
            if (nodes_ % 20 == 1) try_incumbent(x, "rounding");
            cur_depth_ = cur.depth;
            j = select_branch(x, z, cur.changes, ed, eu, cd, cu);
            if (j >= 0 && collect_fp_) write_sample(x);
            if (j == -2) { pruned = true; break; }
            if (j == -1) {
                st = solve_node(cur.changes);
                if (st != LpStatus::Optimal) { pruned = true; break; }
                continue;
            }
            break;
        }
        if (pruned || j < 0) continue;

        const double v = x[j], f = frac(v);
        const double lo = S_.col_lower(j), hi = S_.col_upper(j);
        auto basis = std::make_shared<DualSimplex::Basis>(S_.get_basis());
        Node down, up;
        down.depth = up.depth = cur.depth + 1;
        down.changes = cur.changes; down.changes.push_back({j, lo, std::floor(v)});
        up.changes = cur.changes;   up.changes.push_back({j, std::ceil(v), hi});
        down.basis = up.basis = basis;
        down.parent_obj = up.parent_obj = z;
        down.branch_var = up.branch_var = j;
        down.branch_dir = 0; up.branch_dir = 1;
        down.branch_frac = f; up.branch_frac = 1 - f;
        if (opt_.safe_bounds) {
            // children inherit the certified bound of the parent (strong-branching
            // estimates are not certified, so they only guide the search order)
            double sb = safe_bound(cur.changes);
            if (!std::isfinite(sb)) sb = z;  // uncertifiable: fall back to the LP value
            down.bound = std::max(sb, cd);  // certified child bounds from strong branching, if any
            up.bound = std::max(sb, cu);
            down.est = std::max(z, ed);
            up.est = std::max(z, eu);
        } else {
            down.bound = std::max(z, ed);
            up.bound = std::max(z, eu);
        }
        // plunge into the more promising child, keep the other
        bool go_up = eu < ed || (eu == ed && f >= 0.5);
        Node& next = go_up ? up : down;
        Node& other = go_up ? down : up;
        if (other.bound < cutoff()) push(other);
        if (next.bound < cutoff()) {
            S_.set_col_bounds(j, next.changes.back().lo, next.changes.back().hi);
            cur = next;
            have_dive = true;
            warm = true;
        }
    }

    if (stop == MipStatus::Optimal) {
        res.status = std::isfinite(inc_) ? MipStatus::Optimal : MipStatus::Infeasible;
        global_bound = std::isfinite(inc_) ? inc_ : global_bound;
    } else {
        res.status = stop;
        global_bound = open_bounds.empty() ? global_bound : std::min(global_bound, *open_bounds.begin());
        if (std::isfinite(inc_)) global_bound = std::min(global_bound, inc_);
    }
    res.nodes = nodes_;
    res.cuts = cuts_;
    res.lp_iterations = S_.iterations();
    res.objective = std::isfinite(inc_) ? sign_ * inc_ : NAN;
    res.bound = sign_ * global_bound;
    res.gap = std::isfinite(inc_) ? std::fabs(inc_ - global_bound) / std::max(1.0, std::fabs(inc_)) : kInf;
    res.x = inc_x_;
    res.seconds = elapsed();
    res.certified = opt_.safe_bounds && uncertified_prunes_ == 0;
    res.certified_prunes = certified_prunes_;
    res.samples = samples_;
    res.branch_seconds = branch_time_;
    res.uncertifiable = uncertifiable_;
    log(open.size(), global_bound, true);
    if (opt_.verbose && opt_.safe_bounds)
        std::printf("Safe bounds: %ld bound prunes certified, %ld uncertified (fallback), %ld infeasibility prunes%s\n",
                    certified_prunes_, uncertified_prunes_, infeasible_prunes_,
                    res.certified ? "  => result CERTIFIED" : "");
    if (opt_.verbose)
        std::printf("Status %s | objective %.10g | bound %.10g | gap %.4f%% | nodes %ld | cuts %d | %.2fs\n",
                    to_string(res.status), res.objective, res.bound, 100 * res.gap, res.nodes, res.cuts, res.seconds);
    return res;
}

}  // namespace

MipResult solve_mip(const LP& lp, const MipOptions& opt) {
    Mip mip(lp, opt);
    return mip.run();
}

}  // namespace ganit
