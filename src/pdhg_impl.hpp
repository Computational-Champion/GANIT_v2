// Restarted-average PDHG for LP and convex QP, written once and instantiated
// for each compute backend (CPU / CUDA) and each precision (float / double).
//
// Iteration (scaled space, y = row duals, reduced cost c + Qx - A^T y):
//   x+   = proj_[l,u]( x - tau (c + Qx - A^T y) )
//   xbar = 2 x+ - x
//   q    = y - sigma A xbar
//   y+   = q + clamp(-q, sigma*lo, sigma*hi)      (prox of the row-bound set)
// with sigma = eta*w, tau = eta/w (LP) or min(eta/w, 1/(||Q||/2 + sigma||A||^2)) (QP).
//
// Restarts: adaptive scheme of PDLP (Applegate et al. 2021).
//
// DYNAMIC PRECISION (Options::precision):
//   L1  matrix FP32, vectors FP32  (least memory traffic; PDHG is bandwidth-bound)
//   L2  matrix FP32, vectors FP64  (keeps the matrix saving, removes the iterate
//                                   stagnation floor of FP32 vectors)
//   L3  matrix FP64, vectors FP64
//   Mixed   = L1 -> L3,   Dynamic = L1 -> L2 -> L3,   Double = L3 only.
//   A level is left when the target tolerance is met or progress stalls at that
//   level's accuracy floor; the iterate, primal weight and restart point carry
//   over. Reductions always accumulate in FP64, and the returned solution is
//   ALWAYS re-evaluated and certified in FP64.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "ganit/pdhg.hpp"
#include "scaling.hpp"

namespace ganit {

struct KKT {
    double pobj = 0, dobj = 0, rel_p = 0, rel_d = 0, rel_gap = 0, err = 0;
    bool finite() const {
        return std::isfinite(pobj) && std::isfinite(dobj) && std::isfinite(rel_p) && std::isfinite(rel_d);
    }
    bool converged(double tol) const { return rel_p <= tol && rel_d <= tol && rel_gap <= tol; }
};

struct PdhgShared {
    const ScaledLP& s;
    const Options& opt;
    double eta, normA, obj_scale;
    std::chrono::steady_clock::time_point t_solve;
    double last_log = -1e9;
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve).count();
    }
};

struct PdhgState {
    std::vector<double> x, y;  // host copies (scaled space)
    std::vector<double> xr, yr;  // last restart point (carried across the precision switch)
    double restart_err = -1.0;
    double w = 1.0;
    long it = 0;
    int restarts = 0;
    KKT kkt;
};

enum class PhaseExit { Converged, Stalled, IterationLimit, TimeLimit, Infeasible, Numerical };

// Runs PDHG with Backend B from st.x/st.y until `phase_tol` is met (or other exit).
template <class B>
PhaseExit pdhg_phase(PdhgShared& sh, PdhgState& st, double phase_tol, bool stall_exit, const char* tag,
                     int stall_checks) {
    const ScaledLP& s = sh.s;
    const Options& opt = sh.opt;
    B be(s);
    using Vec = typename B::Vec;
    const int m = s.m, n = s.n;
    const bool hasq = s.has_q();

    Vec x = be.from_host(st.x), xn = be.zeros(n), xbar = be.zeros(n);
    Vec y = be.from_host(st.y), yn = be.zeros(m);
    Vec aty = be.zeros(n), axbar = be.zeros(m);
    Vec sumx = be.zeros(n), sumy = be.zeros(m), xa = be.zeros(n), ya = be.zeros(m);
    Vec xr = be.from_host(st.xr.empty() ? st.x : st.xr), yr = be.from_host(st.yr.empty() ? st.y : st.yr);
    Vec ax_tmp = be.zeros(m), aty_tmp = be.zeros(n), ray = be.zeros(m);
    Vec qx = be.zeros(hasq ? n : 0), qx_tmp = be.zeros(hasq ? n : 0);
    double w = st.w;

    auto step_sizes = [&](double& tau, double& sigma) {
        sigma = sh.eta * w;
        tau = sh.eta / w;
        if (hasq) tau = std::min(tau, 0.998 / (0.5 * s.normQ + sigma * sh.normA * sh.normA));
    };
    auto eval = [&](const Vec& xv, const Vec& yv) {
        KKT k;
        be.spmv_A(xv, ax_tmp);
        double rp2 = be.primal_res2(ax_tmp);
        be.spmv_AT(yv, aty_tmp);
        double xqx = 0.0;
        if (hasq) { be.spmv_Q(xv, qx_tmp); xqx = be.dot(xv, qx_tmp); }
        double rd2 = 0, vobj = 0;
        be.dual_res(aty_tmp, hasq ? &qx_tmp : nullptr, rd2, vobj);
        double robj = be.row_dual_obj(yv);
        k.pobj = (be.dot_c(xv) + 0.5 * xqx) * sh.obj_scale + s.obj_const;
        k.dobj = (robj + vobj - 0.5 * xqx) * sh.obj_scale + s.obj_const;
        k.rel_p = std::sqrt(rp2) / (1.0 + s.bnorm);
        k.rel_d = std::sqrt(rd2) / (1.0 + s.cnorm);
        k.rel_gap = std::fabs(k.pobj - k.dobj) / (1.0 + std::fabs(k.pobj) + std::fabs(k.dobj));
        k.err = std::sqrt(k.rel_p * k.rel_p + k.rel_d * k.rel_d + k.rel_gap * k.rel_gap);
        return k;
    };
    auto primal_infeasible = [&]() {
        be.sub(y, yr, ray);
        be.spmv_AT(ray, aty_tmp);
        double rd2 = 0, vobj = 0;
        be.dual_res(aty_tmp, nullptr, rd2, vobj, false);
        double dray = (be.row_dual_obj(ray) + vobj) * sh.obj_scale;
        if (!(dray > 0)) return false;
        return std::sqrt(rd2) * (1.0 + s.bnorm) / dray <= opt.infeas_tol;
    };

    KKT k_restart = eval(x, y);
    if (st.restart_err > 0) k_restart.err = std::max(k_restart.err, st.restart_err);
    double k_prev_cand = std::numeric_limits<double>::infinity();
    long it_since = 0;
    double best_err = k_restart.err;
    int checks_without_progress = 0;
    PhaseExit exit_reason = PhaseExit::IterationLimit;
    KKT best;
    bool best_is_avg = false;

    while (true) {
        double tau, sigma;
        step_sizes(tau, sigma);
        be.spmv_AT(y, aty);
        if (hasq) be.spmv_Q(x, qx);
        be.primal_step(x, aty, hasq ? &qx : nullptr, tau, xn, xbar, sumx);
        be.spmv_A(xbar, axbar);
        be.dual_step(y, axbar, sigma, yn, sumy);
        std::swap(x, xn);
        std::swap(y, yn);
        ++st.it;
        ++it_since;
        if (st.it % opt.check_every != 0) continue;

        KKT kc = eval(x, y);
        be.scale_into(sumx, 1.0 / static_cast<double>(it_since), xa);
        be.scale_into(sumy, 1.0 / static_cast<double>(it_since), ya);
        KKT ka = eval(xa, ya);
        const double t = sh.elapsed();

        if (!kc.finite() && !ka.finite()) { exit_reason = PhaseExit::Numerical; best = kc; break; }
        const bool avg_better = ka.finite() && (!kc.finite() || ka.err < kc.err);
        KKT kcand = avg_better ? ka : kc;

        if (opt.verbose && (t - sh.last_log >= opt.log_interval)) {
            std::printf("%9ld %8.2fs %4s %15.8e %15.8e %9.2e %9.2e %9.2e %9.2e\n", st.it, t, tag, kcand.pobj,
                        kcand.dobj, kcand.rel_p, kcand.rel_d, kcand.rel_gap, w);
            sh.last_log = t;
        }
        if (kc.converged(phase_tol)) { exit_reason = PhaseExit::Converged; best = kc; best_is_avg = false; break; }
        if (ka.converged(phase_tol)) { exit_reason = PhaseExit::Converged; best = ka; best_is_avg = true; break; }
        if (st.it >= opt.max_iter) { exit_reason = PhaseExit::IterationLimit; best = kcand; best_is_avg = avg_better; break; }
        if (t >= opt.time_limit) { exit_reason = PhaseExit::TimeLimit; best = kcand; best_is_avg = avg_better; break; }
        if (it_since >= 4 * opt.check_every && primal_infeasible()) {
            exit_reason = PhaseExit::Infeasible; best = kcand; best_is_avg = avg_better; break;
        }
        // stall detection (used to leave the FP32 phase at its accuracy floor)
        if (kcand.err < 0.8 * best_err) { best_err = kcand.err; checks_without_progress = 0; }
        else if (++checks_without_progress >= stall_checks && stall_exit) {
            exit_reason = PhaseExit::Stalled; best = kcand; best_is_avg = avg_better; break;
        }

        // ---- adaptive restart
        const bool sufficient = kcand.err <= 0.2 * k_restart.err;
        const bool necessary = kcand.err <= 0.8 * k_restart.err && kcand.err > k_prev_cand;
        const bool artificial = it_since >= 0.36 * static_cast<double>(st.it);
        k_prev_cand = kcand.err;
        if (sufficient || necessary || artificial) {
            if (avg_better) { be.copy(xa, x); be.copy(ya, y); }
            double dx = std::sqrt(be.diff2(x, xr)), dy = std::sqrt(be.diff2(y, yr));
            if (dx > 1e-10 && dy > 1e-10) w = std::exp(0.5 * std::log(dy / dx) + 0.5 * std::log(w));
            be.copy(x, xr);
            be.copy(y, yr);
            be.fill(sumx, 0.0);
            be.fill(sumy, 0.0);
            it_since = 0;
            k_restart = kcand;
            k_prev_cand = std::numeric_limits<double>::infinity();
            ++st.restarts;
        }
    }
    st.x = be.to_host(best_is_avg ? xa : x);
    st.y = be.to_host(best_is_avg ? ya : y);
    st.xr = be.to_host(xr);
    st.yr = be.to_host(yr);
    st.restart_err = k_restart.err;
    st.w = w;
    st.kkt = best;
    return exit_reason;
}

// Evaluate KKT of a host iterate with backend B (used for the FP64 certification).
template <class B>
KKT pdhg_evaluate(PdhgShared& sh, const PdhgState& st) {
    PdhgState tmp = st;
    Options o = sh.opt;
    // zero-iteration phase: build backend, evaluate, return
    B be(sh.s);
    using Vec = typename B::Vec;
    const ScaledLP& s = sh.s;
    const bool hasq = s.has_q();
    Vec x = be.from_host(st.x), y = be.from_host(st.y);
    Vec ax = be.zeros(s.m), aty = be.zeros(s.n), qx = be.zeros(hasq ? s.n : 0);
    KKT k;
    be.spmv_A(x, ax);
    double rp2 = be.primal_res2(ax);
    be.spmv_AT(y, aty);
    double xqx = 0.0;
    if (hasq) { be.spmv_Q(x, qx); xqx = be.dot(x, qx); }
    double rd2 = 0, vobj = 0;
    be.dual_res(aty, hasq ? &qx : nullptr, rd2, vobj);
    double robj = be.row_dual_obj(y);
    k.pobj = (be.dot_c(x) + 0.5 * xqx) * sh.obj_scale + s.obj_const;
    k.dobj = (robj + vobj - 0.5 * xqx) * sh.obj_scale + s.obj_const;
    k.rel_p = std::sqrt(rp2) / (1.0 + s.bnorm);
    k.rel_d = std::sqrt(rd2) / (1.0 + s.cnorm);
    k.rel_gap = std::fabs(k.pobj - k.dobj) / (1.0 + std::fabs(k.pobj) + std::fabs(k.dobj));
    k.err = std::sqrt(k.rel_p * k.rel_p + k.rel_d * k.rel_d + k.rel_gap * k.rel_gap);
    (void)tmp; (void)o;
    return k;
}

inline Status status_from(PhaseExit e) {
    switch (e) {
        case PhaseExit::Converged: return Status::Optimal;
        case PhaseExit::Infeasible: return Status::PrimalInfeasible;
        case PhaseExit::TimeLimit: return Status::TimeLimit;
        case PhaseExit::Numerical: return Status::NumericalError;
        default: return Status::IterationLimit;
    }
}

// B1/B2/B3 = backends for precision levels L1/L2/L3.
template <class B1, class B2, class Bd>
Result run_pdhg(const LP& lp, const Options& opt, const char* device_name) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    Result res;
    res.device = device_name;

    ScaledLP s = scale_lp(lp, opt.ruiz_iters, opt.pock_chambolle);
    const double normA = estimate_norm(s);
    PdhgShared sh{s, opt, 0.998 / normA, normA, s.cscale * s.bscale, Clock::now()};

    PdhgState st;
    st.x.resize(s.n);
    for (int j = 0; j < s.n; ++j) st.x[j] = std::min(std::max(0.0, s.l[j]), s.u[j]);
    st.y.assign(s.m, 0.0);
    {
        double cn = norm2(s.c), bn = finite_bound_norm(s.lo, s.hi);
        st.w = (cn > 1e-10 && bn > 1e-10) ? cn / bn : 1.0;
    }
    const bool mixed = opt.precision != Precision::Double;
    const char* pname = opt.precision == Precision::Dynamic ? "dynamic L1->L2->L3"
                      : opt.precision == Precision::Mixed ? "mixed L1->L3" : "FP64";
    if (opt.verbose) {
        std::printf("GANIT PDHG | device %s | %s | precision %s | rows %d cols %d nnz %lld | ||A||~%.3e | tol %.1e\n",
                    device_name, s.has_q() ? "QP" : "LP", pname, s.m, s.n,
                    (long long)s.A.nnz(), normA, opt.tol);
        if (s.has_q()) std::printf("Hessian nnz %lld | ||Q||~%.3e\n", (long long)s.Q.nnz(), s.normQ);
        std::printf("%9s %9s %4s %15s %15s %9s %9s %9s %9s\n", "iter", "time", "prec", "primal obj", "dual obj",
                    "rel_p", "rel_d", "rel_gap", "weight");
    }
    res.setup_seconds = std::chrono::duration<double>(Clock::now() - t_start).count();
    sh.t_solve = Clock::now();

    PhaseExit ex;
    long fp32_iters = 0;
    auto finished = [&](PhaseExit e, const char* level, const char* next) {
        KKT k64 = pdhg_evaluate<Bd>(sh, st);
        bool done = (e == PhaseExit::Converged && k64.converged(opt.tol)) || e == PhaseExit::TimeLimit ||
                    e == PhaseExit::IterationLimit || e == PhaseExit::Infeasible;
        if (opt.verbose)
            std::printf("  %s phase: %ld iters total, %.3fs, FP64-checked error %.2e -> %s\n", level, st.it,
                        sh.elapsed(), k64.err, done ? "done" : next);
        return done;
    };
    if (mixed) {
        ex = pdhg_phase<B1>(sh, st, opt.tol, true, "L1", opt.stall_checks);
        fp32_iters = st.it;
        bool done = finished(ex, "L1 (FP32)", opt.precision == Precision::Dynamic ? "switching to L2" : "switching to L3");
        if (!done && opt.precision == Precision::Dynamic) {
            ex = pdhg_phase<B2>(sh, st, opt.tol, true, "L2", opt.stall_checks_l2);
            res.l2_iterations = st.it - fp32_iters;
            done = finished(ex, "L2 (FP32 matrix, FP64 vectors)", "switching to L3");
        }
        if (!done) ex = pdhg_phase<Bd>(sh, st, opt.tol, false, "L3", 0);
    } else {
        ex = pdhg_phase<Bd>(sh, st, opt.tol, false, "L3", 0);
    }
    // Final FP64 certification of the returned point.
    KKT fin = pdhg_evaluate<Bd>(sh, st);
    Status status = status_from(ex);
    if (status == Status::Optimal && !fin.converged(opt.tol * 1.0000001)) status = Status::NumericalError;

    res.solve_seconds = sh.elapsed();
    res.status = status;
    res.iterations = st.it;
    res.fp32_iterations = fp32_iters;
    res.restarts = st.restarts;
    const double sign = s.maximize ? -1.0 : 1.0;
    res.x.resize(s.n);
    res.y.resize(s.m);
    for (int j = 0; j < s.n; ++j) res.x[j] = st.x[j] * s.C[j] * s.bscale;
    for (int i = 0; i < s.m; ++i) res.y[i] = sign * st.y[i] * s.R[i] * s.cscale;
    res.pobj = sign * fin.pobj;
    res.dobj = sign * fin.dobj;
    res.rel_primal = fin.rel_p;
    res.rel_dual = fin.rel_d;
    res.rel_gap = fin.rel_gap;
    if (opt.verbose) {
        std::printf("Status %s | iters %ld (L1 %ld, L2 %ld) | restarts %d | solve %.3fs (setup %.3fs)\n",
                    to_string(status), st.it, fp32_iters, res.l2_iterations, res.restarts, res.solve_seconds,
                    res.setup_seconds);
        std::printf("Primal obj %.10e | Dual obj %.10e | FP64-certified rel errors p %.1e d %.1e gap %.1e\n",
                    res.pobj, res.dobj, fin.rel_p, fin.rel_d, fin.rel_gap);
    }
    return res;
}

}  // namespace ganit
