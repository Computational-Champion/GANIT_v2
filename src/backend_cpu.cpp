// CPU backend: reference implementation of the fused PDHG kernels, templated
// on the matrix-value precision M and the vector precision R (dynamic precision
// levels: L1 <float,float>, L2 <float,double>, L3 <double,double>).
// Reductions always accumulate in double. Mirrors backend_cuda.cu one-to-one so results can be cross-checked.
#include <algorithm>
#include <cmath>
#include <vector>

#include "pdhg_impl.hpp"

namespace ganit {

const char* to_string(Status s) {
    switch (s) {
        case Status::Optimal: return "OPTIMAL";
        case Status::PrimalInfeasible: return "PRIMAL_INFEASIBLE";
        case Status::IterationLimit: return "ITERATION_LIMIT";
        case Status::TimeLimit: return "TIME_LIMIT";
        default: return "NUMERICAL_ERROR";
    }
}

namespace {

template <class M>
struct CsrR {
    int rows = 0;
    const int* ptr = nullptr;
    const int* idx = nullptr;
    std::vector<M> val;
    CsrR() = default;
    explicit CsrR(const Csr& a) : rows(a.rows), ptr(a.ptr.data()), idx(a.idx.data()), val(a.val.begin(), a.val.end()) {}
};

template <class R>
std::vector<R> conv(const std::vector<double>& v) { return std::vector<R>(v.begin(), v.end()); }

template <class M, class R>
struct CpuBackendT {
    using Vec = std::vector<R>;
    const ScaledLP& s;
    CsrR<M> A, AT, Q;
    Vec c, l, u, lo, hi, Rres, Cres;

    explicit CpuBackendT(const ScaledLP& lp)
        : s(lp), A(lp.A), AT(lp.AT), Q(lp.Q), c(conv<R>(lp.c)), l(conv<R>(lp.l)), u(conv<R>(lp.u)),
          lo(conv<R>(lp.lo)), hi(conv<R>(lp.hi)), Rres(conv<R>(lp.Rres)), Cres(conv<R>(lp.Cres)) {}

    Vec zeros(int n) { return Vec(n, R(0)); }
    Vec from_host(const std::vector<double>& h) { return conv<R>(h); }
    std::vector<double> to_host(const Vec& v) { return std::vector<double>(v.begin(), v.end()); }
    void copy(const Vec& src, Vec& dst) { dst = src; }
    void fill(Vec& v, double a) { std::fill(v.begin(), v.end(), R(a)); }

    static void spmv(const CsrR<M>& a, const Vec& x, Vec& y) {
#pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < a.rows; ++i) {
            R t = 0;
            for (int p = a.ptr[i]; p < a.ptr[i + 1]; ++p) t += R(a.val[p]) * x[a.idx[p]];
            y[i] = t;
        }
    }
    void spmv_A(const Vec& x, Vec& ax) { spmv(A, x, ax); }
    void spmv_AT(const Vec& y, Vec& aty) { spmv(AT, y, aty); }
    void spmv_Q(const Vec& x, Vec& qx) { spmv(Q, x, qx); }

    void primal_step(const Vec& x, const Vec& aty, const Vec* qx, double tau_d, Vec& xn, Vec& xbar, Vec& sumx) {
        const R tau = R(tau_d);
        const R* q = qx ? qx->data() : nullptr;
        const int n = s.n;
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            R g = c[j] - aty[j] + (q ? q[j] : R(0));
            R v = x[j] - tau * g;
            v = std::min(std::max(v, l[j]), u[j]);
            xn[j] = v;
            xbar[j] = R(2) * v - x[j];
            sumx[j] += v;
        }
    }
    void dual_step(const Vec& y, const Vec& axbar, double sigma_d, Vec& yn, Vec& sumy) {
        const R sigma = R(sigma_d);
        const int m = s.m;
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            R q = y[i] - sigma * axbar[i];
            R t = std::min(std::max(-q, sigma * lo[i]), sigma * hi[i]);
            R v = q + t;
            yn[i] = v;
            sumy[i] += v;
        }
    }
    void sub(const Vec& a, const Vec& b, Vec& d) {
#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    }
    void scale_into(const Vec& src, double a, Vec& dst) {
#pragma omp parallel for
        for (size_t i = 0; i < src.size(); ++i) dst[i] = R(a * double(src[i]));
    }
    double primal_res2(const Vec& ax) {
        double acc = 0.0;
        const int m = s.m;
#pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < m; ++i) {
            double a = ax[i];
            double p = std::min(std::max(a, double(lo[i])), double(hi[i]));
            double r = (a - p) / double(Rres[i]);
            acc += r * r;
        }
        return acc;
    }
    void dual_res(const Vec& aty, const Vec* qx, double& rd2, double& vobj, bool with_c = true) {
        double a2 = 0.0, ob = 0.0;
        const double cw = with_c ? 1.0 : 0.0;
        const R* q = qx ? qx->data() : nullptr;
        const int n = s.n;
#pragma omp parallel for reduction(+ : a2, ob)
        for (int j = 0; j < n; ++j) {
            double z = cw * double(c[j]) - double(aty[j]) + (q ? double(q[j]) : 0.0);
            double lj = l[j], uj = u[j];
            bool lf = std::isfinite(lj), uf = std::isfinite(uj);
            double r = 0.0;
            if (z > 0) { if (lf) ob += z * lj; else r = z; }
            else if (z < 0) { if (uf) ob += z * uj; else r = z; }
            r /= double(Cres[j]);
            a2 += r * r;
        }
        rd2 = a2;
        vobj = ob;
    }
    double row_dual_obj(const Vec& y) {
        double acc = 0.0;
        const int m = s.m;
#pragma omp parallel for reduction(+ : acc)
        for (int i = 0; i < m; ++i) {
            double yi = y[i];
            if (yi > 0 && std::isfinite(double(lo[i]))) acc += yi * double(lo[i]);
            else if (yi < 0 && std::isfinite(double(hi[i]))) acc += yi * double(hi[i]);
        }
        return acc;
    }
    double dot(const Vec& a, const Vec& b) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (size_t i = 0; i < a.size(); ++i) acc += double(a[i]) * double(b[i]);
        return acc;
    }
    double dot_c(const Vec& x) { return dot(c, x); }
    double diff2(const Vec& a, const Vec& b) {
        double acc = 0.0;
#pragma omp parallel for reduction(+ : acc)
        for (size_t i = 0; i < a.size(); ++i) { double d = double(a[i]) - double(b[i]); acc += d * d; }
        return acc;
    }
};

}  // namespace

Result solve_pdhg_cpu(const LP& lp, const Options& opt) {
    return run_pdhg<CpuBackendT<float, float>, CpuBackendT<float, double>, CpuBackendT<double, double>>(lp, opt, "cpu");
}

}  // namespace ganit
