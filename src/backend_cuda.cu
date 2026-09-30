// CUDA backend for GANIT PDHG (LP and convex QP).
// All kernels written from scratch (no cuSPARSE / cuBLAS / solver libraries):
//   * CSR SpMV: scalar (thread-per-row) or vector (warp-per-row) chosen by
//     average row length
//   * fused primal step: gradient + projection + extrapolation + running average
//   * fused dual step:   prox of row bounds + running average
//   * block reductions with warp shuffles, always accumulated in FP64
// Templated on the matrix-value type M and vector type V, giving the dynamic
// precision levels L1 <float,float>, L2 <float,double>, L3 <double,double>.
// Requires compute capability >= 6.0 (double atomicAdd).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pdhg_impl.hpp"

#define GANIT_CUDA_CHECK(call)                                                          \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess)                                                       \
            throw std::runtime_error(std::string("CUDA error: ") +                      \
                                     cudaGetErrorString(err__) + " at " __FILE__ ":" +  \
                                     std::to_string(__LINE__));                         \
    } while (0)

namespace ganit {
namespace {

constexpr int kBlock = 256;

inline int grid_for(long long n, int block = kBlock) {
    long long g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > 2147483647LL) g = 2147483647LL;
    return static_cast<int>(g);
}

// ---------------------------------------------------------------- device memory
template <class T>
class DevBuf {
   public:
    DevBuf() = default;
    explicit DevBuf(size_t n) : n_(n) {
        if (n_) GANIT_CUDA_CHECK(cudaMalloc(&p_, n_ * sizeof(T)));
    }
    ~DevBuf() { if (p_) cudaFree(p_); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        std::swap(p_, o.p_);
        std::swap(n_, o.n_);
        return *this;
    }
    T* get() const { return p_; }
    size_t size() const { return n_; }
    void upload(const std::vector<T>& h) {
        if (n_) GANIT_CUDA_CHECK(cudaMemcpy(p_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(n_);
        if (n_) GANIT_CUDA_CHECK(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }

   private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// ---------------------------------------------------------------- kernels
template <class M, class V>
__global__ void k_spmv_scalar(int rows, const int* __restrict__ ptr, const int* __restrict__ idx,
                              const M* __restrict__ val, const V* __restrict__ x, V* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    V t = V(0);
    for (int p = ptr[i]; p < ptr[i + 1]; ++p) t += V(val[p]) * __ldg(&x[idx[p]]);
    y[i] = t;
}

// One warp per row; blockDim.x must be a multiple of 32 so whole warps exit together.
template <class M, class V>
__global__ void k_spmv_warp(int rows, const int* __restrict__ ptr, const int* __restrict__ idx,
                            const M* __restrict__ val, const V* __restrict__ x, V* __restrict__ y) {
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int row = static_cast<int>(gid >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    V t = V(0);
    for (int p = ptr[row] + lane; p < ptr[row + 1]; p += 32) t += V(val[p]) * __ldg(&x[idx[p]]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (lane == 0) y[row] = t;
}

template <class V>
__global__ void k_primal_step(int n, const V* __restrict__ x, const V* __restrict__ aty,
                              const V* __restrict__ qx, const V* __restrict__ c, const V* __restrict__ l,
                              const V* __restrict__ u, V tau, V* __restrict__ xn, V* __restrict__ xbar,
                              V* __restrict__ sumx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    V xj = x[j];
    V g = c[j] - aty[j] + (qx ? qx[j] : V(0));
    V v = xj - tau * g;
    v = v < l[j] ? l[j] : v;
    v = v > u[j] ? u[j] : v;
    xn[j] = v;
    xbar[j] = V(2) * v - xj;
    sumx[j] += v;
}

template <class V>
__global__ void k_dual_step(int m, const V* __restrict__ y, const V* __restrict__ axbar,
                            const V* __restrict__ lo, const V* __restrict__ hi, V sigma,
                            V* __restrict__ yn, V* __restrict__ sumy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    V q = y[i] - sigma * axbar[i];
    V t = -q;
    V a = sigma * lo[i], b = sigma * hi[i];
    t = t < a ? a : t;
    t = t > b ? b : t;
    V v = q + t;
    yn[i] = v;
    sumy[i] += v;
}

template <class V>
__global__ void k_scale(int n, double a, const V* __restrict__ s, V* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = V(a * double(s[i]));
}

template <class V>
__global__ void k_sub(int n, const V* __restrict__ a, const V* __restrict__ b, V* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] - b[i];
}

template <class V>
__global__ void k_fill(int n, double a, V* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = V(a);
}

// host -> device conversion of a double array into precision T
template <class T>
__global__ void k_convert(int n, const double* __restrict__ src, T* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = T(src[i]);
}
template <class T>
__global__ void k_to_double(int n, const T* __restrict__ src, double* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = double(src[i]);
}

// ---- reductions: functor gives per-index contribution(s), block reduce, atomicAdd.
__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

template <class F>
__global__ void k_reduce2(int n, F f, double* out) {
    __shared__ double sh0[kBlock / 32], sh1[kBlock / 32];
    double a0 = 0.0, a1 = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double v0, v1;
        f(i, v0, v1);
        a0 += v0;
        a1 += v1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sh0[wid] = a0; sh1[wid] = a1; }
    __syncthreads();
    if (wid == 0) {
        a0 = (lane < kBlock / 32) ? sh0[lane] : 0.0;
        a1 = (lane < kBlock / 32) ? sh1[lane] : 0.0;
        a0 = warp_sum(a0);
        a1 = warp_sum(a1);
        if (lane == 0) {
            atomicAdd(&out[0], a0);
            atomicAdd(&out[1], a1);
        }
    }
}

template <class V>
struct PrimalResF {
    const V *ax, *lo, *hi, *R;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double a = double(ax[i]);
        double p = fmin(fmax(a, double(lo[i])), double(hi[i]));
        double r = (a - p) / double(R[i]);
        v0 = r * r;
        v1 = 0.0;
    }
};
template <class V>
struct DualResF {
    const V *aty, *qx, *c, *l, *u, *C;
    double cw;  // 1 = normal dual, 0 = homogeneous dual (infeasibility ray)
    __device__ void operator()(int j, double& v0, double& v1) const {
        double z = cw * double(c[j]) - double(aty[j]) + (qx ? double(qx[j]) : 0.0);
        double lj = double(l[j]), uj = double(u[j]);
        double r = 0.0, ob = 0.0;
        if (z > 0) { if (isfinite(lj)) ob = z * lj; else r = z; }
        else if (z < 0) { if (isfinite(uj)) ob = z * uj; else r = z; }
        r /= double(C[j]);
        v0 = r * r;
        v1 = ob;
    }
};
template <class V>
struct RowDualObjF {
    const V *y, *lo, *hi;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double yi = double(y[i]), l = double(lo[i]), h = double(hi[i]);
        v0 = 0.0;
        if (yi > 0 && isfinite(l)) v0 = yi * l;
        else if (yi < 0 && isfinite(h)) v0 = yi * h;
        v1 = 0.0;
    }
};
template <class V>
struct DotF {
    const V *a, *b;
    __device__ void operator()(int i, double& v0, double& v1) const { v0 = double(a[i]) * double(b[i]); v1 = 0.0; }
};
template <class V>
struct Diff2F {
    const V *a, *b;
    __device__ void operator()(int i, double& v0, double& v1) const {
        double d = double(a[i]) - double(b[i]);
        v0 = d * d;
        v1 = 0.0;
    }
};

// ---------------------------------------------------------------- backend
template <class M>
struct DevCsr {
    int rows = 0;
    bool warp = false;
    DevBuf<int> ptr, idx;
    DevBuf<M> val;
};

// Upload a double host array into device precision T (conversion on the device).
template <class T>
DevBuf<T> upload_as(const std::vector<double>& h) {
    DevBuf<T> d(h.size());
    if (h.empty()) return d;
    DevBuf<double> tmp(h.size());
    tmp.upload(h);
    k_convert<T><<<grid_for(h.size()), kBlock>>>(static_cast<int>(h.size()), tmp.get(), d.get());
    GANIT_CUDA_CHECK(cudaDeviceSynchronize());
    return d;
}

template <class M>
DevCsr<M> to_device(const Csr& a) {
    DevCsr<M> d;
    d.rows = a.rows;
    d.ptr = DevBuf<int>(a.ptr.size());
    d.ptr.upload(a.ptr);
    d.idx = DevBuf<int>(a.idx.size());
    d.idx.upload(a.idx);
    d.val = upload_as<M>(a.val);
    double avg = a.rows ? static_cast<double>(a.val.size()) / a.rows : 0.0;
    d.warp = avg > 12.0;  // long rows -> warp-per-row kernel
    return d;
}

template <class M, class V>
struct CudaBackend {
    using Vec = DevBuf<V>;
    int m, n, reduce_grid;
    DevCsr<M> A, AT, Q;
    Vec c, l, u, lo, hi, R, C;
    DevBuf<double> scratch;

    explicit CudaBackend(const ScaledLP& s)
        : m(s.m), n(s.n), A(to_device<M>(s.A)), AT(to_device<M>(s.AT)), Q(to_device<M>(s.Q)), scratch(2) {
        c = upload_as<V>(s.c); l = upload_as<V>(s.l); u = upload_as<V>(s.u);
        lo = upload_as<V>(s.lo); hi = upload_as<V>(s.hi);
        R = upload_as<V>(s.Rres); C = upload_as<V>(s.Cres);  // residual unscaling
        int dev = 0, sms = 0;
        GANIT_CUDA_CHECK(cudaGetDevice(&dev));
        GANIT_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        reduce_grid = sms * 8;
    }

    Vec zeros(int k) { Vec v(k); fill(v, 0.0); return v; }
    Vec from_host(const std::vector<double>& h) { return upload_as<V>(h); }
    std::vector<double> to_host(const Vec& v) {
        std::vector<double> h(v.size());
        if (v.size()) {
            DevBuf<double> tmp(v.size());
            k_to_double<V><<<grid_for(v.size()), kBlock>>>(static_cast<int>(v.size()), v.get(), tmp.get());
            h = tmp.download();
        }
        return h;
    }
    void copy(const Vec& src, Vec& dst) {
        if (src.size())
            GANIT_CUDA_CHECK(cudaMemcpy(dst.get(), src.get(), src.size() * sizeof(V), cudaMemcpyDeviceToDevice));
    }
    void fill(Vec& v, double a) {
        if (v.size()) k_fill<V><<<grid_for(v.size()), kBlock>>>((int)v.size(), a, v.get());
    }

    static void spmv(const DevCsr<M>& a, const Vec& x, Vec& y) {
        if (a.rows == 0) return;
        if (a.warp)
            k_spmv_warp<M, V><<<grid_for(static_cast<long long>(a.rows) * 32), kBlock>>>(
                a.rows, a.ptr.get(), a.idx.get(), a.val.get(), x.get(), y.get());
        else
            k_spmv_scalar<M, V><<<grid_for(a.rows), kBlock>>>(a.rows, a.ptr.get(), a.idx.get(), a.val.get(),
                                                             x.get(), y.get());
    }
    void spmv_A(const Vec& x, Vec& ax) { spmv(A, x, ax); }
    void spmv_AT(const Vec& y, Vec& aty) { spmv(AT, y, aty); }
    void spmv_Q(const Vec& x, Vec& qx) { spmv(Q, x, qx); }

    void primal_step(const Vec& x, const Vec& aty, const Vec* qx, double tau, Vec& xn, Vec& xbar, Vec& sumx) {
        if (n) k_primal_step<V><<<grid_for(n), kBlock>>>(n, x.get(), aty.get(), qx ? qx->get() : nullptr, c.get(),
                                                         l.get(), u.get(), V(tau), xn.get(), xbar.get(), sumx.get());
    }
    void dual_step(const Vec& y, const Vec& axbar, double sigma, Vec& yn, Vec& sumy) {
        if (m) k_dual_step<V><<<grid_for(m), kBlock>>>(m, y.get(), axbar.get(), lo.get(), hi.get(), V(sigma),
                                                       yn.get(), sumy.get());
    }
    void sub(const Vec& a, const Vec& b, Vec& d) {
        if (a.size()) k_sub<V><<<grid_for(a.size()), kBlock>>>((int)a.size(), a.get(), b.get(), d.get());
    }
    void scale_into(const Vec& src, double a, Vec& dst) {
        if (src.size()) k_scale<V><<<grid_for(src.size()), kBlock>>>((int)src.size(), a, src.get(), dst.get());
    }

    template <class F>
    std::pair<double, double> reduce(int k, F f) {
        GANIT_CUDA_CHECK(cudaMemset(scratch.get(), 0, 2 * sizeof(double)));
        if (k > 0) {
            int g = std::min(grid_for(k), reduce_grid);
            k_reduce2<F><<<g, kBlock>>>(k, f, scratch.get());
        }
        double h[2];
        GANIT_CUDA_CHECK(cudaMemcpy(h, scratch.get(), 2 * sizeof(double), cudaMemcpyDeviceToHost));
        return {h[0], h[1]};
    }
    double primal_res2(const Vec& ax) {
        return reduce(m, PrimalResF<V>{ax.get(), lo.get(), hi.get(), R.get()}).first;
    }
    void dual_res(const Vec& aty, const Vec* qx, double& rd2, double& vobj, bool with_c = true) {
        auto r = reduce(n, DualResF<V>{aty.get(), qx ? qx->get() : nullptr, c.get(), l.get(), u.get(), C.get(),
                                       with_c ? 1.0 : 0.0});
        rd2 = r.first;
        vobj = r.second;
    }
    double row_dual_obj(const Vec& y) { return reduce(m, RowDualObjF<V>{y.get(), lo.get(), hi.get()}).first; }
    double dot(const Vec& a, const Vec& b) { return reduce(static_cast<int>(a.size()), DotF<V>{a.get(), b.get()}).first; }
    double dot_c(const Vec& x) { return dot(c, x); }
    double diff2(const Vec& a, const Vec& b) {
        return reduce(static_cast<int>(a.size()), Diff2F<V>{a.get(), b.get()}).first;
    }
};

}  // namespace

Result solve_pdhg_gpu(const LP& lp, const Options& opt) {
    int dev = 0;
    GANIT_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    GANIT_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 6) throw std::runtime_error("GANIT GPU backend needs compute capability >= 6.0");
    std::string name = std::string("gpu:") + prop.name;
    Result r = run_pdhg<CudaBackend<float, float>, CudaBackend<float, double>, CudaBackend<double, double>>(
        lp, opt, name.c_str());
    GANIT_CUDA_CHECK(cudaGetLastError());
    return r;
}

}  // namespace ganit
