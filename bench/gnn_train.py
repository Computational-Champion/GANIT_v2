#!/usr/bin/env python3
"""Train GANIT's GNN branching policy by imitating strong branching.

Written from scratch in NumPy (manual backpropagation, Adam) -- no PyTorch, so
the trained model has no ML-framework dependency; inference runs in C++ (src/gnn.cpp).

usage:
  # 1) collect expert samples (strong branching) on small training instances
  for f in train/*.mps; do build/ganit $f --collect data/$(basename $f .mps).bin --collect-max 300 --quiet; done
  # 2) train
  python3 bench/gnn_train.py data/*.bin -o model_ref.gnn --epochs 30
  # 3) use
  build/ganit test.mps --branching gnn --gnn-model model_ref.gnn

  --no-index   ablation: zero the index-aware features (class one-hot + index positions)
  --gradcheck  verify the analytic gradients against finite differences and exit
"""
import argparse, os, struct, sys, time
import numpy as np
import scipy.sparse as sp

FV, FC = 26, 17
IDX_V = list(range(15, 26))   # index-aware variable features
IDX_C = list(range(6, 17))    # index-aware constraint features


# ----------------------------------------------------------------------------- data
def load_file(path):
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 24 or buf[:4] != b"GNS1":
        return None
    off = 4
    n, m, fv, fc, nnz = struct.unpack_from("<5i", buf, off); off += 20
    assert fv == FV and fc == FC, f"{path}: feature dims {fv},{fc} != {FV},{FC}"
    er = np.frombuffer(buf, np.int32, nnz, off); off += 4 * nnz
    ec = np.frombuffer(buf, np.int32, nnz, off); off += 4 * nnz
    ew = np.frombuffer(buf, np.float32, nnz, off).astype(np.float64); off += 4 * nnz
    rdeg = np.maximum(np.bincount(er, minlength=m), 1).astype(np.float64)
    cdeg = np.maximum(np.bincount(ec, minlength=n), 1).astype(np.float64)
    E = sp.csr_matrix((ew / rdeg[er], (er, ec)), shape=(m, n))
    G = sp.csr_matrix((ew / cdeg[ec], (ec, er)), shape=(n, m))
    graph = {"n": n, "m": m, "E": E, "G": G, "ET": E.T.tocsr(), "GT": G.T.tocsr()}
    samples = []
    while off < len(buf):
        K, label = struct.unpack_from("<2i", buf, off); off += 8
        Fv = np.frombuffer(buf, np.float32, n * FV, off).reshape(n, FV).astype(np.float64); off += 4 * n * FV
        Fc = np.frombuffer(buf, np.float32, m * FC, off).reshape(m, FC).astype(np.float64); off += 4 * m * FC
        cand = np.frombuffer(buf, np.int32, K, off).copy(); off += 4 * K
        score = np.frombuffer(buf, np.float32, K, off).astype(np.float64); off += 4 * K
        samples.append({"Fv": Fv, "Fc": Fc, "cand": cand, "label": label, "score": score})
    return graph, samples


# ----------------------------------------------------------------------------- model
PARAMS = [("Wv1", (FV, "D")), ("bv1", (1, "D")), ("Wc1", (FC, "D")), ("bc1", (1, "D")),
          ("Wc2", ("D", "D")), ("Wm1", ("D", "D")), ("bc2", (1, "D")), ("Wv2", ("D", "D")),
          ("Wm2", ("D", "D")), ("bv2", (1, "D")), ("Wo1", ("D", "D")), ("bo1", (1, "D")), ("wo2", ("D", 1))]


def init_params(D, rng):
    p = {}
    for name, shape in PARAMS:
        shp = tuple(D if s == "D" else s for s in shape)
        if name.startswith("b"):
            p[name] = np.zeros(shp)
        else:
            p[name] = rng.normal(0, np.sqrt(2.0 / shp[0]), shp)
    return p


def relu(x):
    return np.maximum(x, 0.0)


def forward(p, g, Fv, Fc, cache=False):
    Zv1 = Fv @ p["Wv1"] + p["bv1"]; Hv = relu(Zv1)
    Zc1 = Fc @ p["Wc1"] + p["bc1"]; Hc = relu(Zc1)
    Mc = g["E"] @ Hv
    Zc2 = Hc @ p["Wc2"] + Mc @ p["Wm1"] + p["bc2"]; Hc2 = relu(Zc2)
    Mv = g["G"] @ Hc2
    Zv2 = Hv @ p["Wv2"] + Mv @ p["Wm2"] + p["bv2"]; Hv2 = relu(Zv2)
    Zo = Hv2 @ p["Wo1"] + p["bo1"]; Ho = relu(Zo)
    s = (Ho @ p["wo2"]).ravel()
    if not cache:
        return s
    return s, dict(Fv=Fv, Fc=Fc, Zv1=Zv1, Hv=Hv, Zc1=Zc1, Hc=Hc, Mc=Mc, Zc2=Zc2, Hc2=Hc2, Mv=Mv,
                   Zv2=Zv2, Hv2=Hv2, Zo=Zo, Ho=Ho)


def loss_and_grad(p, g, smp, Fv, Fc):
    s, c = forward(p, g, Fv, Fc, cache=True)
    cand, label = smp["cand"], smp["label"]
    sk = s[cand]
    sk = sk - sk.max()
    e = np.exp(sk)
    prob = e / e.sum()
    loss = -np.log(max(prob[label], 1e-300))
    ds = np.zeros_like(s)
    dk = prob.copy()
    dk[label] -= 1.0
    ds[cand] = dk
    gr = {}
    dHo = ds[:, None] @ p["wo2"].T
    gr["wo2"] = c["Ho"].T @ ds[:, None]
    dZo = dHo * (c["Zo"] > 0)
    gr["Wo1"] = c["Hv2"].T @ dZo; gr["bo1"] = dZo.sum(0, keepdims=True)
    dHv2 = dZo @ p["Wo1"].T
    dZv2 = dHv2 * (c["Zv2"] > 0)
    gr["Wv2"] = c["Hv"].T @ dZv2; gr["Wm2"] = c["Mv"].T @ dZv2; gr["bv2"] = dZv2.sum(0, keepdims=True)
    dHv = dZv2 @ p["Wv2"].T
    dMv = dZv2 @ p["Wm2"].T
    dHc2 = g["GT"] @ dMv
    dZc2 = dHc2 * (c["Zc2"] > 0)
    gr["Wc2"] = c["Hc"].T @ dZc2; gr["Wm1"] = c["Mc"].T @ dZc2; gr["bc2"] = dZc2.sum(0, keepdims=True)
    dHc = dZc2 @ p["Wc2"].T
    dMc = dZc2 @ p["Wm1"].T
    dHv = dHv + g["ET"] @ dMc
    dZv1 = dHv * (c["Zv1"] > 0)
    gr["Wv1"] = c["Fv"].T @ dZv1; gr["bv1"] = dZv1.sum(0, keepdims=True)
    dZc1 = dHc * (c["Zc1"] > 0)
    gr["Wc1"] = c["Fc"].T @ dZc1; gr["bc1"] = dZc1.sum(0, keepdims=True)
    return loss, gr, s


# ----------------------------------------------------------------------------- utils
def normalise(samples, mv, sv, mc, sc, no_index):
    out = []
    for smp in samples:
        Fv = (smp["Fv"] - mv) / sv
        Fc = (smp["Fc"] - mc) / sc
        if no_index:
            Fv[:, IDX_V] = 0.0
            Fc[:, IDX_C] = 0.0
        out.append((Fv, Fc))
    return out


def evaluate(p, data):
    n = top1 = top5 = 0
    qual = 0.0
    for g, smp, (Fv, Fc) in data:
        s = forward(p, g, Fv, Fc)[smp["cand"]]
        order = np.argsort(-s)
        n += 1
        top1 += order[0] == smp["label"]
        top5 += smp["label"] in order[:5]
        best = smp["score"].max()
        qual += smp["score"][order[0]] / best if best > 0 else 1.0
    return (top1 / n, top5 / n, qual / n) if n else (0, 0, 0)


def save(p, path, D, mv, sv, mc, sc):
    with open(path, "w") as f:
        f.write(f"GANIT_GNN 1 {FV} {FC} {D}\n")
        def arr(name, a):
            a = np.atleast_2d(a)
            f.write(f"{name} {a.shape[0]} {a.shape[1]}\n")
            f.write(" ".join(f"{v:.9g}" for v in a.ravel()) + "\n")
        arr("mean_v", mv); arr("std_v", sv); arr("mean_c", mc); arr("std_c", sc)
        for name, _ in PARAMS:
            arr(name, p[name])


def gradcheck(p, g, smp, Fv, Fc):
    rng = np.random.default_rng(0)
    loss, gr, _ = loss_and_grad(p, g, smp, Fv, Fc)
    worst = 0.0
    for name, _ in PARAMS:
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in p[name].shape)
            old = p[name][idx]
            h = 1e-6
            p[name][idx] = old + h; lp, _, _ = loss_and_grad(p, g, smp, Fv, Fc)
            p[name][idx] = old - h; lm, _, _ = loss_and_grad(p, g, smp, Fv, Fc)
            p[name][idx] = old
            num = (lp - lm) / (2 * h)
            ana = gr[name][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel if abs(num) + abs(ana) > 1e-9 else 0.0)
    return worst


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="model.gnn")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--time", type=float, default=1e9, help="training time limit (s)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    insts = []
    for f in a.files:
        r = load_file(f)
        if r and r[1]:
            insts.append(r)
    if not insts:
        sys.exit("no samples found")
    order = rng.permutation(len(insts))
    nval = max(1, int(round(a.val_frac * len(insts)))) if len(insts) > 1 else 0
    val_i, tr_i = set(order[:nval]), order[nval:]
    tr = [(insts[i][0], s) for i in tr_i for s in insts[i][1]]
    va = [(insts[i][0], s) for i in val_i for s in insts[i][1]]
    allv = np.vstack([s["Fv"] for _, s in tr])
    allc = np.vstack([s["Fc"] for _, s in tr])
    mv, sv = allv.mean(0), allv.std(0)
    mc, sc = allc.mean(0), allc.std(0)
    sv[sv < 1e-6] = 1.0
    sc[sc < 1e-6] = 1.0
    trn = [(g, s, x) for (g, s), x in zip(tr, normalise([s for _, s in tr], mv, sv, mc, sc, a.no_index))]
    van = [(g, s, x) for (g, s), x in zip(va, normalise([s for _, s in va], mv, sv, mc, sc, a.no_index))]
    print(f"instances: {len(tr_i)} train / {len(val_i)} validation | samples: {len(trn)} train / {len(van)} val"
          f"{' | ABLATION: no index features' if a.no_index else ''}")

    p = init_params(a.dim, rng)
    if a.gradcheck:
        g, smp, (Fv, Fc) = trn[0]
        w = gradcheck(p, g, smp, Fv, Fc)
        print(f"gradient check: worst relative error {w:.2e} -> {'OK' if w < 1e-4 else 'FAILED'}")
        return
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best = (-1, None)
    t0 = time.time()
    for ep in range(a.epochs):
        idx = rng.permutation(len(trn))
        tot = 0.0
        for k in idx:
            g, smp, (Fv, Fc) = trn[k]
            loss, gr, _ = loss_and_grad(p, g, smp, Fv, Fc)
            tot += loss
            t += 1
            for name in p:
                gg = gr[name] + 1e-5 * p[name]
                m1[name] = b1 * m1[name] + (1 - b1) * gg
                m2[name] = b2 * m2[name] + (1 - b2) * gg * gg
                mh = m1[name] / (1 - b1 ** t)
                vh = m2[name] / (1 - b2 ** t)
                p[name] -= a.lr * mh / (np.sqrt(vh) + eps)
        tr_m = evaluate(p, trn[:300])
        va_m = evaluate(p, van) if van else (0, 0, 0)
        print(f"epoch {ep+1:3d} | loss {tot/len(trn):.4f} | train top1 {tr_m[0]:.3f} | "
              f"val top1 {va_m[0]:.3f} top5 {va_m[1]:.3f} SB-quality {va_m[2]:.3f} | {time.time()-t0:.0f}s")
        key = va_m[2] if van else tr_m[2]
        if key > best[0]:
            best = (key, {k: v.copy() for k, v in p.items()})
        if time.time() - t0 > a.time:
            break
    save(best[1], a.out, a.dim, mv, sv, mc, sc)
    print(f"saved {a.out} (best validation SB-quality {best[0]:.3f})")


if __name__ == "__main__":
    main()
