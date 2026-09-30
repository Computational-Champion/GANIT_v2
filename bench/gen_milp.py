#!/usr/bin/env python3
"""Multi-index MILP generators for GANIT (MPS output).

1) refinery : crude scheduling with integer cargoes and CDU operating modes
   indices  c crude, p product, k quality spec, m CDU mode, t period
   integer  NC_c_t in {0,1,2}   number of cargoes of crude c arriving in period t
   binary   Z_m_t               CDU runs in mode m in period t (at most one mode)
   contin.  X_c_p_t  crude c processed into product p
            INV_c_t  crude tank inventory,  PINV_p_t product inventory
            SELL_p_t sales (<= demand),     SW_t mode-switch indicator
   rows     CB_c_t   crude balance    INV[t-1] + size_c*NC - sum_p X - INV = 0
            PB_p_t   product balance  PINV[t-1] + sum_c X - SELL - PINV = 0
            Q_p_k_t  product quality  sum_c (q_ck - qmax_pk) X_cpt <= 0
            TU_t/TL_t mode throughput band   L_m Z <= sum X <= U_m Z
            ONE_t    sum_m Z_m_t <= 1
            SUL_t    crude-diet sulfur limited by the active mode
            SWR_m_t  SW_t >= Z_m_t - Z_m_(t-1)
   Always feasible (everything zero) and bounded.

2) uc : unit commitment (power system scheduling)
   indices  g unit, t hour
   binary   U_g_t on/off       contin. V_g_t start-up, W_g_t shut-down (in [0,1]),
            P_g_t output,      SHED_t unserved load (penalised)
   rows     DEM_t   sum_g P + SHED = demand
            RES_t   sum_g Pmax U >= (1+r) demand - big*SHED   (spinning reserve)
            PMAX/PMIN_g_t  Pmin U <= P <= Pmax U
            SU_g_t  V - W - U_t + U_(t-1) = 0      (start-up / shut-down logic)
            RU/RD_g_t  ramping limits
            MUT/MDT_g_t minimum up / down times
   Always feasible thanks to load shedding.

usage:
  python3 bench/gen_milp.py refinery --crudes 4 --products 3 --modes 3 --periods 8 --seed 1 -o ref.mps
  python3 bench/gen_milp.py uc --units 10 --hours 24 --seed 1 -o uc.mps
  python3 bench/gen_milp.py batch --kind uc --count 40 --units 6 --hours 12 --seed0 100 --outdir train/
"""
import argparse, os, random


class Mps:
    def __init__(self, name):
        self.name = name
        self.rows = []            # (name, type)
        self.rhs = {}
        self.cols = {}            # col -> list[(row, val)]
        self.order = []
        self.obj = {}
        self.bnd = {}             # col -> (lo, hi)
        self.ints = set()

    def row(self, name, typ, rhs=0.0):
        self.rows.append((name, typ))
        if rhs:
            self.rhs[name] = rhs
        return name

    def col(self, name, cost=0.0, lo=0.0, hi=None, integer=False):
        self.order.append(name)
        self.cols[name] = []
        if cost:
            self.obj[name] = cost
        self.bnd[name] = (lo, hi)
        if integer:
            self.ints.add(name)
        return name

    def a(self, col, row, v):
        if v:
            self.cols[col].append((row, v))

    def write(self, path):
        with open(path, "w") as f:
            f.write(f"NAME {self.name}\nROWS\n N OBJ\n")
            for r, t in self.rows:
                f.write(f" {t} {r}\n")
            f.write("COLUMNS\n")
            in_int = False
            for c in self.order:
                is_int = c in self.ints
                if is_int and not in_int:
                    f.write("    MARKER 'MARKER' 'INTORG'\n")
                    in_int = True
                elif not is_int and in_int:
                    f.write("    MARKER 'MARKER' 'INTEND'\n")
                    in_int = False
                if c in self.obj:
                    f.write(f"    {c} OBJ {self.obj[c]:.8g}\n")
                for r, v in self.cols[c]:
                    f.write(f"    {c} {r} {v:.8g}\n")
                if not self.cols[c] and c not in self.obj:
                    f.write(f"    {c} OBJ 0\n")
            if in_int:
                f.write("    MARKER 'MARKER' 'INTEND'\n")
            f.write("RHS\n")
            for r, v in self.rhs.items():
                f.write(f"    RHS {r} {v:.8g}\n")
            f.write("BOUNDS\n")
            for c in self.order:
                lo, hi = self.bnd[c]
                if lo is None and hi is None:
                    f.write(f" FR BND {c}\n")
                    continue
                if lo is None:
                    f.write(f" MI BND {c}\n")
                elif lo != 0.0:
                    f.write(f" LO BND {c} {lo:.8g}\n")
                if hi is not None:
                    f.write(f" UP BND {c} {hi:.8g}\n")
            f.write("ENDATA\n")
        nnz = sum(len(v) for v in self.cols.values())
        return len(self.rows), len(self.order), len(self.ints), nnz


def refinery(C, P, M, T, K, seed):
    rnd = random.Random(seed)
    mp = Mps(f"REF_C{C}_P{P}_M{M}_T{T}_S{seed}")
    size = [rnd.choice([300, 400, 500, 600]) for _ in range(C)]
    price = [rnd.uniform(40, 70) for _ in range(C)]
    freight = [rnd.uniform(2000, 6000) for _ in range(C)]
    q = [[rnd.uniform(0.05, 1.0) for _ in range(K)] for _ in range(C)]
    q[0] = [0.05] * K
    sulf = [rnd.uniform(0.2, 3.0) for _ in range(C)]
    sulf[0] = 0.2
    qmax = [[rnd.uniform(0.3, 0.8) for _ in range(K)] for _ in range(P)]
    sell_price = [rnd.uniform(75, 110) for _ in range(P)]
    modes = []
    base = sum(size) / C * 1.2
    for m in range(M):
        lo = base * (0.3 + 0.35 * m)
        hi = lo * rnd.uniform(1.4, 1.9)
        modes.append((lo, hi, rnd.uniform(0.6, 2.5), rnd.uniform(1500, 4000)))  # (L, U, smax, opcost)
    switch_cost = rnd.uniform(3000, 8000)
    tank = [s * 2.5 for s in size]
    init = [rnd.uniform(0, 0.5) * t for t in tank]

    for t in range(T):
        for c in range(C):
            mp.row(f"CB_{c}_{t}", "E", -init[c] if t == 0 else 0.0)
        for p in range(P):
            mp.row(f"PB_{p}_{t}", "E")
            for k in range(K):
                mp.row(f"Q_{p}_{k}_{t}", "L")
        mp.row(f"TU_{t}", "L")
        mp.row(f"TL_{t}", "G")
        mp.row(f"ONE_{t}", "L", 1.0)
        mp.row(f"SUL_{t}", "L")
        if t > 0:
            for m in range(M):
                mp.row(f"SWR_{m}_{t}", "G")

    for t in range(T):
        for c in range(C):
            nc = mp.col(f"NC_{c}_{t}", cost=size[c] * price[c] * rnd.uniform(0.95, 1.05) + freight[c],
                        lo=0.0, hi=2.0, integer=True)
            mp.a(nc, f"CB_{c}_{t}", size[c])
            iv = mp.col(f"INV_{c}_{t}", cost=0.3, hi=tank[c])
            mp.a(iv, f"CB_{c}_{t}", -1.0)
            if t + 1 < T:
                mp.a(iv, f"CB_{c}_{t+1}", 1.0)
            for p in range(P):
                x = mp.col(f"X_{c}_{p}_{t}")
                mp.a(x, f"CB_{c}_{t}", -1.0)
                mp.a(x, f"PB_{p}_{t}", 1.0)
                for k in range(K):
                    mp.a(x, f"Q_{p}_{k}_{t}", round(q[c][k] - qmax[p][k], 6))
                mp.a(x, f"TU_{t}", 1.0)
                mp.a(x, f"TL_{t}", 1.0)
                mp.a(x, f"SUL_{t}", sulf[c])
        for m, (L, U, smax, opc) in enumerate(modes):
            z = mp.col(f"Z_{m}_{t}", cost=opc, lo=0.0, hi=1.0, integer=True)
            mp.a(z, f"TU_{t}", -U)
            mp.a(z, f"TL_{t}", -L)
            mp.a(z, f"ONE_{t}", 1.0)
            mp.a(z, f"SUL_{t}", -smax * U)
            if t > 0:
                mp.a(z, f"SWR_{m}_{t}", -1.0)
            if t + 1 < T:
                mp.a(z, f"SWR_{m}_{t+1}", 1.0)
        if t > 0:
            sw = mp.col(f"SW_{t}", cost=switch_cost, hi=1.0)
            for m in range(M):
                mp.a(sw, f"SWR_{m}_{t}", 1.0)
        for p in range(P):
            demand = rnd.uniform(0.15, 0.4) * base
            s = mp.col(f"SELL_{p}_{t}", cost=-sell_price[p] * rnd.uniform(0.95, 1.05), hi=demand)
            mp.a(s, f"PB_{p}_{t}", -1.0)
            pi = mp.col(f"PINV_{p}_{t}", cost=0.5, hi=base)
            mp.a(pi, f"PB_{p}_{t}", -1.0)
            if t + 1 < T:
                mp.a(pi, f"PB_{p}_{t+1}", 1.0)
    return mp


def uc(G, T, seed, reserve=0.1):
    rnd = random.Random(seed)
    mp = Mps(f"UC_G{G}_T{T}_S{seed}")
    units = []
    for g in range(G):
        pmax = rnd.choice([50, 80, 100, 150, 200, 300, 400])
        pmin = pmax * rnd.uniform(0.2, 0.5)
        b = rnd.uniform(15, 60) * (1.3 if pmax < 120 else 1.0)  # small units are more expensive
        a = rnd.uniform(1.0, 4.0) * pmax
        su = rnd.uniform(20, 80) * pmax
        ramp = pmax * rnd.uniform(0.3, 0.7)
        ut = rnd.randint(1, 5)
        dt = rnd.randint(1, 4)
        on0 = rnd.random() < 0.5
        units.append((pmin, pmax, a, b, su, ramp, ut, dt, on0))
    cap = sum(u[1] for u in units)
    peak = cap / (1.0 + reserve) / 1.25
    demand = []
    for t in range(T):
        h = t % 24
        shape = 0.55 + 0.45 * max(0.0, __import__("math").sin((h - 6) / 24 * 2 * 3.14159)) + (0.1 if 17 <= h <= 21 else 0)
        demand.append(peak * min(1.0, shape) * rnd.uniform(0.95, 1.05))

    for t in range(T):
        mp.row(f"DEM_{t}", "E", demand[t])
        mp.row(f"RES_{t}", "G", (1 + reserve) * demand[t])
        for g in range(G):
            mp.row(f"PMAX_{g}_{t}", "L")
            mp.row(f"PMIN_{g}_{t}", "G")
            mp.row(f"SU_{g}_{t}", "E", (-1.0 if units[g][8] else 0.0) if t == 0 else 0.0)
            if t > 0:
                mp.row(f"RU_{g}_{t}", "L")
                mp.row(f"RD_{g}_{t}", "L")
            mp.row(f"MUT_{g}_{t}", "L")
            mp.row(f"MDT_{g}_{t}", "L", 1.0)

    for t in range(T):
        for g, (pmin, pmax, a, b, su, ramp, ut, dt, on0) in enumerate(units):
            u = mp.col(f"U_{g}_{t}", cost=a, lo=0.0, hi=1.0, integer=True)
            mp.a(u, f"RES_{t}", pmax)
            mp.a(u, f"PMAX_{g}_{t}", -pmax)
            mp.a(u, f"PMIN_{g}_{t}", -pmin)
            mp.a(u, f"SU_{g}_{t}", -1.0)
            if t + 1 < T:
                mp.a(u, f"SU_{g}_{t+1}", 1.0)
                mp.a(u, f"RU_{g}_{t+1}", -ramp)
            if t > 0:
                mp.a(u, f"RD_{g}_{t}", -ramp)
            mp.a(u, f"MUT_{g}_{t}", -1.0)
            mp.a(u, f"MDT_{g}_{t}", 1.0)
            v = mp.col(f"V_{g}_{t}", cost=su, hi=1.0)
            mp.a(v, f"SU_{g}_{t}", 1.0)
            if t > 0:
                mp.a(v, f"RU_{g}_{t}", -pmin)
            for tau in range(t, min(T, t + ut)):
                mp.a(v, f"MUT_{g}_{tau}", 1.0)
            w = mp.col(f"W_{g}_{t}", hi=1.0)
            mp.a(w, f"SU_{g}_{t}", -1.0)
            if t > 0:
                mp.a(w, f"RD_{g}_{t}", -pmax)
            for tau in range(t, min(T, t + dt)):
                mp.a(w, f"MDT_{g}_{tau}", 1.0)
            pcol = mp.col(f"P_{g}_{t}", cost=b, hi=pmax)
            mp.a(pcol, f"DEM_{t}", 1.0)
            mp.a(pcol, f"PMAX_{g}_{t}", 1.0)
            mp.a(pcol, f"PMIN_{g}_{t}", 1.0)
            if t > 0:
                mp.a(pcol, f"RU_{g}_{t}", 1.0)
                mp.a(pcol, f"RD_{g}_{t}", -1.0)
            if t + 1 < T:
                mp.a(pcol, f"RU_{g}_{t+1}", -1.0)
                mp.a(pcol, f"RD_{g}_{t+1}", 1.0)
        shed = mp.col(f"SHED_{t}", cost=1000.0, hi=demand[t])
        mp.a(shed, f"DEM_{t}", 1.0)
        mp.a(shed, f"RES_{t}", 1.0 + reserve)
    # initial ramp rows (t=0) omitted: initial output is free
    return mp


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    r = sub.add_parser("refinery")
    r.add_argument("--crudes", type=int, default=4)
    r.add_argument("--products", type=int, default=3)
    r.add_argument("--modes", type=int, default=3)
    r.add_argument("--periods", type=int, default=8)
    r.add_argument("--specs", type=int, default=2)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("-o", "--out", default="refinery.mps")
    u = sub.add_parser("uc")
    u.add_argument("--units", type=int, default=10)
    u.add_argument("--hours", type=int, default=24)
    u.add_argument("--seed", type=int, default=1)
    u.add_argument("-o", "--out", default="uc.mps")
    b = sub.add_parser("batch")
    b.add_argument("--type", choices=["refinery", "uc"], required=True)
    b.add_argument("--count", type=int, default=20)
    b.add_argument("--seed0", type=int, default=1000)
    b.add_argument("--outdir", default="train")
    b.add_argument("--crudes", type=int, default=4)
    b.add_argument("--products", type=int, default=3)
    b.add_argument("--modes", type=int, default=3)
    b.add_argument("--periods", type=int, default=8)
    b.add_argument("--specs", type=int, default=2)
    b.add_argument("--units", type=int, default=8)
    b.add_argument("--hours", type=int, default=12)
    a = ap.parse_args()
    if a.kind == "refinery":
        st = refinery(a.crudes, a.products, a.modes, a.periods, a.specs, a.seed).write(a.out)
        print(f"wrote {a.out}: {st[0]} rows, {st[1]} cols, {st[2]} integer, {st[3]} nnz")
    elif a.kind == "uc":
        st = uc(a.units, a.hours, a.seed).write(a.out)
        print(f"wrote {a.out}: {st[0]} rows, {st[1]} cols, {st[2]} integer, {st[3]} nnz")
    else:
        os.makedirs(a.outdir, exist_ok=True)
        for i in range(a.count):
            seed = a.seed0 + i
            if a.type == "uc":
                mp = uc(a.units, a.hours, seed)
                path = os.path.join(a.outdir, f"uc_g{a.units}_t{a.hours}_s{seed}.mps")
            else:
                mp = refinery(a.crudes, a.products, a.modes, a.periods, a.specs, seed)
                path = os.path.join(a.outdir, f"ref_c{a.crudes}_t{a.periods}_s{seed}.mps")
            mp.write(path)
        print(f"wrote {a.count} {a.type} instances to {a.outdir}/")


if __name__ == "__main__":
    main()
