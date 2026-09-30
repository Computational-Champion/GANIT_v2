#!/usr/bin/env python3
"""Compare GANIT branching rules on a set of MILP instances.

usage:
  python3 bench/gnn_eval.py --ganit build/ganit --model ref.gnn --rules reliability,pseudocost,gnn \
      --time 60 --csv eval.csv test/*.mps
Reports solved counts, shifted geometric means of nodes and time (shift 10 / 1 s, the
standard MIP benchmarking convention) and checks that all rules reach the same optimum.
"""
import argparse, csv, json, math, os, subprocess, sys


def run(binary, path, rule, model, tlim):
    cmd = [binary, path, "--quiet", "--json", "--time", str(tlim), "--branching", rule]
    if rule in ("gnn", "gnn-strong"):
        cmd += ["--gnn-model", model]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=tlim + 60)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"status": "CRASH", "pobj": float("nan"), "nodes": 0, "solve_s": tlim, "branch_s": 0}


def sgm(vals, shift):
    return math.exp(sum(math.log(v + shift) for v in vals) / len(vals)) - shift if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ganit", default="build/ganit")
    ap.add_argument("--model", default="")
    ap.add_argument("--models", default="", help="extra gnn variants: name=path,name=path")
    ap.add_argument("--rules", default="reliability,pseudocost,gnn")
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rules = [r.strip() for r in a.rules.split(",")]
    variants = [(r, a.model if r in ("gnn", "gnn-strong") else "") for r in rules]
    for kv in filter(None, a.models.split(",")):
        name, path = kv.split("=")
        variants.append((name, path))
    res = {v[0]: [] for v in variants}
    rows = []
    for f in a.files:
        name = os.path.basename(f).rsplit(".", 1)[0]
        line = [f"{name:22s}"]
        objs = []
        for vname, path in variants:
            rule = vname if vname in ("gnn", "gnn-strong") or not path else "gnn"
            d = run(a.ganit, f, rule, path, a.time)
            ok = d["status"] == "OPTIMAL"
            res[vname].append((ok, d["nodes"], d["solve_s"], d.get("branch_s", 0), d["pobj"]))
            if ok:
                objs.append(d["pobj"])
            line.append(f"{vname}:{d['status'][:4]} n={d['nodes']} t={d['solve_s']:.2f}")
            rows.append({"instance": name, "rule": vname, "status": d["status"], "nodes": d["nodes"],
                         "time": d["solve_s"], "branch_time": d.get("branch_s", 0), "obj": d["pobj"]})
        agree = (max(objs) - min(objs)) / (1 + abs(objs[0])) <= 2e-4 if objs else True
        print(" | ".join(line) + ("" if agree else "  <-- OBJECTIVES DIFFER"), file=sys.stderr, flush=True)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    base = variants[0][0]
    # compare on instances solved by all rules
    n_inst = len(a.files)
    all_ok = [all(res[v[0]][i][0] for v in variants) for i in range(n_inst)]
    print(f"\n| Rule | Solved | Nodes (SGM, all-solved set) | Time s (SGM) | Branching time share | vs {base}: nodes / time |")
    print("|---|---:|---:|---:|---:|---:|")
    bn = sgm([res[base][i][1] for i in range(n_inst) if all_ok[i]], 10)
    bt = sgm([res[base][i][2] for i in range(n_inst) if all_ok[i]], 1)
    for vname, _ in variants:
        r = res[vname]
        solved = sum(1 for x in r if x[0])
        nodes = sgm([r[i][1] for i in range(n_inst) if all_ok[i]], 10)
        tm = sgm([r[i][2] for i in range(n_inst) if all_ok[i]], 1)
        bshare = sum(x[3] for x in r) / max(1e-9, sum(x[2] for x in r))
        print(f"| {vname} | {solved}/{n_inst} | {nodes:.1f} | {tm:.3f} | {100*bshare:.0f}% | "
              f"{nodes/bn:.2f}x / {tm/bt:.2f}x |")
    print(f"\n({sum(all_ok)} instances solved by every rule are used for the SGM columns)")


if __name__ == "__main__":
    main()
