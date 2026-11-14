"""
Experiments 2 and 3: multi-day simulation of the DSS (case-study replication + extensions).

Exp 2 (replication of Section 5.2 / Figs. 6-8): FCFS, GO, RE-NOCM, RE with 4 and 5
          iterations per day.
Exp 3 (extensions proposed by the group):
    3a  MILP as printed ("paper") vs corrected MILP inside GO / RE-NOCM / RE
    3b  SATP: none vs common p_hat (paper) vs product-specific p_hat_j, and reserve
        multiplier beta
    3c  number of iterations T = 2..10 (trade-off described in Section 4)
    3d  weight alpha in the objective function
    3e  OCM: acceptance probability and global visibility of offers
    3f  objective term 1 weighted by orders (paper) or by volume (units)

Usage:  python experiments/exp_sim.py [all|exp2|3a|3b|3c|3d|3e|3f] --seeds 3 --days 20
Each world seed = an independent synthetic company; every policy sees exactly the
same orders and stock (common random numbers).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from op.sim import PolicyConfig, Simulator, World, WorldConfig  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
WARMUP = 15


def build(seed, days):
    w = World(WorldConfig(), seed=seed)
    hist = [w.make_day() for _ in range(WARMUP)]
    ev = [w.make_day() for _ in range(days)]
    return Simulator(w, ev, hist)


def job(args):
    seed, days, label, pc_kwargs = args
    sim = build(seed, days)
    t = time.time()
    rows = sim.run(PolicyConfig(seed=seed, **pc_kwargs))
    for r in rows:
        r.update(seed=seed, label=label, wall=time.time() - t, **{f"cfg_{k}": v for k, v in pc_kwargs.items()})
    return rows


def configs(which):
    C = []
    if which in ("exp2", "all"):
        for T in (4, 5):
            for name in ("FCFS", "GO", "RE-NOCM", "RE"):
                C.append(("exp2", dict(name=name, iterations=T)))
    if which in ("3a", "all"):
        for v in ("paper", "corrected"):
            for name in ("GO", "RE-NOCM", "RE"):
                C.append(("3a", dict(name=name, iterations=5, variant=v)))
    if which in ("3b", "all"):
        for name in ("RE-NOCM", "RE"):
            C.append(("3b", dict(name=name, iterations=5, satp="none", beta=0.0)))
            for mode in ("common", "product"):
                for beta in (0.5, 1.0, 1.5, 2.0):
                    C.append(("3b", dict(name=name, iterations=5, satp=mode, beta=beta)))
    if which in ("3c", "all"):
        for T in (2, 3, 4, 5, 6, 8, 10):
            for name in ("RE-NOCM", "RE"):
                C.append(("3c", dict(name=name, iterations=T)))
    if which in ("3d", "all"):
        for a in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            for name in ("GO", "RE"):
                C.append(("3d", dict(name=name, iterations=5, alpha=a)))
    if which in ("3e", "all"):
        for p in (0.3, 0.5, 0.7, 0.9):
            for gv in (True, False):
                C.append(("3e", dict(name="RE", iterations=5, p_accept=p, global_visibility=gv)))
    if which in ("3f", "all"):
        for wgt in ("orders", "volume"):
            for name in ("GO", "RE"):
                C.append(("3f", dict(name=name, iterations=5, weight=wgt)))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", default="all", nargs="?")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    tasks = [(s, a.days, lab, kw) for lab, kw in configs(a.which) for s in range(1, a.seeds + 1)]
    print(f"{len(tasks)} simulation runs", flush=True)
    rows = []
    with Pool(a.jobs) as pool:
        for n, r in enumerate(pool.imap_unordered(job, tasks), 1):
            rows += r
            if n % 10 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    os.makedirs(RES, exist_ok=True)
    df = pd.DataFrame(rows)
    for lab, g in df.groupby("label"):
        g.to_csv(os.path.join(RES, f"sim_{lab}.csv"), index=False)
    print("saved", sorted(df.label.unique()))


if __name__ == "__main__":
    main()
