"""
Experiment 1 — Replication of the testbed of Section 5.1 (Framinan et al., 2024).

Phase A (screening, Table 1 of the paper): 8 factors x 2 levels, R replicates,
          multi-factor ANOVA on CPU time, fill rate and minimum service level.
Phase B (CPU grid, Table 2 / Figs. 3-5 of the paper, reduced): alpha=0.1, LV fixed,
          more levels for the relevant factors.

Usage:
    python experiments/exp1_testbed.py screening --reps 3 --tl 30 --jobs 2
    python experiments/exp1_testbed.py grid      --reps 3 --tl 30 --jobs 2
Results are appended to results/exp1_<phase>.csv (the run can be resumed).
"""
import argparse
import csv
import itertools
import os
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from op.model import solve_order_promising, true_min_service_level  # noqa: E402
from op.testbed import generate_instance  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIELDS = ["phase", "variant", "alpha", "n_orders", "n_products", "order_size", "homog",
          "sl_var", "stock_cov", "fresh_cov", "rep", "cpu", "hit_limit", "fill_rate",
          "min_sl_model", "min_sl_true", "over_units", "unserved_units", "n_vars", "n_cons",
          "n_batches", "resolved", "repaired"]

SCREENING = dict(alpha=[0.1, 0.9], n_orders=[20, 100], n_products=[20, 500],
                 order_size=["SOZ", "BOZ"], homog=["HH", "LH"], sl_var=["LV", "HV"],
                 stock_cov=["MSC", "FSC"], fresh_cov=["MFC", "FFC"])
VARIANT = dict(alpha=[0.5], n_orders=[20, 60], n_products=[20, 100],
               order_size=["SOZ", "MOZ"], homog=["HH", "LH"], sl_var=["LV", "HV"],
               stock_cov=["MSC", "FSC"], fresh_cov=["MFC", "FFC"])
GRID = dict(alpha=[0.1], n_orders=[50, 100, 150], n_products=[20, 100, 500],
            order_size=["SOZ", "MOZ", "BOZ"], homog=["HH"], sl_var=["LV"],
            stock_cov=["MSC", "FSC"], fresh_cov=["MFC", "HFC", "FFC"])


def run_one(args):
    phase, variant, cfg, rep, tl = args
    # deterministic seed: same instance for both alpha levels / both model variants
    seed = zlib.crc32(repr((cfg["n_orders"], cfg["n_products"], cfg["order_size"], cfg["homog"],
                            cfg["sl_var"], cfg["stock_cov"], cfg["fresh_cov"], rep)).encode())
    orders, batches, hist = generate_instance(cfg["n_orders"], cfg["n_products"],
                                              cfg["order_size"], cfg["homog"], cfg["sl_var"],
                                              cfg["stock_cov"], cfg["fresh_cov"], seed=seed)
    r = solve_order_promising(orders, batches, hist, alpha=cfg["alpha"], variant=variant,
                              time_limit=tl)
    return dict(phase=phase, variant=variant, rep=rep, **cfg, cpu=round(r["cpu"], 3),
                hit_limit=int(r["hit_limit"]),
                fill_rate=round(len(r["served"]) / len(orders), 4),
                min_sl_model=round(r["L"], 4),
                min_sl_true=round(true_min_service_level(orders, r["served"], hist), 4),
                over_units=round(r["over_units"], 1), unserved_units=round(r["unserved_units"], 1),
                n_vars=r["n_vars"], n_cons=r["n_cons"], n_batches=len(batches),
                resolved=int(r.get("resolved", False)), repaired=r.get("repaired", 0))


def key(d):
    return (d["variant"], str(d["alpha"]), str(d["n_orders"]), str(d["n_products"]),
            d["order_size"], d["homog"], d["sl_var"], d["stock_cov"], d["fresh_cov"], str(d["rep"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["screening", "grid", "variant"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tl", type=float, default=30)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--variant", default="paper", choices=["paper", "corrected"])
    a = ap.parse_args()

    levels = dict(screening=SCREENING, grid=GRID, variant=VARIANT)[a.phase]
    variants = ["paper", "corrected"] if a.phase == "variant" else [a.variant]
    names = list(levels)
    out = os.path.join(RES, "exp3a_testbed.csv" if a.phase == "variant" else f"exp1_{a.phase}.csv")
    os.makedirs(RES, exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {key(r) for r in csv.DictReader(f)}
    tasks = []
    for combo in itertools.product(*levels.values()):
        cfg = dict(zip(names, combo))
        for rep in range(a.reps):
            for v in variants:
                t = (a.phase, v, cfg, rep, a.tl)
                if key(dict(variant=v, rep=rep, **cfg)) not in done:
                    tasks.append(t)
    # cheap instances first so partial results cover the whole design early
    tasks.sort(key=lambda t: (t[3], t[2]["n_orders"] * {"SOZ": 3, "MOZ": 7.5, "BOZ": 20}[t[2]["order_size"]]))
    print(f"{len(tasks)} instances to solve ({len(done)} already done)", flush=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f, Pool(a.jobs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for n, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
            w.writerow(row)
            f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
