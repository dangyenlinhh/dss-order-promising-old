"""
Multi-day simulation of the order promising process of the case study
(Framinan et al., 2024, Sections 3, 4 and 5.2), used to compare:

    FCFS     : ATP at the start of the day allocated to orders in arrival order
    GO       : one MILP run with all orders of the day and the start-of-day ATP (benchmark)
    RE-NOCM  : iterative framework (SATP, 2 MILP runs per iteration), red orders rejected
    RE       : iterative framework + Order Confirmation & Modification (OCM)

Because the company data are confidential, a synthetic "world" is generated that
reproduces the qualitative features described in Section 3 of the paper:
two channels sharing SKUs (HORECA: many small, erratic orders; FOOD: few large
orders with strict freshness), multi-line orders that must be served in-full,
customer/product-specific freshness limits, a 8:00-15:00 shift, products grouped
in families of formats (substitutes), and days with stock scarcity.
All parameters are in WorldConfig and documented in README.md.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .model import solve_order_promising


# ----------------------------------------------------------------------------
# World / data generation
# ----------------------------------------------------------------------------
@dataclass
class WorldConfig:
    n_families: int = 20            # beer brands
    formats_per_family: int = 3     # bottle sizes / packs  -> 60 SKUs
    n_horeca: int = 200
    n_food: int = 15
    p_order_horeca: float = 0.45    # probability a customer orders on a given day
    p_order_food: float = 0.60
    lines_horeca: tuple = (2, 8)
    lines_food: tuple = (6, 12)
    qty_horeca: tuple = (1, 12)
    qty_food: tuple = (10, 60)
    maxage_horeca: tuple = (40, 90)  # freshness limit (days since production)
    maxage_food: tuple = (30, 60)    # FOOD customers are stricter
    batches_per_product: int = 4
    batch_age: tuple = (0, 45)
    p_scarce_day: float = 0.35      # share of days with general shortage
    cov_normal: tuple = (0.95, 1.25)  # stock / expected demand
    cov_scarce: tuple = (0.65, 0.85)
    p_product_shock: float = 0.03   # production breakdown / delivery delay
    shock_factor: float = 0.3
    overnight_food: float = 0.6     # share of orders already waiting at 8:00
    overnight_horeca: float = 0.3


@dataclass
class Day:
    orders: list            # dicts with id, customer, lines, maxage, arrival (0..1 of shift)
    batches: list           # dicts with id, product, qty, age
    scarce: bool


class World:
    def __init__(self, cfg: WorldConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        c = cfg
        self.nP = c.n_families * c.formats_per_family
        self.family = np.arange(self.nP) // c.formats_per_family
        pop = 1.0 / np.arange(1, self.nP + 1) ** 0.8
        r.shuffle(pop)
        # channel affinity: some formats (kegs) mostly HORECA, others (multipacks) FOOD
        aff = r.beta(2, 2, size=self.nP)
        self.w_food = pop * aff / (pop * aff).sum()
        self.w_horeca = pop * (1 - aff) / (pop * (1 - aff)).sum()
        self.customers = []
        for k in range(c.n_horeca + c.n_food):
            ch = "HORECA" if k < c.n_horeca else "FOOD"
            lo, hi = c.maxage_horeca if ch == "HORECA" else c.maxage_food
            self.customers.append(dict(id=k, channel=ch,
                                       maxage=r.integers(lo, hi + 1, size=self.nP)))
        self._oid = 0
        # expected daily demand per product (for stock generation), from 60 sample days
        self.mu = np.zeros(self.nP)
        for _ in range(60):
            for o in self._orders_of_day():
                for j, q in o["lines"].items():
                    self.mu[j] += q / 60.0

    def _orders_of_day(self):
        c, r = self.cfg, self.rng
        orders = []
        for cu in self.customers:
            food = cu["channel"] == "FOOD"
            if r.random() > (c.p_order_food if food else c.p_order_horeca):
                continue
            lo, hi = c.lines_food if food else c.lines_horeca
            n = int(r.integers(lo, hi + 1))
            w = self.w_food if food else self.w_horeca
            prods = r.choice(self.nP, size=min(n, self.nP), replace=False, p=w)
            qlo, qhi = c.qty_food if food else c.qty_horeca
            lines = {int(j): int(r.integers(qlo, qhi + 1)) for j in prods}
            if r.random() < (c.overnight_food if food else c.overnight_horeca):
                arr = 0.0
            else:
                arr = float(r.uniform(0, 0.5)) if food else float(r.beta(2, 3))
            orders.append(dict(id=self._oid, customer=cu["id"], lines=lines,
                               maxage={j: int(cu["maxage"][j]) for j in lines},
                               arrival=arr, channel=cu["channel"]))
            self._oid += 1
        return orders

    def make_day(self) -> Day:
        c, r = self.cfg, self.rng
        orders = self._orders_of_day()
        scarce = r.random() < c.p_scarce_day
        lo, hi = c.cov_scarce if scarce else c.cov_normal
        cov = r.uniform(lo, hi)
        batches, bid = [], 0
        for j in range(self.nP):
            f = cov * r.uniform(0.85, 1.15)
            if r.random() < c.p_product_shock:
                f *= c.shock_factor
            S = int(round(self.mu[j] * f))
            if S <= 0:
                continue
            parts = r.dirichlet(np.ones(c.batches_per_product)) * S
            for q in parts:
                q = int(round(q))
                if q > 0:
                    batches.append(dict(id=bid, product=j, qty=q,
                                        age=int(r.integers(c.batch_age[0], c.batch_age[1] + 1))))
                    bid += 1
        return Day(orders=orders, batches=batches, scarce=scarce)


# ----------------------------------------------------------------------------
# Forecasts used by SATP
# ----------------------------------------------------------------------------
def pending_ratio_curves(history_days, nP, grid=101):
    """p_hat(t): share of the daily volume that arrives AFTER time t (t in [0,1]).
    Returns (common curve, per-product curves) evaluated on a grid."""
    ts = np.linspace(0, 1, grid)
    tot = np.zeros(nP)
    after = np.zeros((nP, grid))
    for d in history_days:
        for o in d.orders:
            for j, q in o["lines"].items():
                tot[j] += q
                after[j] += q * (o["arrival"] > ts)
    common = after.sum(0) / max(tot.sum(), 1)
    per = np.where(tot[:, None] > 0, after / np.maximum(tot[:, None], 1), common[None, :])
    return ts, common, per


def daily_demand(day, nP):
    v = np.zeros(nP)
    for o in day.orders:
        for j, q in o["lines"].items():
            v[j] += q
    return v


# ----------------------------------------------------------------------------
# Stock helpers
# ----------------------------------------------------------------------------
def _feasible_avail(stock, batches_by_prod, j, maxage):
    return sum(stock[b] for b, age in batches_by_prod.get(j, []) if age <= maxage)


def _fefo_take(stock, batches_by_prod, j, maxage, qty):
    """Take qty of product j from oldest feasible batches. Returns {batch: qty} or None."""
    cand = sorted([(age, b) for b, age in batches_by_prod.get(j, []) if age <= maxage],
                  reverse=True)
    if sum(stock[b] for _, b in cand) + 1e-9 < qty:
        return None
    take, need = {}, qty
    for _, b in cand:
        t = min(stock[b], need)
        if t > 0:
            take[b] = t
            need -= t
        if need <= 1e-9:
            break
    return take


# ----------------------------------------------------------------------------
# Policy runner
# ----------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    name: str                      # FCFS | GO | RE-NOCM | RE
    iterations: int = 5
    alpha: float = 0.5
    D: int = 5
    variant: str = "corrected"     # MILP variant
    satp: str = "common"           # common | product | none
    beta: float = 1.0              # multiplier of the reserved quantity in SATP
    p_accept: float = 0.7          # probability a contacted customer accepts a modification
    global_visibility: bool = True # OCM sees offers already made to other customers
    time_limit: float = 20.0
    weight: str = "orders"         # objective term 1: "orders" (paper) or "volume" (extension)
    seed: int = 0


class Simulator:
    def __init__(self, world: World, days: list, history: list, D_hat=5):
        self.world = world
        self.days = days            # evaluation days
        self.history = history      # warm-up days (forecasts only)
        self.nP = world.nP
        _, self.p_common, self.p_prod = pending_ratio_curves(history, self.nP)
        self.demand_hist = [daily_demand(d, self.nP) for d in history]
        self.D_hat = D_hat

    def _p_hat(self, t, mode):
        idx = int(round(min(max(t, 0), 1) * 100))
        if mode == "product":
            return self.p_prod[:, idx]
        return np.full(self.nP, self.p_common[idx])

    def run(self, pc: PolicyConfig):
        rng = np.random.default_rng(pc.seed)
        cust_hist = defaultdict(lambda: deque(maxlen=pc.D))   # past D days (ordered, served)
        dem_hist = list(self.demand_hist[-self.D_hat:])
        out = []
        for di, day in enumerate(self.days):
            d_hat = np.mean(dem_hist[-self.D_hat:], axis=0)
            res = self._run_day(day, pc, cust_hist, d_hat, rng)
            res.update(day=di, scarce=int(day.scarce), policy=pc.name, T=pc.iterations)
            out.append(res)
            dem_hist.append(daily_demand(day, self.nP))
        return out

    # ------------------------------------------------------------------
    def _run_day(self, day, pc, cust_hist, d_hat, rng):
        stock = {b["id"]: b["qty"] for b in day.batches}
        bage = {b["id"]: b["age"] for b in day.batches}
        bprod = {b["id"]: b["product"] for b in day.batches}
        by_prod = defaultdict(list)
        for b in day.batches:
            by_prod[b["product"]].append((b["id"], b["age"]))
        orig = {o["id"]: o for o in day.orders}
        cust_maxage = {cu["id"]: cu["maxage"] for cu in self.world.customers}

        # today's bookkeeping
        t_ord = defaultdict(float)     # units originally ordered (resolved orders)
        t_srv = defaultdict(float)     # units delivered (incl. substitutes)
        t_srv_strict = defaultdict(float)  # units of originally requested products delivered
        status = {}                    # order id -> served | rejected
        stats = dict(cpu=0.0, n_runs=0, n_mod=0, n_subst=0, n_dropped_lines=0,
                     n_hit_limit=0, over_units=0.0, unserved_units=0.0, n_resolved=0,
                     n_repaired=0)

        def hist_view():
            h = {}
            for k in set(o["customer"] for o in day.orders):
                d = sum(x[0] for x in cust_hist[k]) + t_ord[k]
                e = sum(x[1] for x in cust_hist[k]) + t_srv[k]
                h[k] = (d, e)
            return h

        def commit(o, alloc_lines):
            """alloc_lines: {batch: qty} actually shipped for order o (current version)."""
            for b, q in alloc_lines.items():
                stock[b] -= q
                # solver tolerances may overshoot by ~1e-5 units on large quantities
                assert stock[b] > -1e-3 * max(1.0, q), "negative stock"
                stock[b] = max(0.0, stock[b])
            oo = orig[o["id"]]
            t_ord[o["customer"]] += sum(oo["lines"].values())
            t_srv[o["customer"]] += sum(o["lines"].values())
            t_srv_strict[o["customer"]] += sum(min(q, o["lines"].get(j, 0))
                                               for j, q in oo["lines"].items())
            status[o["id"]] = "served"

        def reject(o):
            t_ord[o["customer"]] += sum(orig[o["id"]]["lines"].values())
            status[o["id"]] = "rejected"

        def milp(orders, stk):
            batches = [dict(id=b, product=bprod[b], qty=q, age=bage[b])
                       for b, q in stk.items() if q > 1e-9]
            r = solve_order_promising(orders, batches, hist_view(), alpha=pc.alpha,
                                      variant=pc.variant, time_limit=pc.time_limit,
                                      weight=pc.weight)
            stats["cpu"] += r["cpu"]
            stats["n_runs"] += 1
            stats["n_hit_limit"] += int(r.get("hit_limit", False))
            stats["n_resolved"] += int(r.get("resolved", False))
            stats["n_repaired"] += int(r.get("repaired", 0))
            stats["over_units"] += r.get("over_units", 0.0)
            stats["unserved_units"] += r.get("unserved_units", 0.0)
            per_order = defaultdict(dict)
            for (i, b), q in r["alloc"].items():
                per_order[i][b] = per_order[i].get(b, 0) + q
            return set(r["served"]), per_order

        # ---------------- FCFS ----------------
        if pc.name == "FCFS":
            for o in sorted(day.orders, key=lambda o: (o["arrival"], o["id"])):
                takes, ok = {}, True
                tmp = dict(stock)
                for j, q in o["lines"].items():
                    t = _fefo_take(tmp, by_prod, j, o["maxage"][j], q)
                    if t is None:
                        ok = False
                        break
                    for b, x in t.items():
                        tmp[b] -= x
                        takes[b] = takes.get(b, 0) + x
                commit(o, takes) if ok else reject(o)

        # ---------------- GO ----------------
        elif pc.name == "GO":
            served, alloc = milp(day.orders, dict(stock))
            for o in day.orders:
                commit(o, alloc[o["id"]]) if o["id"] in served else reject(o)

        # ---------------- Iterative (RE-NOCM / RE) ----------------
        else:
            T = pc.iterations
            pool = {}                       # current version of open orders
            arrived = sorted(day.orders, key=lambda o: o["arrival"])
            ai = 0
            for k in range(1, T + 1):
                now = k / T
                while ai < len(arrived) and arrived[ai]["arrival"] <= now + 1e-12:
                    o = arrived[ai]
                    pool[o["id"]] = dict(o)
                    ai += 1
                if not pool:
                    continue
                last = k == T
                # pre-processing: a line with less feasible stock than requested -> red
                red, cand = [], []
                for o in pool.values():
                    if all(_feasible_avail(stock, by_prod, j, o["maxage"][j]) >= q
                           for j, q in o["lines"].items()):
                        cand.append(o)
                    else:
                        red.append(o)
                # step 2: SATP
                if pc.satp == "none" or last:
                    satp_stock = dict(stock)
                else:
                    ph = self._p_hat(now, pc.satp)
                    atp = np.zeros(self.nP)
                    for b, q in stock.items():
                        atp[bprod[b]] += q
                    reserve = pc.beta * d_hat * ph
                    ratio = np.where(atp > 0, np.clip((atp - reserve) / np.maximum(atp, 1e-9), 0, 1), 0)
                    satp_stock = {b: np.floor(q * ratio[bprod[b]]) for b, q in stock.items()}
                # step 3: MILP run 1 -> green
                green, alloc1 = milp(cand, satp_stock) if cand else (set(), {})
                for o in cand:
                    if o["id"] in green:
                        commit(o, alloc1[o["id"]])
                        del pool[o["id"]]
                # steps 4-5: UATP and MILP run 2 -> orange / red
                rest = [o for o in cand if o["id"] not in green]
                orange, alloc2 = milp(rest, dict(stock)) if rest else (set(), {})
                red += [o for o in rest if o["id"] not in orange]
                if last:
                    for o in rest:
                        if o["id"] in orange:
                            commit(o, alloc2[o["id"]])
                            del pool[o["id"]]
                    for o in red:
                        reject(o)
                        del pool[o["id"]]
                    break
                # OCM
                if pc.name == "RE-NOCM":
                    for o in red:
                        reject(o)
                        del pool[o["id"]]
                    continue
                # remaining stock after green (committed) and orange (virtually booked)
                virt = dict(stock)
                for o in rest:
                    if o["id"] in orange:
                        for b, q in alloc2[o["id"]].items():
                            virt[b] -= q
                offered_blind = dict(stock)  # what a CMT member without global view sees
                for o in sorted(red, key=lambda o: rng.random()):
                    del pool[o["id"]]
                    if rng.random() > pc.p_accept:
                        reject(o)
                        continue
                    view = virt if pc.global_visibility else offered_blind
                    new_lines, new_age = {}, {}
                    cmax = cust_maxage[o["customer"]]
                    for j, q in o["lines"].items():
                        if _feasible_avail(view, by_prod, j, o["maxage"][j]) >= q:
                            new_lines[j], new_age[j] = q, o["maxage"][j]
                            continue
                        fam = self.world.family[j]
                        subs = [j2 for j2 in range(self.nP)
                                if self.world.family[j2] == fam and j2 != j and j2 not in o["lines"]
                                and j2 not in new_lines]
                        best, best_av = None, -1
                        for j2 in subs:
                            av = _feasible_avail(view, by_prod, j2, int(cmax[j2]))
                            if av >= q and av > best_av:
                                best, best_av = j2, av
                        if best is not None:
                            new_lines[best], new_age[best] = q, int(cmax[best])
                            stats["n_subst"] += 1
                        else:
                            stats["n_dropped_lines"] += 1
                    if not new_lines:
                        reject(o)
                        continue
                    # book the modified order in the view so that the same units /
                    # substitute are not offered twice (only with global visibility)
                    if pc.global_visibility:
                        for j, q in new_lines.items():
                            t = _fefo_take(virt, by_prod, j, new_age[j], q)
                            if t:
                                for b, x in t.items():
                                    virt[b] -= x
                    mo = dict(o, lines=new_lines, maxage=new_age)
                    pool[o["id"]] = mo
                    stats["n_mod"] += 1
            for o in pool.values():   # safety: nothing should remain
                reject(o)

        # ---------------- metrics ----------------
        custs = sorted(set(o["customer"] for o in day.orders))
        sl = [t_srv[k] / t_ord[k] for k in custs if t_ord[k] > 0]
        sl_strict = [t_srv_strict[k] / t_ord[k] for k in custs if t_ord[k] > 0]
        by_c = defaultdict(list)
        for o in day.orders:
            by_c[o["customer"]].append(status.get(o["id"]) == "served")
        dD = []
        for k in custs:
            cust_hist[k].append((t_ord[k], t_srv[k]))
            d = sum(x[0] for x in cust_hist[k])
            e = sum(x[1] for x in cust_hist[k])
            if d > 0:
                dD.append(e / d)
        food = [o for o in day.orders if o["channel"] == "FOOD"]
        hor = [o for o in day.orders if o["channel"] == "HORECA"]
        return dict(
            n_orders=len(day.orders),
            fill_rate=np.mean([status.get(o["id"]) == "served" for o in day.orders]),
            fill_food=np.mean([status.get(o["id"]) == "served" for o in food]) if food else np.nan,
            fill_horeca=np.mean([status.get(o["id"]) == "served" for o in hor]) if hor else np.nan,
            avg_sl=float(np.mean(sl)), avg_sl_strict=float(np.mean(sl_strict)),
            avg_sl_D=float(np.mean(dD)),
            min_sl_day=float(np.min(sl)), min_sl_D=float(np.min(dD)),
            sd_sl_D=float(np.std(dD)),
            pct_cust_full=float(np.mean([all(v) for v in by_c.values()])),
            **stats)
