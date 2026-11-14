"""
MILP model for (batch) order promising — Framinan et al. (2024), Section 4.1.2.

    min  alpha * (1/|I|) * sum_i P_i  -  (1 - alpha) * L                       (1)
    s.t. sum_i X_il                    <= s_l                    for all l     (2)
         T_ij                          >= o_ij * (1 - Y_ij)      for all i, j  (3)
         |J| * P_i                     >= sum_{j in J_i} Y_ij    for all i     (4)
         sum_{l in B_ij} X_il           = T_ij                   for all i, j  (5)
         (d_k + sum o_ij) * L - sum X_il <= e_k                  for all k     (6)
         P_i, Y_ij binary;  X_il, T_ij, L >= 0                                 (7)

Two variants are provided:

* variant="paper"     : exactly the formulation printed in the paper.
* variant="corrected" : the group's correction (extension). The printed model has
  no upper bound on T_ij and allows units to be assigned to orders that are NOT
  served, while those units still raise the service level L through (6).
  The corrected variant adds
        T_ij <= o_ij                (no over-allocation)
        T_ij <= o_ij * (1 - P_i)    (no allocation to unserved orders)
  and uses |J_i| instead of |J| in (4) (tighter big-M).

Data structures (plain Python, no external dependencies besides PuLP):

    orders  : list of dict(id, customer, lines={product: qty}, maxage={product: days})
    batches : list of dict(id, product, qty, age)
    hist    : dict customer -> (d_k, e_k) units ordered / served in the last D days
"""
from __future__ import annotations

import time
from collections import defaultdict

import pulp


def feasible_batches(orders, batches):
    """B_ij: batches of product j whose age satisfies the freshness limit of line (i, j)."""
    by_prod = defaultdict(list)
    for b in batches:
        if b["qty"] > 1e-9:
            by_prod[b["product"]].append(b)
    B = {}
    for o in orders:
        for j in o["lines"]:
            lim = o["maxage"].get(j, float("inf"))
            B[(o["id"], j)] = [b["id"] for b in by_prod.get(j, []) if b["age"] <= lim]
    return B


def _is_feasible(prob, tol=1e-4):
    """Primal feasibility check of the point returned by the solver."""
    for v in prob.variables():
        x = v.value()
        if x is None:
            return False
        if v.lowBound is not None and x < v.lowBound - tol:
            return False
        if v.upBound is not None and x > v.upBound + tol:
            return False
    for c in prob.constraints.values():
        r = c.value()
        if r is None:
            return False
        scale = tol * max(1.0, abs(c.constant))
        if c.sense == pulp.LpConstraintLE and r > scale:
            return False
        if c.sense == pulp.LpConstraintGE and r < -scale:
            return False
        if c.sense == pulp.LpConstraintEQ and abs(r) > scale:
            return False
    return True


def solve_order_promising(orders, batches, hist, alpha=0.5, variant="paper",
                          time_limit=600, threads=1, msg=False, weight="orders"):
    """Solve the MILP. Returns dict with served/unserved order ids, allocations,
    objective terms, CPU time and diagnostic quantities.

    weight="orders" : first term = share of orders not served (paper).
    weight="volume" : first term = share of requested UNITS not served (extension:
                      large FOOD orders are no longer worth the same as a 2-line order)."""
    t0 = time.perf_counter()
    if not orders:
        return dict(served=[], unserved=[], alloc={}, L=1.0, cpu=0.0, status="Empty",
                    over_units=0.0, unserved_units=0.0, n_vars=0, n_cons=0)

    stock = {b["id"]: b["qty"] for b in batches}
    B = feasible_batches(orders, batches)
    all_products = {j for o in orders for j in o["lines"]}
    nJ = len(all_products)

    prob = pulp.LpProblem("OrderPromising", pulp.LpMinimize)
    P = {o["id"]: pulp.LpVariable(f"P_{o['id']}", cat="Binary") for o in orders}
    Y, T, X = {}, {}, {}
    for o in orders:
        i = o["id"]
        for j in o["lines"]:
            Y[(i, j)] = pulp.LpVariable(f"Y_{i}_{j}", cat="Binary")
            T[(i, j)] = pulp.LpVariable(f"T_{i}_{j}", lowBound=0)
            for l in B[(i, j)]:
                if (i, l) not in X:
                    X[(i, l)] = pulp.LpVariable(f"X_{i}_{l}", lowBound=0)
    L = pulp.LpVariable("L", lowBound=0)

    # (1) objective
    if weight == "volume":
        vol = {o["id"]: float(sum(o["lines"].values())) for o in orders}
        tot = sum(vol.values())
        prob += alpha * pulp.lpSum((vol[i] / tot) * P[i] for i in P) - (1 - alpha) * L
    else:
        prob += alpha * (1.0 / len(orders)) * pulp.lpSum(P.values()) - (1 - alpha) * L

    # (2) batch capacity
    by_batch = defaultdict(list)
    for (i, l), v in X.items():
        by_batch[l].append(v)
    for l, vs in by_batch.items():
        prob += pulp.lpSum(vs) <= stock[l], f"cap_{l}"

    for o in orders:
        i = o["id"]
        nJi = len(o["lines"])
        bigM = nJi if variant == "corrected" else nJ
        for j, q in o["lines"].items():
            # (3) full quantity of a served line
            prob += T[(i, j)] >= q * (1 - Y[(i, j)]), f"qty_{i}_{j}"
            # (5) freshness: units of j only from feasible batches
            prob += pulp.lpSum(X[(i, l)] for l in B[(i, j)]) == T[(i, j)], f"fresh_{i}_{j}"
            if variant == "corrected":
                prob += T[(i, j)] <= q * (1 - P[i]), f"ub_{i}_{j}"
        # (4) in-full: order served only if every line is served
        prob += bigM * P[i] >= pulp.lpSum(Y[(i, j)] for j in o["lines"]), f"full_{i}"

    # (6) minimum service level over customers present in the instance
    by_cust = defaultdict(list)
    for o in orders:
        by_cust[o["customer"]].append(o)
    for k, ords in by_cust.items():
        d_k, e_k = hist.get(k, (0.0, 0.0))
        new_req = sum(q for o in ords for q in o["lines"].values())
        served_units = pulp.lpSum(X[(o["id"], l)] for o in ords for j in o["lines"]
                                  for l in B[(o["id"], j)])
        if d_k + new_req > 0:
            prob += (d_k + new_req) * L - served_units <= e_k, f"sl_{k}"

    build = time.perf_counter() - t0
    t1 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, threads=threads))
    # CBC 2.10.3 (bundled with PuLP) sometimes reports "Optimal" while returning an
    # infeasible point when its preprocessing fails ("possible tolerance issue").
    # Every solution is therefore checked; if infeasible, the model is re-solved
    # with preprocessing switched off, and as a last resort repaired greedily below.
    resolved = False
    if not _is_feasible(prob):
        resolved = True
        left = max(1.0, time_limit - (time.perf_counter() - t1))
        prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=left, threads=threads,
                                     options=["preprocess off"]))
    feasible = _is_feasible(prob)
    solve = time.perf_counter() - t1
    status = pulp.LpStatus[prob.status]
    hit_limit = solve >= time_limit * 0.98

    def val(v):
        x = v.value()
        return 0.0 if x is None else max(0.0, x)

    served, unserved = [], []
    for o in orders:
        (served if val(P[o["id"]]) < 0.5 else unserved).append(o["id"])
    repaired = 0
    if not feasible:  # last resort: keep only orders whose lines fit, in solver order
        rem = dict(stock)
        keep = []
        for i in served:
            o = next(x for x in orders if x["id"] == i)
            take, ok = {}, True
            for j, q in o["lines"].items():
                need = q
                for l in B[(i, j)]:
                    t = min(rem[l] - take.get(l, 0), need)
                    if t > 0:
                        take[l] = take.get(l, 0) + t
                        need -= t
                if need > 1e-6:
                    ok = False
                    break
            if ok:
                for l, t in take.items():
                    rem[l] -= t
                keep.append(i)
        repaired = len(served) - len(keep)
        unserved += [i for i in served if i not in keep]
        served = keep
        for (i, l), v in X.items():   # rebuild X consistently with the repaired plan
            v.varValue = 0.0
        rem = dict(stock)
        for i in served:
            o = next(x for x in orders if x["id"] == i)
            for j, q in o["lines"].items():
                need = q
                for l in B[(i, j)]:
                    t = min(rem[l], need)
                    if t > 0:
                        X[(i, l)].varValue = t
                        rem[l] -= t
                        need -= t

    # Allocation actually shipped: exactly o_ij for served lines (FEFO over the X values)
    alloc = {}
    over_units = 0.0
    unserved_units = 0.0
    served_set = set(served)
    batch_age = {b["id"]: b["age"] for b in batches}
    for o in orders:
        i = o["id"]
        for j, q in o["lines"].items():
            xs = [(l, val(X[(i, l)])) for l in B[(i, j)]]
            tot = sum(x for _, x in xs)
            if i in served_set:
                over_units += max(0.0, tot - q)
                need = q
                for l, x in sorted(xs, key=lambda t: -batch_age[t[0]]):
                    take = min(x, need)
                    if take > 1e-9:
                        alloc[(i, l)] = alloc.get((i, l), 0.0) + take
                        need -= take
                    if need <= 1e-9:
                        break
                if need > 1e-6:  # numerical safety: should not happen
                    over_units -= need
            else:
                unserved_units += tot

    return dict(served=served, unserved=unserved, alloc=alloc, L=val(L),
                obj=pulp.value(prob.objective), cpu=solve, build=build,
                status=status, hit_limit=hit_limit, resolved=resolved,
                feasible=feasible, repaired=repaired,
                over_units=over_units, unserved_units=unserved_units,
                n_vars=len(prob.variables()), n_cons=len(prob.constraints))


def true_min_service_level(orders, served, hist):
    """Min service level over customers in the instance, computed from what is
    really shipped (o_ij of served orders only) — used to audit the model's L."""
    served = set(served)
    by_cust = defaultdict(lambda: [0.0, 0.0])
    for o in orders:
        req = sum(o["lines"].values())
        by_cust[o["customer"]][0] += req
        if o["id"] in served:
            by_cust[o["customer"]][1] += req
    vals = []
    for k, (req, srv) in by_cust.items():
        d_k, e_k = hist.get(k, (0.0, 0.0))
        if d_k + req > 0:
            vals.append((e_k + srv) / (d_k + req))
    return min(vals) if vals else 1.0
