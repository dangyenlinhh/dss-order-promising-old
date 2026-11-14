"""
Artificial testbed generator following Section 5.1 of Framinan et al. (2024).

Factors (paper notation):
    alpha                 weight in the objective function
    n_orders  |I|         number of orders
    n_products |J|        number of products
    order_size            SOZ U[1,5] lines, MOZ U[5,10], BOZ U[10,30]
    homogeneity           HH qty per line U[1,10], LH U[1,1000]
    sl_var                LV service level U[95,100]%, HV U[60,100]%
    stock_cov             FSC stock 100-200% of demand, HSC 80-100%, MSC 50-80%
    fresh_cov             FFC all stock usable by any order, HFC 80-100%, MFC 50-80%

Instances are built in two steps, as in the paper: first the demand (orders),
then the supply (batches) according to stock and freshness coverage.
Each customer issues a single order (assumption stated in the paper).

Interpretation choices (the paper is not fully explicit, documented in README):
* Freshness requirement of every order line: max age U[10, 60] days.
* The part of the stock NOT usable by every order is split in batches whose age
  equals the requirement of one specific order line; such a batch is usable only
  by lines at least as lenient. The paper says lines are visited "starting from
  the most stringent", but a batch matching the most stringent requirement is
  usable by everybody, so lines are visited in random order here (otherwise the
  restricted stock would not restrict anything).
* Historic data of customer k: d_k ~ U[50, 500] units, e_k = d_k * SL_k.
"""
from __future__ import annotations

import numpy as np

ORDER_SIZE = {"SOZ": (1, 5), "MOZ": (5, 10), "BOZ": (10, 30)}
HOMOG = {"HH": (1, 10), "LH": (1, 1000)}
SL_VAR = {"LV": (0.95, 1.00), "HV": (0.60, 1.00)}
STOCK_COV = {"FSC": (1.0, 2.0), "HSC": (0.8, 1.0), "MSC": (0.5, 0.8)}
FRESH_COV = {"FFC": (1.0, 1.0), "HFC": (0.8, 1.0), "MFC": (0.5, 0.8)}


def generate_instance(n_orders, n_products, order_size, homogeneity, sl_var,
                      stock_cov, fresh_cov, seed=0):
    rng = np.random.default_rng(seed)

    # ---- demand -----------------------------------------------------------
    lo, hi = ORDER_SIZE[order_size]
    qlo, qhi = HOMOG[homogeneity]
    orders = []
    for i in range(n_orders):
        n_lines = int(min(rng.integers(lo, hi + 1), n_products))
        prods = rng.choice(n_products, size=n_lines, replace=False)
        lines = {int(j): int(rng.integers(qlo, qhi + 1)) for j in prods}
        maxage = {int(j): int(rng.integers(10, 61)) for j in prods}
        orders.append(dict(id=i, customer=i, lines=lines, maxage=maxage))

    slo, shi = SL_VAR[sl_var]
    hist = {}
    for o in orders:
        d = float(rng.uniform(50, 500))
        hist[o["customer"]] = (d, d * float(rng.uniform(slo, shi)))

    # ---- supply -----------------------------------------------------------
    demand = np.zeros(n_products)
    lines_by_prod = {j: [] for j in range(n_products)}
    for o in orders:
        for j, q in o["lines"].items():
            demand[j] += q
            lines_by_prod[j].append((o["maxage"][j], q))

    clo, chi = STOCK_COV[stock_cov]
    flo, fhi = FRESH_COV[fresh_cov]
    batches = []
    bid = 0
    for j in range(n_products):
        if demand[j] == 0:
            continue
        S = int(round(demand[j] * rng.uniform(clo, chi)))
        if S <= 0:
            continue
        f = rng.uniform(flo, fhi)
        free = int(round(S * f))
        if free > 0:  # usable by every order (age 0)
            batches.append(dict(id=bid, product=j, qty=free, age=0)); bid += 1
        rest = S - free
        # restricted part: batches matching specific line requirements
        lst = list(lines_by_prod[j])
        rng.shuffle(lst)
        for age, q in lst:
            if rest <= 0:
                break
            take = min(rest, q)
            batches.append(dict(id=bid, product=j, qty=take, age=age)); bid += 1
            rest -= take
        if rest > 0:  # leftover goes to the most lenient requirement
            age = max(a for a, _ in lines_by_prod[j])
            batches.append(dict(id=bid, product=j, qty=rest, age=age)); bid += 1
    return orders, batches, hist
