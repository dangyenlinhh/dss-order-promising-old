"""
Correctness checks. Run:  python tests/test_all.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from op.model import solve_order_promising, true_min_service_level  # noqa: E402
from op.sim import PolicyConfig, Simulator, World, WorldConfig  # noqa: E402
from op.testbed import generate_instance  # noqa: E402


def check_solution(orders, batches, r):
    """Shipped allocation respects capacity, freshness and in-full delivery."""
    cap = {b["id"]: b["qty"] for b in batches}
    age = {b["id"]: b["age"] for b in batches}
    prod = {b["id"]: b["product"] for b in batches}
    used = defaultdict(float)
    per_line = defaultdict(float)
    od = {o["id"]: o for o in orders}
    for (i, l), q in r["alloc"].items():
        assert i in r["served"], "allocation to an unserved order was shipped"
        j = prod[l]
        assert j in od[i]["lines"], "product not in order"
        assert age[l] <= od[i]["maxage"][j], "freshness violated"
        used[l] += q
        per_line[(i, j)] += q
    for l, q in used.items():
        assert q <= cap[l] + 1e-6, "capacity violated"
    for i in r["served"]:
        for j, q in od[i]["lines"].items():
            assert abs(per_line[(i, j)] - q) <= 1e-4 * max(1, q), "order not shipped in full"


def test_toy_optimum():
    # 1 product, 10 units. Order A wants 10 (customer 0), orders B and C want 5 each.
    # Best for "number of orders served": serve B and C.
    orders = [dict(id=0, customer=0, lines={0: 10}, maxage={0: 99}),
              dict(id=1, customer=1, lines={0: 5}, maxage={0: 99}),
              dict(id=2, customer=2, lines={0: 5}, maxage={0: 99})]
    batches = [dict(id=0, product=0, qty=10, age=1)]
    hist = {0: (100, 100), 1: (100, 100), 2: (100, 100)}
    for v in ("paper", "corrected"):
        r = solve_order_promising(orders, batches, hist, alpha=0.9, variant=v)
        assert sorted(r["served"]) == [1, 2], (v, r["served"])
        check_solution(orders, batches, r)


def test_freshness():
    # Order 0 only accepts batches up to 10 days old; the only fresh batch must go to it.
    orders = [dict(id=0, customer=0, lines={0: 5}, maxage={0: 10}),
              dict(id=1, customer=1, lines={0: 5}, maxage={0: 99})]
    batches = [dict(id=0, product=0, qty=5, age=5), dict(id=1, product=0, qty=5, age=30)]
    for v in ("paper", "corrected"):
        r = solve_order_promising(orders, batches, {}, alpha=0.5, variant=v)
        assert sorted(r["served"]) == [0, 1], (v, r["served"])
        check_solution(orders, batches, r)


def test_in_full():
    # Order needs products 0 and 1; product 1 is missing -> cannot be served.
    orders = [dict(id=0, customer=0, lines={0: 5, 1: 5}, maxage={0: 99, 1: 99})]
    batches = [dict(id=0, product=0, qty=50, age=1)]
    r = solve_order_promising(orders, batches, {}, alpha=0.5, variant="corrected")
    assert r["served"] == []


def test_paper_model_flaw():
    # The printed model can raise L by assigning units to an order it does not serve.
    orders = [dict(id=0, customer=0, lines={0: 10, 1: 10}, maxage={0: 99, 1: 99})]
    batches = [dict(id=0, product=0, qty=10, age=1)]          # product 1 missing
    hist = {0: (10, 0)}
    rp = solve_order_promising(orders, batches, hist, alpha=0.5, variant="paper")
    rc = solve_order_promising(orders, batches, hist, alpha=0.5, variant="corrected")
    assert rp["served"] == [] and rc["served"] == []
    assert rp["L"] > 0.3 and abs(rc["L"]) < 1e-6        # model L inflated (0.33) vs real 0
    assert abs(true_min_service_level(orders, [], hist)) < 1e-9


def test_cbc_preprocessing_bug():
    # Instance captured from the simulation: CBC 2.10.3 reports "Optimal" but returns
    # an infeasible point (negative X, capacity violated). The wrapper must detect it.
    import pickle
    orders, batches, hist, kw = pickle.load(open(os.path.join(os.path.dirname(__file__), "data",
                                                              "cbc_preprocess_case.pkl"), "rb"))
    r = solve_order_promising(orders, batches, hist, **kw)
    assert r["feasible"]
    check_solution(orders, batches, r)


def test_random_instances():
    for seed in range(6):
        o, b, h = generate_instance(20, 20, "MOZ", "LH", "HV", "MSC", "MFC", seed=seed)
        for v in ("paper", "corrected"):
            r = solve_order_promising(o, b, h, alpha=0.5, variant=v, time_limit=10)
            check_solution(o, b, r)
            if v == "corrected":
                assert abs(r["L"] - true_min_service_level(o, r["served"], h)) < 1e-4


def test_simulator_invariants():
    w = World(WorldConfig(), seed=7)
    hist = [w.make_day() for _ in range(10)]
    days = [w.make_day() for _ in range(3)]
    sim = Simulator(w, days, hist)
    for name in ("FCFS", "GO", "RE-NOCM", "RE"):
        rows = sim.run(PolicyConfig(name=name, iterations=4))
        for r in rows:
            assert 0 <= r["fill_rate"] <= 1 and 0 <= r["avg_sl"] <= 1 + 1e-9
            if name != "RE":
                assert r["n_mod"] == 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"PASS {name}")
