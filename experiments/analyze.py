"""
Analysis of all experiments: ANOVA tables, summary tables (CSV + Markdown) and
figures (PNG, 200 dpi) ready to be inserted in the report.

Usage: python experiments/analyze.py
Reads results/*.csv, writes results/tables/*.csv, results/figures/*.png and
results/SUMMARY.md
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---- chart style: reference categorical palette, fixed order, recessive axes ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLICY_ORDER = ["FCFS", "GO", "RE-NOCM", "RE"]
POLICY_COLOR = dict(zip(POLICY_ORDER, SERIES))
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
MD = []


def md(s=""):
    MD.append(s)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def md_table(df, floatfmt=3):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.{floatfmt}f}" if abs(v) < 1e5 else f"{v:.3g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def anova(df, resp, factors):
    f = f"{resp} ~ " + " + ".join(f"C({x}, Sum)" for x in factors)
    m = smf.ols(f, data=df).fit()
    a = anova_lm(m, typ=3).drop(index="Intercept")
    a.index = [i.replace("C(", "").replace(", Sum)", "") for i in a.index]
    a = a.rename(columns={"sum_sq": "SS", "df": "df", "F": "F", "PR(>F)": "Sig."})
    a["Mean Sq."] = a["SS"] / a["df"]
    a = a[["SS", "df", "Mean Sq.", "F", "Sig."]]
    return a


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# =============================================================================
# Experiment 1 — testbed
# =============================================================================
def exp1():
    p = os.path.join(RES, "exp1_screening.csv")
    if not os.path.exists(p):
        return
    d_all = pd.read_csv(p)
    d = d_all[d_all.variant == "paper"]
    md("## Experiment 1 — Testbed (replication of Section 5.1)")
    md(f"Screening design: 8 factors x 2 levels, {d.rep.nunique()} replicates, "
       f"{len(d)} instances solved; CPU limit {d.cpu.max():.0f}s "
       f"({int(d.hit_limit.sum())} instances reached it). "
       f"CBC returned an infeasible point and the model was re-solved without "
       f"preprocessing in {int(d.resolved.sum())} instances ({d.resolved.mean():.1%}); "
       f"greedy repair needed in {int((d.repaired > 0).sum())}.")
    factors = ["alpha", "n_orders", "n_products", "order_size", "homog", "sl_var",
               "stock_cov", "fresh_cov"]
    names = {"alpha": "alpha", "n_orders": "Number of orders", "n_products": "No. of products",
             "order_size": "Order size", "homog": "Homog. qty", "sl_var": "Var. service level",
             "stock_cov": "Stock coverage", "fresh_cov": "Freshness coverage"}
    paper_sig = {  # Table 1 of the paper: significant (p<0.001)?
        "cpu": dict(alpha=0, n_orders=1, n_products=1, order_size=1, homog=1, sl_var=0, stock_cov=1, fresh_cov=1),
        "fill_rate": dict(alpha=0, n_orders=1, n_products=1, order_size=1, homog=1, sl_var=0, stock_cov=1, fresh_cov=0),
        "min_sl_model": dict(alpha=0, n_orders=1, n_products=1, order_size=1, homog=1, sl_var=0, stock_cov=1, fresh_cov=0),
    }
    out = []
    for resp, lab in [("cpu", "CPU time"), ("fill_rate", "Fill rate"),
                      ("min_sl_model", "Min. service level (model L)"),
                      ("min_sl_true", "Min. service level (true)")]:
        a = anova(d, resp, factors)
        a.insert(0, "Response", lab)
        a.insert(1, "Factor", [names.get(i, i) for i in a.index])
        ps = paper_sig.get(resp if resp != "min_sl_true" else "min_sl_model")
        a["Significant (ours)"] = np.where(a["Sig."] < 0.001, "yes", "no")
        a["Significant (paper)"] = [("yes" if ps[i] else "no") if i in ps else "" for i in a.index]
        a.loc[a.index == "Residual", "Significant (ours)"] = ""
        out.append(a)
    t1 = pd.concat(out).reset_index(drop=True)
    t1.to_csv(os.path.join(TAB, "T1_anova_screening.csv"), index=False)
    md("\n**Table 1 (replication). Multi-factor ANOVA, two levels per factor.**\n")
    show = t1.copy()
    show["Sig."] = show["Sig."].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    md(md_table(show[["Response", "Factor", "F", "Sig.", "Significant (ours)", "Significant (paper)"]], 1))
    m = (t1.Response != "Min. service level (true)") & (t1.Factor != "Residual")
    agree = (t1["Significant (ours)"] == t1["Significant (paper)"])[m]
    md(f"\nAgreement with the paper on significance: {agree.sum()}/{len(agree)} factor-response pairs.")

    # main effects table
    me = []
    for f in factors:
        g = d.groupby(f)[["cpu", "fill_rate", "min_sl_model", "min_sl_true"]].mean()
        for lv, r in g.iterrows():
            me.append(dict(Factor=names[f], Level=lv, **r.round(3).to_dict()))
    me = pd.DataFrame(me)
    me.to_csv(os.path.join(TAB, "T1b_main_effects.csv"), index=False)
    md("\n**Main effects (means by level).**\n")
    md(md_table(me, 3))

    # audit of the printed model (paper variant): L reported vs L realised
    md("\n**Audit of the MILP as printed in the paper (all screening instances).**\n")
    au = pd.DataFrame([dict(
        Instances=len(d),
        **{"Share with units assigned to unserved orders": (d.unserved_units > 0.5).mean(),
           "Share with over-allocation (T_ij > o_ij)": (d.over_units > 0.5).mean(),
           "Mean model L": d.min_sl_model.mean(),
           "Mean true min SL": d.min_sl_true.mean(),
           "Share with L > 1": (d.min_sl_model > 1.0001).mean(),
           "Mean |L - true L|": (d.min_sl_model - d.min_sl_true).abs().mean()})])
    au.to_csv(os.path.join(TAB, "T1c_audit_paper_model.csv"), index=False)
    md(md_table(au, 3))

    # same screening with the corrected MILP (1 replicate): is alpha still relevant?
    dc = d_all[d_all.variant == "corrected"]
    if len(dc) > 50:
        md(f"\n**Screening with the corrected MILP ({len(dc)} instances, replicate 0).**\n")
        rows = []
        dp0 = d[d.rep == 0]
        for resp in ["cpu", "fill_rate", "min_sl_true"]:
            a = anova(dc, resp, factors)
            ap = anova(dp0, resp, factors)
            for f_ in factors:
                rows.append(dict(Response=resp, Factor=names[f_],
                                 **{"F corrected": a.loc[f_, "F"], "Sig. corrected": a.loc[f_, "Sig."],
                                    "F printed (rep 0)": ap.loc[f_, "F"], "Sig. printed (rep 0)": ap.loc[f_, "Sig."]}))
        tc = pd.DataFrame(rows)
        key_ = {"cpu": "cpu", "fill_rate": "fill_rate", "min_sl_true": "min_sl_model"}
        inv = {v: k for k, v in names.items()}
        tc["Significant (paper)"] = [("yes" if paper_sig[key_[r.Response]][inv[r.Factor]] else "no")
                                     for r in tc.itertuples()]
        tc["Agree corrected"] = (np.where(tc["Sig. corrected"] < 0.001, "yes", "no") == tc["Significant (paper)"])
        tc["Agree printed"] = (np.where(tc["Sig. printed (rep 0)"] < 0.001, "yes", "no") == tc["Significant (paper)"])
        tc.to_csv(os.path.join(TAB, "T1d_anova_corrected_vs_printed.csv"), index=False)
        md(md_table(tc, 3))
        md(f"\nAgreement with the paper's Table 1 (same 256 instances): corrected MILP "
           f"{int(tc['Agree corrected'].sum())}/24, printed MILP {int(tc['Agree printed'].sum())}/24.")
        g = pd.concat([dp0.assign(model="printed"), dc.assign(model="corrected")]) \
            .groupby(["model", "alpha"])[["fill_rate", "min_sl_true", "min_sl_model", "cpu"]].mean().round(3)
        g.to_csv(os.path.join(TAB, "T1e_alpha_effect_by_model.csv"))
        md("\n**Effect of alpha by model variant (replicate 0).**\n")
        md(md_table(g.reset_index(), 3))


def exp1_grid():
    p = os.path.join(RES, "exp1_grid.csv")
    if not os.path.exists(p):
        return
    d_all = pd.read_csv(p)
    md("\n## Experiment 1b — CPU grid (replication of Table 2, Figs. 3-5)")
    if (d_all.variant == "corrected").sum() >= 150:
        cmp_ = d_all.groupby(["variant", "fresh_cov"]).agg(cpu=("cpu", "mean"), hit=("hit_limit", "mean"),
                                                            fill=("fill_rate", "mean")).round(3)
        cmp_.to_csv(os.path.join(TAB, "T2c_grid_by_variant.csv"))
        md("\n**CPU grid: printed vs corrected MILP, by freshness coverage.**\n")
        md(md_table(cmp_.reset_index(), 3))
        d = d_all[d_all.variant == "corrected"]
        md("\nTables/figures below use the corrected MILP.")
    else:
        d = d_all[d_all.variant == "paper"]
    md(f"{len(d)} instances, {d.rep.nunique()} replicates per cell, limit {d.cpu.max():.0f}s, "
       f"{int(d.hit_limit.sum())} hit the limit.")
    t2 = d.pivot_table(index=["n_orders", "n_products", "order_size", "fresh_cov"],
                       columns="stock_cov", values="cpu", aggfunc="mean").round(2)
    lim = d.pivot_table(index=["n_orders", "n_products", "order_size", "fresh_cov"],
                        columns="stock_cov", values="hit_limit", aggfunc="max")
    t2.to_csv(os.path.join(TAB, "T2_cpu_grid.csv"))
    lim.to_csv(os.path.join(TAB, "T2b_cpu_grid_hit_limit.csv"))
    md("\nFull table in tables/T2_cpu_grid.csv.")
    # Table 3: ANOVA fill rate
    a = anova(d, "fill_rate", ["n_orders", "n_products", "order_size", "stock_cov", "fresh_cov"])
    a.to_csv(os.path.join(TAB, "T3_anova_fillrate_grid.csv"))
    show = a.reset_index().rename(columns={"index": "Factor"})
    show["Sig."] = show["Sig."].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    md("\n**Table 3 (replication). ANOVA on fill rate, second testbed.**\n")
    md(md_table(show[["Factor", "df", "F", "Sig."]], 1))

    sizes = ["SOZ", "MOZ", "BOZ"]
    # Fig 3: CPU vs number of orders by order size
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for c, s in zip(SERIES, sizes):
        g = d[d.order_size == s].groupby("n_orders").cpu.mean()
        ax.plot(g.index, g.values, color=c, lw=2, marker="o", ms=5, label=s)
    ax.set_xlabel("Number of orders"); ax.set_ylabel("Mean CPU time (s)")
    ax.set_title("CPU time vs number of orders"); ax.legend(title="Order size")
    save(fig, "fig_exp1_cpu_orders.png")
    # Fig 4: CPU vs freshness coverage
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fc = ["MFC", "HFC", "FFC"]
    for c, s in zip(SERIES, sizes):
        g = d[d.order_size == s].groupby("fresh_cov").cpu.mean().reindex(fc)
        ax.plot(fc, g.values, color=c, lw=2, marker="o", ms=5, label=s)
    ax.set_xlabel("Freshness coverage"); ax.set_ylabel("Mean CPU time (s)")
    ax.set_title("CPU time vs freshness coverage"); ax.legend(title="Order size")
    save(fig, "fig_exp1_cpu_freshness.png")
    # Fig 5: fill rate vs stock coverage by order size
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(2)
    w = 0.26
    for n, (c, s) in enumerate(zip(SERIES, sizes)):
        g = d[d.order_size == s].groupby("stock_cov").fill_rate.mean().reindex(["MSC", "FSC"])
        b = ax.bar(x + (n - 1) * w, g.values, w - 0.02, color=c, label=s)
        ax.bar_label(b, fmt="%.2f", fontsize=7, color=INK2, padding=2)
    ax.set_xticks(x, ["Medium (MSC)", "Full (FSC)"]); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Stock coverage"); ax.set_ylabel("Fill rate"); ax.set_title("Fill rate vs stock coverage")
    ax.legend(title="Order size", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_exp1_fill_stock.png")
    md("\nFigures: fig_exp1_cpu_orders.png, fig_exp1_cpu_freshness.png, fig_exp1_fill_stock.png")


# =============================================================================
# Simulation experiments
# =============================================================================
METRICS = [("avg_sl_D", "Avg. customer service level (5-day)"),
           ("fill_rate", "% orders served in full"),
           ("pct_cust_full", "% customers served in full")]


def exp2():
    p = os.path.join(RES, "sim_exp2.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    md("\n## Experiment 2 — Scenario comparison (replication of Section 5.2, Figs. 6-8)")
    md(f"{d.seed.nunique()} synthetic companies x {d.day.nunique()} days, "
       f"{d.n_orders.mean():.0f} orders/day on average, "
       f"{d.scarce.mean():.0%} scarcity days.")
    cols = ["avg_sl_D", "avg_sl", "fill_rate", "pct_cust_full", "fill_food", "fill_horeca",
            "min_sl_D", "sd_sl_D", "cpu", "n_hit_limit", "n_resolved", "n_repaired", "n_mod", "n_subst"]
    t = d.groupby(["T", "policy"])[cols].mean().reindex(POLICY_ORDER, level=1).round(3)
    t.to_csv(os.path.join(TAB, "T4_scenarios.csv"))
    md("\n**Mean indicators by scenario and number of iterations.**\n")
    md(md_table(t.reset_index(), 3))
    ts = d.groupby(["scarce", "policy"])[["avg_sl_D", "fill_rate", "pct_cust_full"]].mean() \
        .reindex(POLICY_ORDER, level=1).round(3)
    ts.to_csv(os.path.join(TAB, "T4b_scenarios_by_scarcity.csv"))
    md("\n**By type of day (scarce = 1).**\n")
    md(md_table(ts.reset_index(), 3))

    # paired differences with bootstrap CI (day x seed is the unit)
    rows = []
    key = ["seed", "day", "T"]
    piv = d.pivot_table(index=key, columns="policy", values="fill_rate")
    pivs = d.pivot_table(index=key, columns="policy", values="avg_sl_D")
    for a, b in [("RE", "FCFS"), ("RE-NOCM", "FCFS"), ("GO", "FCFS"), ("RE", "GO"),
                 ("GO", "RE-NOCM"), ("RE", "RE-NOCM")]:
        for lab, pv in [("fill_rate", piv), ("avg_sl_D", pivs)]:
            m, lo, hi = boot_ci((pv[a] - pv[b]).values)
            wins = ((pv[a] - pv[b]) > 1e-9).mean()
            rows.append(dict(Comparison=f"{a} - {b}", Metric=lab, Mean=m, CI_low=lo, CI_high=hi,
                             **{"Share of days a > b": wins}))
    pd_ = pd.DataFrame(rows)
    pd_.to_csv(os.path.join(TAB, "T5_paired_differences.csv"), index=False)
    md("\n**Paired differences (bootstrap 95% CI over company-days).**\n")
    md(md_table(pd_, 3))

    # Figs 6-8
    panels = [("avg_sl_D", "Avg. customer service level (5-day)", d),
              ("fill_rate", "% orders served in full (all days)", d),
              ("fill_rate", "% orders served in full (scarcity days)", d[d.scarce == 1])]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, (m, lab, dd) in zip(axs, panels):
        x = np.arange(len(POLICY_ORDER))
        for n, T in enumerate([4, 5]):
            g = dd[dd["T"] == T].groupby("policy")[m].mean().reindex(POLICY_ORDER)
            b = ax.bar(x + (n - 0.5) * 0.4, g.values, 0.37,
                       color=SERIES[n], label=f"{T} iterations/day")
            ax.bar_label(b, fmt="%.2f", fontsize=6.5, color=INK2, padding=1)
        ax.set_xticks(x, POLICY_ORDER)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=8)
    save(fig, "fig_exp2_scenarios.png")
    md("\nFigure: fig_exp2_scenarios.png")


def exp3a():
    p = os.path.join(RES, "sim_3a.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    md("\n## Experiment 3a — MILP as printed vs corrected MILP (inside the DSS)")
    t = d.groupby(["cfg_variant", "policy"])[["fill_rate", "avg_sl_D", "pct_cust_full", "min_sl_D",
                                               "over_units", "unserved_units", "cpu"]].mean() \
        .reindex(["GO", "RE-NOCM", "RE"], level=1).round(3)
    t.to_csv(os.path.join(TAB, "T6_variant_sim.csv"))
    md(md_table(t.reset_index(), 3))
    fig, ax = plt.subplots(figsize=(5, 3.0))
    x = np.arange(3)
    for n, v in enumerate(["paper", "corrected"]):
        g = d[d.cfg_variant == v].groupby("policy").fill_rate.mean().reindex(["GO", "RE-NOCM", "RE"])
        b = ax.bar(x + (n - 0.5) * 0.38, g.values, 0.36, color=SERIES[n],
                   label="MILP as printed" if v == "paper" else "Corrected MILP")
        ax.bar_label(b, fmt="%.3f", fontsize=7, color=INK2, padding=1)
    ax.set_xticks(x, ["GO", "RE-NOCM", "RE"]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("% orders served in full"); ax.set_title("Effect of the MILP correction")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_exp3a_variant_sim.png")


def exp3a_testbed():
    p = os.path.join(RES, "exp3a_testbed.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    md("\n## Experiment 3a (testbed) — printed vs corrected MILP on identical instances")
    k = ["n_orders", "n_products", "order_size", "homog", "sl_var", "stock_cov", "fresh_cov", "rep", "alpha"]
    w = d.pivot_table(index=k, columns="variant", values=["fill_rate", "min_sl_true", "min_sl_model", "cpu"])
    s = pd.DataFrame(dict(
        metric=["fill_rate", "min_sl_true", "min_sl_model", "cpu"],
        printed=[w[(m, "paper")].mean() for m in ["fill_rate", "min_sl_true", "min_sl_model", "cpu"]],
        corrected=[w[(m, "corrected")].mean() for m in ["fill_rate", "min_sl_true", "min_sl_model", "cpu"]],
    ))
    s["share corrected better"] = [
        ((w[(m, "corrected")] - w[(m, "paper")]) > 1e-6).mean() for m in s.metric]
    s["share printed better"] = [
        ((w[(m, "paper")] - w[(m, "corrected")]) > 1e-6).mean() for m in s.metric]
    s.to_csv(os.path.join(TAB, "T6b_variant_testbed.csv"), index=False)
    md(f"{len(w)} paired instances.\n")
    md(md_table(s, 3))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labs = ["Fill rate", "True min.\nservice level", "Min. service level\nreported by model (L)"]
    ms = ["fill_rate", "min_sl_true", "min_sl_model"]
    x = np.arange(3)
    for n, v in enumerate(["paper", "corrected"]):
        vals = [w[(m, v)].mean() for m in ms]
        b = ax.bar(x + (n - 0.5) * 0.38, vals, 0.36, color=SERIES[n],
                   label="MILP as printed" if v == "paper" else "Corrected MILP")
        ax.bar_label(b, fmt="%.2f", fontsize=7.5, color=INK2, padding=1)
    ax.set_xticks(x, labs, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("Printed vs corrected MILP (128 paired testbed instances)", pad=8)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_exp3a_variant.png")


def exp3_sweeps():
    # 3b SATP
    p = os.path.join(RES, "sim_3b.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        md("\n## Experiment 3b — SATP variants (future research line of the paper)")
        d["satp_cfg"] = np.where(d.cfg_satp == "none", "none", d.cfg_satp + " b=" + d.cfg_beta.astype(str))
        t = d.groupby(["policy", "cfg_satp", "cfg_beta"])[["fill_rate", "avg_sl_D", "pct_cust_full", "min_sl_D",
                                                           "sd_sl_D"]].mean().round(4)
        t.to_csv(os.path.join(TAB, "T7_satp.csv"))
        md(md_table(t.reset_index(), 4))
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0))
        for ax, pol in zip(axs, ["RE-NOCM", "RE"]):
            g = d[d.policy == pol]
            none = g[g.cfg_satp == "none"].fill_rate.mean()
            ax.axhline(none, color=INK2, lw=1, ls="--", label="no SATP")
            for c, mode in zip(SERIES, ["common", "product"]):
                s = g[g.cfg_satp == mode].groupby("cfg_beta").fill_rate.mean()
                ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=5,
                        label="common p(t) (paper)" if mode == "common" else "product p_j(t)")
            ax.set_xlabel("Reserve multiplier beta"); ax.set_title(pol)
            ax.set_ylabel("% orders served in full")
        axs[0].legend(fontsize=7)
        save(fig, "fig_exp3b_satp.png")
    # 3c iterations
    p = os.path.join(RES, "sim_3c.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        md("\n## Experiment 3c — Number of iterations per day")
        t = d.groupby(["policy", "T"])[["fill_rate", "avg_sl_D", "pct_cust_full", "n_mod", "cpu"]].mean().round(4)
        t.to_csv(os.path.join(TAB, "T8_iterations.csv"))
        md(md_table(t.reset_index(), 4))
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        for c, pol in zip([SERIES[2], SERIES[3]], ["RE-NOCM", "RE"]):
            s = d[d.policy == pol].groupby("T").fill_rate.mean()
            ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=5, label=pol)
        ax.set_xlabel("Iterations per day (T)"); ax.set_ylabel("% orders served in full")
        ax.set_title("Effect of the number of iterations"); ax.legend()
        save(fig, "fig_exp3c_iterations.png")
    # 3d alpha
    p = os.path.join(RES, "sim_3d.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        md("\n## Experiment 3d — Weight alpha")
        t = d.groupby(["policy", "cfg_alpha"])[["fill_rate", "avg_sl_D", "min_sl_D", "sd_sl_D",
                                                "fill_food", "fill_horeca"]].mean().round(4)
        t.to_csv(os.path.join(TAB, "T9_alpha.csv"))
        md(md_table(t.reset_index(), 4))
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0))
        for ax, (m, lab) in zip(axs, [("fill_rate", "% orders served in full"),
                                      ("min_sl_D", "Min. customer service level (5-day)")]):
            for c, pol in zip([SERIES[1], SERIES[3]], ["GO", "RE"]):
                s = d[d.policy == pol].groupby("cfg_alpha")[m].mean()
                ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=5, label=pol)
            ax.set_xlabel("alpha"); ax.set_title(lab)
        axs[0].legend()
        save(fig, "fig_exp3d_alpha.png")
    # 3e OCM
    p = os.path.join(RES, "sim_3e.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        md("\n## Experiment 3e — OCM: acceptance probability and global visibility")
        t = d.groupby(["cfg_global_visibility", "cfg_p_accept"])[["fill_rate", "avg_sl_D", "pct_cust_full",
                                                                  "n_mod", "n_subst"]].mean().round(4)
        t.to_csv(os.path.join(TAB, "T10_ocm.csv"))
        md(md_table(t.reset_index(), 4))
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        for c, gv in zip(SERIES, [True, False]):
            s = d[d.cfg_global_visibility == gv].groupby("cfg_p_accept").fill_rate.mean()
            ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=5,
                    label="global visibility (DSS)" if gv else "no global visibility")
        ax.set_xlabel("Probability a customer accepts a modification")
        ax.set_ylabel("% orders served in full"); ax.set_title("Value of OCM and of stock visibility")
        ax.legend()
        save(fig, "fig_exp3e_ocm.png")


def exp3f():
    p = os.path.join(RES, "sim_3f.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    md("\n## Experiment 3f — Objective weighted by orders (paper) vs by volume")
    t = d.groupby(["policy", "cfg_weight"])[["fill_rate", "avg_sl_D", "avg_sl", "fill_food", "fill_horeca",
                                             "min_sl_D", "sd_sl_D"]].mean().round(4)
    t.to_csv(os.path.join(TAB, "T11_weight.csv"))
    md(md_table(t.reset_index(), 4))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(4)
    labs = ["GO / orders", "GO / volume", "RE / orders", "RE / volume"]
    keys = [("GO", "orders"), ("GO", "volume"), ("RE", "orders"), ("RE", "volume")]
    for n, (m, lab) in enumerate([("fill_horeca", "HORECA"), ("fill_food", "FOOD")]):
        v = [d[(d.policy == p_) & (d.cfg_weight == w_)][m].mean() for p_, w_ in keys]
        b = ax.bar(x + (n - 0.5) * 0.38, v, 0.36, color=SERIES[n], label=lab)
        ax.bar_label(b, fmt="%.2f", fontsize=7, color=INK2, padding=1)
    ax.set_xticks(x, labs, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("% orders served in full"); ax.set_title("Fill rate by channel and objective weighting")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig_exp3f_weight.png")


if __name__ == "__main__":
    md("# Results summary (auto-generated by experiments/analyze.py)\n")
    exp1()
    exp1_grid()
    exp2()
    exp3a()
    exp3a_testbed()
    exp3_sweeps()
    exp3f()
    with open(os.path.join(RES, "SUMMARY.md"), "w") as f:
        f.write("\n".join(MD) + "\n")
    print("\n".join(MD))
