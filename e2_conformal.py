"""E2: calibrated fatigue-strength intervals under batch (heat) and tramp-element shift.

Score: absolute residual on log fatigue strength. Base learner: GBR (common.model).
Methods
  std    : split conformal, calibration specimens as given by the protocol
  eq4    : the feedstock-inflated interval CI=q(1+rho*v); rho fitted on calibration
           residuals, then the normalized score |r|/(1+rho v) is conformalized
  wcp    : weighted split conformal with classifier density ratio (cal -> test)
  heat1  : one randomly chosen specimen per calibration heat (finite-sample valid
           for a new heat under heat-level exchangeability)
"""
import json, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from common import load, model, FEAT, TRAMP, conf_q, weighted_q, density_ratio

d = load()
X = d[FEAT].values.astype(float); y = d.y.values; heat = d.heat.values; v = d.v.values
ALPHA = 0.10
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
B_LIST = [-0.06, -0.09, -0.12]           # Basquin exponents for life-band translation
hv = d.groupby("heat")["v"].first()


def fit_rho(r, vv):
    """Least-absolute fit of |r| ~ a(1+rho v) over rho>=0 (a profiled as median ratio)."""
    def loss(rho):
        s = np.abs(r) / (1 + rho * vv)
        a = np.median(s)
        return np.mean(np.abs(np.abs(r) - a * (1 + rho * vv)))
    return minimize_scalar(loss, bounds=(0, 20), method="bounded").x


def tv_estimate(Xa, Xb, rng):
    """Held-out balanced-accuracy lower bound on total variation (2*BA-1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Z = np.vstack([Xa, Xb]); lab = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    idx = rng.permutation(len(Z)); h = len(Z) // 2
    sc = StandardScaler().fit(Z[idx[:h]])
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Z[idx[:h]]), lab[idx[:h]])
    pr = clf.predict(sc.transform(Z[idx[h:]])); lt = lab[idx[h:]]
    ba = 0.5 * (np.mean(pr[lt == 0] == 0) + np.mean(pr[lt == 1] == 1))
    return max(0.0, 2 * ba - 1)


def run(tr, cal, te, rng, seed, methods=("std", "eq4", "wcp", "heat1")):
    m = model("GBR", seed=seed).fit(X[tr], y[tr])
    pc, pt = m.predict(X[cal]), m.predict(X[te])
    rc, rt = y[cal] - pc, y[te] - pt
    out = {}
    if "std" in methods:
        q = conf_q(np.abs(rc), ALPHA)
        out["std"] = (np.abs(rt) <= q, np.full(len(te), q))
    if "eq4" in methods:
        rho = fit_rho(rc, v[cal])
        q = conf_q(np.abs(rc) / (1 + rho * v[cal]), ALPHA)
        hw = q * (1 + rho * v[te])
        out["eq4"] = (np.abs(rt) <= hw, hw); out["_rho"] = rho
    if "wcp" in methods:
        Fc, Ft = X[cal], X[te]
        wc = density_ratio(Fc, Ft, Fc); wt = density_ratio(Fc, Ft, Ft)
        wc = np.minimum(wc, 50); wt = np.minimum(wt, 50)
        hw = np.array([weighted_q(np.abs(rc), wc, w, ALPHA) for w in wt])
        out["wcp"] = (np.abs(rt) <= hw, hw)
    if "heat1" in methods:
        hc = heat[cal]
        pick = np.array([rng.choice(np.where(hc == h)[0]) for h in np.unique(hc)])
        q = conf_q(np.abs(rc[pick]), ALPHA)
        out["heat1"] = (np.abs(rt) <= q, np.full(len(te), q))
    out["_resid_test"] = rt
    return out


def heats_split(rng, pool, f):
    pool = rng.permutation(pool); k = int(round(f * len(pool)))
    return pool[:k], pool[k:]


def summarize(acc):
    res = {}
    for meth, L in acc.items():
        if meth.startswith("_"):
            continue
        cov = np.array([c.mean() for c, _ in L])
        hw = np.array([np.median(h[np.isfinite(h)]) if np.isfinite(h).any() else np.inf
                       for _, h in L])
        inf_frac = np.mean([np.mean(~np.isfinite(h)) for _, h in L])
        r = dict(cov=[float(cov.mean()), float(cov.std())],
                 cov_p10=float(np.quantile(cov, 0.1)), cov_min=float(cov.min()),
                 hw=[float(np.mean(hw)), float(np.std(hw))], inf_frac=float(inf_frac))
        for b in B_LIST:
            r[f"k_{abs(b):.2f}"] = float(np.exp(np.mean(hw) / abs(b)))
        r["strength_pct"] = float(100 * (np.exp(np.mean(hw)) - 1))
        res[meth] = r
    return res


all_heats = np.array(hv.index)
out = {}
t0 = time.time()

# P1: literature practice -- rows split at random into train/cal/test (60/20/20)
acc = {}
for r in range(REPS):
    rng = np.random.default_rng(r); idx = rng.permutation(len(y)); n = len(y)
    te, cal, tr = idx[:n // 5], idx[n // 5:2 * n // 5], idx[2 * n // 5:]
    o = run(tr, cal, te, rng, r, methods=("std",))
    for k, val in o.items():
        acc.setdefault(k, []).append(val)
out["P1_row"] = summarize(acc); print("P1", out["P1_row"], f"{time.time()-t0:.0f}s", flush=True)

# P2: new-heat test; calibration rows share heats with training rows (specimen-level)
# P3: new-heat test; calibration heats disjoint from training heats (heat-level)
for tag in ("P2_heattest_rowcal", "P3_heattest_heatcal"):
    acc = {}
    for r in range(REPS):
        rng = np.random.default_rng(100 + r)
        te_h, rest_h = heats_split(rng, all_heats, 0.2)
        te = np.where(np.isin(heat, te_h))[0]
        if tag.startswith("P2"):
            rest = rng.permutation(np.where(np.isin(heat, rest_h))[0]); k = len(rest) // 4
            cal, tr = rest[:k], rest[k:]
        else:
            cal_h, tr_h = heats_split(rng, rest_h, 0.25)
            cal = np.where(np.isin(heat, cal_h))[0]; tr = np.where(np.isin(heat, tr_h))[0]
        o = run(tr, cal, te, rng, r, methods=("std", "heat1") if tag.startswith("P3") else ("std",))
        for k, val in o.items():
            acc.setdefault(k, []).append(val)
    out[tag] = summarize(acc); print(tag, out[tag], f"{time.time()-t0:.0f}s", flush=True)

# P4: tramp shift -- test = top 30% heats by v; cal heats random among the rest
hs = hv.sort_values(); k30 = int(round(0.3 * len(hs)))
te_h, rest_h = np.array(hs.index[-k30:]), np.array(hs.index[:-k30])
te = np.where(np.isin(heat, te_h))[0]
acc = {}; rhos = []; tvs = []; vbins = []
for r in range(REPS):
    rng = np.random.default_rng(200 + r)
    cal_h, tr_h = heats_split(rng, rest_h, 0.25)
    cal = np.where(np.isin(heat, cal_h))[0]; tr = np.where(np.isin(heat, tr_h))[0]
    o = run(tr, cal, te, rng, r)
    rhos.append(o.pop("_rho")); tvs.append(tv_estimate(X[cal], X[te], rng))
    for k, val in o.items():
        acc.setdefault(k, []).append(val)
out["P4_shift_v"] = summarize(acc)
out["P4_shift_v"]["rho"] = [float(np.mean(rhos)), float(np.std(rhos))]
out["P4_shift_v"]["tv"] = [float(np.mean(tvs)), float(np.std(tvs))]
out["P4_shift_v"]["test_bias"] = float(np.mean([z.mean() for z in acc["_resid_test"]]))
out["P4_shift_v"]["v_range_test"] = [float(v[te].min()), float(v[te].max())]
out["P4_shift_v"]["v_range_train"] = [float(hv[rest_h].min()), float(hv[rest_h].max())]
print("P4", out["P4_shift_v"], f"{time.time()-t0:.0f}s", flush=True)

# P3 again with eq4/wcp to compare under no shift, and coverage vs v tercile
acc = {}; rhos = []; byv = {0: [], 1: [], 2: []}
cut = np.quantile(hv.values, [1 / 3, 2 / 3])
for r in range(REPS):
    rng = np.random.default_rng(300 + r)
    te_h, rest_h2 = heats_split(rng, all_heats, 0.3)
    cal_h, tr_h = heats_split(rng, rest_h2, 0.25)
    te2 = np.where(np.isin(heat, te_h))[0]
    cal = np.where(np.isin(heat, cal_h))[0]; tr = np.where(np.isin(heat, tr_h))[0]
    o = run(tr, cal, te2, rng, r, methods=("std", "eq4", "wcp"))
    rhos.append(o.pop("_rho"))
    t = np.digitize(v[te2], cut)
    for b in range(3):
        if (t == b).any():
            byv[b].append(float(o["std"][0][t == b].mean()))
    for k, val in o.items():
        acc.setdefault(k, []).append(val)
out["P3b_heat_allmethods"] = summarize(acc)
out["P3b_heat_allmethods"]["rho"] = [float(np.mean(rhos)), float(np.std(rhos))]
out["P3b_heat_allmethods"]["cov_by_v_tercile"] = [float(np.mean(byv[b])) for b in range(3)]
print("P3b", out["P3b_heat_allmethods"], f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results/e2_conformal.json", "w"), indent=1)
