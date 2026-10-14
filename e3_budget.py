"""E3: specimen budget -- coverage of new-heat intervals when only B heats are calibrated.
E4: learning curve -- training heats needed for a target life-scatter band.
"""
import json, sys, time
import numpy as np
from scipy.stats import beta as Beta
from common import load, model, FEAT, conf_q

d = load()
X = d[FEAT].values.astype(float); y = d.y.values; heat = d.heat.values
heats = np.array(sorted(np.unique(heat)))
ALPHA = 0.10
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SUB = 25                                      # calibration subsamples per trained model
B_LIST = [3, 5, 9, 14, 19, 30]
out = {"E3": {}, "E4": {}}
t0 = time.time()

# ---------------- E3 ----------------
cov = {(b, m): [] for b in B_LIST for m in ("spec", "heat1")}
inf = {(b, m): [] for b in B_LIST for m in ("spec", "heat1")}
for r in range(REPS):
    rng = np.random.default_rng(500 + r)
    p = rng.permutation(heats)
    te_h, cal_pool, tr_h = p[:30], p[30:65], p[65:]
    te = np.isin(heat, te_h); tr = np.isin(heat, tr_h)
    mdl = model("GBR", seed=r).fit(X[tr], y[tr])
    res = np.abs(y - mdl.predict(X))
    for b in B_LIST:
        for _ in range(SUB):
            ch = rng.choice(cal_pool, b, replace=False)
            spec = np.where(np.isin(heat, ch))[0]
            one = np.array([rng.choice(np.where(heat == h)[0]) for h in ch])
            for m, idx in (("spec", spec), ("heat1", one)):
                q = conf_q(res[idx], ALPHA)
                inf[(b, m)].append(not np.isfinite(q))
                cov[(b, m)].append(float(np.mean(res[te] <= q)))
    print("E3 rep", r, f"{time.time()-t0:.0f}s", flush=True)
for b in B_LIST:
    for m in ("spec", "heat1"):
        c = np.array(cov[(b, m)])
        n_eff = b if m == "heat1" else None
        th = None
        if m == "heat1":
            l = int(np.ceil((b + 1) * (1 - ALPHA)))
            if l <= b:
                th = dict(mean=l / (b + 1), p_below_080=float(Beta(l, b + 1 - l).cdf(0.8)))
        out["E3"][f"{m}_B{b}"] = dict(cov_mean=float(c.mean()), cov_p10=float(np.quantile(c, .1)),
                                     p_below_080=float(np.mean(c < 0.8)),
                                     inf_frac=float(np.mean(inf[(b, m)])),
                                     n_spec=float(np.mean([np.isin(heat, [0]).sum()])) if False else None,
                                     theory=th)
        print(m, b, out["E3"][f"{m}_B{b}"], flush=True)

# ---------------- E4 ----------------
H_LIST = [10, 20, 40, 60, 80, 100, 115]
for mname in ("GBR", "Ridge"):
    for H in H_LIST:
        q50, q90, within2 = [], [], []
        for r in range(REPS // 2):
            rng = np.random.default_rng(900 + r)
            p = rng.permutation(heats)
            te_h, tr_h = p[:40], p[40:40 + H]
            te = np.isin(heat, te_h); tr = np.isin(heat, tr_h)
            mdl = model(mname, seed=r).fit(X[tr], y[tr])
            a = np.abs(y[te] - mdl.predict(X[te]))
            q50.append(np.median(a)); q90.append(np.quantile(a, 0.9))
            within2.append(np.mean(a <= 0.09 * np.log(2)))
        out["E4"][f"{mname}_H{H}"] = dict(n_spec=float(np.isin(heat, tr_h).sum()),
                                          q50=float(np.mean(q50)), q90=float(np.mean(q90)),
                                          within2_b009=float(np.mean(within2)))
        print("E4", mname, H, out["E4"][f"{mname}_H{H}"], f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results/e3_budget.json", "w"), indent=1)
