"""E5: is the copper shift real? Null distribution of random-heat splits, class control,
and calibrated intervals on the copper-shift split."""
import json, sys, time
import numpy as np
from common import load, model, FEAT, conf_q
from e2_helpers import fit_rho

d = load()
X = d[FEAT].values.astype(float); y = d.y.values; heat = d.heat.values
NULL = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ALPHA = 0.10
d["PS"] = d.P + d.S
th = (d.cls == "through-hardened").values
out = {}
t0 = time.time()


def shift_idx(key, mask=None, f=0.3):
    base = np.ones(len(d), bool) if mask is None else mask
    hk = d[base].groupby("heat")[key].first().sort_values()
    k = int(round(f * len(hk)))
    return (np.where(base & np.isin(heat, hk.index[:-k]))[0],
            np.where(base & np.isin(heat, hk.index[-k:]))[0])


def score(m, tr, te, seed=0):
    mdl = model(m, seed=seed).fit(X[tr], y[tr])
    r = mdl.predict(X[te]) - y[te]
    return float(np.mean(np.abs(np.exp(y[te] + r) - np.exp(y[te])))), float(np.mean(r))


for scope, mask in (("all", None), ("TH", th)):
    pool = np.unique(heat) if mask is None else np.unique(heat[mask])
    base = np.ones(len(d), bool) if mask is None else mask
    for m in ("GBR", "Ridge"):
        null_mae, null_bias = [], []
        for r in range(NULL):
            rng = np.random.default_rng(7000 + r)
            p = rng.permutation(pool); k = int(round(0.3 * len(p)))
            te = np.where(base & np.isin(heat, p[:k]))[0]; tr = np.where(base & np.isin(heat, p[k:]))[0]
            a, b = score(m, tr, te, seed=r % 5)
            null_mae.append(a); null_bias.append(b)
        null_mae, null_bias = np.array(null_mae), np.array(null_bias)
        res = dict(null_mae=[float(null_mae.mean()), float(np.quantile(null_mae, .05)),
                             float(np.quantile(null_mae, .95))],
                   null_bias=[float(null_bias.mean()), float(np.quantile(null_bias, .05)),
                              float(np.quantile(null_bias, .95))])
        for key in ("v", "Cu", "incl", "PS"):
            tr, te = shift_idx(key, mask)
            vals = [score(m, tr, te, seed=s) for s in range(5 if m == "GBR" else 1)]
            a = float(np.mean([z[0] for z in vals])); b = float(np.mean([z[1] for z in vals]))
            res[key] = dict(mae=a, bias=b,
                            p_mae=float((1 + np.sum(null_mae >= a)) / (NULL + 1)),
                            p_bias=float((1 + np.sum(null_bias >= b)) / (NULL + 1)),
                            n_test=int(len(te)),
                            frac_carb=float(np.mean(d.cls.values[te] == "carburized")),
                            cu_min_test=float(d.Cu.values[te].min()))
        out[f"{scope}_{m}"] = res
        print(scope, m, json.dumps(res), f"{time.time()-t0:.0f}s", flush=True)

# calibrated intervals on the copper shift (all data and TH)
for scope, mask in (("all", None), ("TH", th)):
    tr_all, te = shift_idx("Cu", mask)
    rest_h = np.unique(heat[tr_all])
    cu_pct = d.Cu.rank(pct=True).values
    acc = {"std": [], "eq4": [], "heat1": []}; hws = {"std": [], "eq4": [], "heat1": []}
    for r in range(50):
        rng = np.random.default_rng(8000 + r)
        p = rng.permutation(rest_h); k = int(round(0.25 * len(p)))
        cal = np.where(np.isin(heat, p[:k]))[0]; tr = np.where(np.isin(heat, p[k:]))[0]
        mdl = model("GBR", seed=r).fit(X[tr], y[tr])
        rc = y[cal] - mdl.predict(X[cal]); rt = y[te] - mdl.predict(X[te])
        q = conf_q(np.abs(rc), ALPHA); acc["std"].append(np.mean(np.abs(rt) <= q)); hws["std"].append(q)
        rho = fit_rho(rc, cu_pct[cal]); q2 = conf_q(np.abs(rc) / (1 + rho * cu_pct[cal]), ALPHA)
        hw = q2 * (1 + rho * cu_pct[te]); acc["eq4"].append(np.mean(np.abs(rt) <= hw)); hws["eq4"].append(np.median(hw))
        one = np.array([rng.choice(np.where(heat[cal] == h)[0]) for h in np.unique(heat[cal])])
        q3 = conf_q(np.abs(rc[one]), ALPHA); acc["heat1"].append(np.mean(np.abs(rt) <= q3)); hws["heat1"].append(q3)
    out[f"conf_Cu_{scope}"] = {k: dict(cov=float(np.mean(acc[k])), cov_p10=float(np.quantile(acc[k], .1)),
                                       hw=float(np.mean(hws[k])), k009=float(np.exp(np.mean(hws[k]) / 0.09)))
                               for k in acc}
    print("conf Cu", scope, out[f"conf_Cu_{scope}"], flush=True)
json.dump(out, open("results/e5_cu.json", "w"), indent=1)
