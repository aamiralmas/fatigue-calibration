"""E6: which side do the misses fall on? One-sided (lower, design-allowable) conformal bounds
and the split of two-sided misses into over- and under-prediction."""
import json
import numpy as np
from common import load, model, FEAT, conf_q

d = load()
X = d[FEAT].values.astype(float); y = d.y.values; heat = d.heat.values
th = (d.cls == "through-hardened").values
out = {}


def shift_heats(key, mask):
    base = np.ones(len(d), bool) if mask is None else mask
    hk = d[base].groupby("heat")[key].first().sort_values()
    k = int(round(0.3 * len(hk)))
    return np.array(hk.index[-k:]), np.array(hk.index[:-k]), base


for name, key, mask in (("heat", None, None), ("shift_v", "v", None), ("shift_Cu", "Cu", None),
                        ("heat_TH", None, th), ("shift_Cu_TH", "Cu", th)):
    lo95, two_cov, miss_hi, miss_lo = [], [], [], []
    for r in range(50):
        rng = np.random.default_rng(4000 + r)
        if key is None:
            base = np.ones(len(d), bool) if mask is None else mask
            p = rng.permutation(np.unique(heat[base])); k = int(round(0.3 * len(p)))
            te_h, rest_h = p[:k], p[k:]
        else:
            te_h, rest_h, base = shift_heats(key, mask)
        rest_h = rng.permutation(rest_h); kc = int(round(0.25 * len(rest_h)))
        te = np.where(base & np.isin(heat, te_h))[0]
        cal = np.where(base & np.isin(heat, rest_h[:kc]))[0]
        tr = np.where(base & np.isin(heat, rest_h[kc:]))[0]
        m = model("GBR", seed=r).fit(X[tr], y[tr])
        pc, pt = m.predict(X[cal]), m.predict(X[te])
        # one-sided lower bound at 95%: score = overprediction (pred - true)
        q1 = conf_q(pc - y[cal], 0.05)
        lo95.append(np.mean(y[te] >= pt - q1))
        q2 = conf_q(np.abs(y[cal] - pc), 0.10)
        rt = y[te] - pt
        two_cov.append(np.mean(np.abs(rt) <= q2))
        miss_hi.append(np.mean(rt > q2)); miss_lo.append(np.mean(rt < -q2))
    out[name] = dict(lower95_cov=float(np.mean(lower := np.array(lo95))),
                     lower95_p10=float(np.quantile(lower, .1)),
                     two90_cov=float(np.mean(two_cov)),
                     miss_true_above=float(np.mean(miss_hi)), miss_true_below=float(np.mean(miss_lo)))
    print(name, out[name], flush=True)
json.dump(out, open("results/e6_onesided.json", "w"), indent=1)
