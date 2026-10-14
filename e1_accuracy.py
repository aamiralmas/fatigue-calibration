"""E1: point accuracy under literature (row), heat-grouped, and tramp-shift protocols."""
import json, sys, time
import numpy as np
from common import load, model, FEAT, heat_v

d = load()
X = d[FEAT].values.astype(float); y = d.y.values; heat = d.heat.values
hv = heat_v(d)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODELS = ["GBR", "RF", "Ridge", "GP"]


def metrics(yt, yp):
    S, P = np.exp(yt), np.exp(yp)
    r = yp - yt
    return dict(rmse=float(np.sqrt(np.mean((S - P) ** 2))), mae=float(np.mean(np.abs(S - P))),
                r2=float(1 - np.sum((S - P) ** 2) / np.sum((S - S.mean()) ** 2)),
                lrmse=float(np.sqrt(np.mean(r ** 2))), bias=float(np.mean(r)),
                within5=float(np.mean(np.abs(r) <= np.log(1.05))))


def split_row(rng, f=0.3):
    idx = rng.permutation(len(y)); k = int(f * len(y))
    return idx[k:], idx[:k]


def split_heat(rng, f=0.3, pool=None):
    hs = np.array(hv.index if pool is None else pool)
    hs = rng.permutation(hs); k = int(round(f * len(hs)))
    te = np.isin(heat, hs[:k]); tr = np.isin(heat, hs[k:])
    return np.where(tr)[0], np.where(te)[0]


def split_shift(key, f=0.3, reverse=False, mask=None):
    g = d if mask is None else d[mask]
    hk = g.groupby("heat")[key].first().sort_values()
    if reverse:
        hk = hk[::-1]
    k = int(round(f * len(hk)))
    te_h, tr_h = hk.index[-k:], hk.index[:-k]
    base = np.ones(len(d), bool) if mask is None else mask
    return np.where(base & np.isin(heat, tr_h))[0], np.where(base & np.isin(heat, te_h))[0]


d["PS"] = d.P + d.S
th = (d.cls == "through-hardened").values
protocols = {
    "row": lambda r: split_row(r),
    "heat": lambda r: split_heat(r),
    "shift_v": lambda r: split_shift("v"),
    "shift_v_rev": lambda r: split_shift("v", reverse=True),
    "shift_Cu": lambda r: split_shift("Cu"),
    "shift_incl": lambda r: split_shift("incl"),
    "shift_PS": lambda r: split_shift("PS"),
    "heat_TH": lambda r: split_heat(r, pool=d[th].heat.unique()),
    "shift_v_TH": lambda r: split_shift("v", mask=th),
}
out = {}
t0 = time.time()
for p, fn in protocols.items():
    out[p] = {}
    for m in MODELS:
        reps = REPS if m != "GP" else max(5, REPS // 4)
        if p.startswith("shift") and m in ("Ridge", "GP"):
            reps = 1                                    # deterministic split and model
        res = []
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            tr, te = fn(rng)
            mdl = model(m, seed=r).fit(X[tr], y[tr])
            res.append(metrics(y[te], mdl.predict(X[te])))
        out[p][m] = {k: [float(np.mean([z[k] for z in res])), float(np.std([z[k] for z in res]))]
                     for k in res[0]}
        out[p][m]["n_test"] = int(len(te)); out[p][m]["n_train"] = int(len(tr))
        out[p][m]["test_v_mean"] = float(d.v.values[te].mean())
        print(p, m, {k: round(v[0], 3) for k, v in out[p][m].items() if isinstance(v, list)},
              len(tr), len(te), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results/e1_accuracy.json", "w"), indent=1)
