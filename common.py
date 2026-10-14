"""Shared data loading, feature definitions, models and conformal utilities.

Dataset: NIMS MatNavi rotating-bending fatigue strength of 437 steels, as released in
the supplementary file of Agrawal et al. (2014), IMMI 3:8 (40192_2013_16_MOESM1_ESM.csv).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NIMS_CSV", os.path.join(HERE, "data", "40192_2013_16_MOESM1_ESM.csv"))

COMP = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
INCL = ["dA", "dB", "dC"]
PROC = ["NT", "THT", "THt", "THQCr", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt", "TCr",
        "RedRatio"]
FEAT = PROC + COMP + INCL
TRAMP = ["Cu", "incl", "P", "S"]   # scrap-inherited residuals and inclusions


def load():
    d = pd.read_csv(DATA)
    d["incl"] = d.dA + d.dB + d.dC
    d["heat"] = d.groupby(COMP + INCL, sort=False).ngroup()   # same chemistry = same heat
    d["cls"] = np.where(d.CT > 30, "carburized",
                        np.where(d.THT > 30, "through-hardened", "normalized"))
    d["y"] = np.log(d.Fatigue.values)                          # log fatigue strength
    # feedstock-variability descriptor v(x) in [0,1]: mean percentile rank of tramp signals
    d["v"] = d[TRAMP].rank(pct=True).mean(axis=1)
    return d


def heat_v(d):
    """One v per heat (heats share chemistry, so v is constant within heat)."""
    return d.groupby("heat")["v"].first()


def model(name, seed=0):
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=600, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=seed)
    if name == "RF":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=seed,
                                     n_jobs=1)
    if name == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if name == "GP":
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(FEAT))) + WhiteKernel(1e-3)
        return make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                      random_state=seed, n_restarts_optimizer=0))
    raise ValueError(name)


def conf_q(scores, alpha):
    """Split-conformal quantile: ceil((n+1)(1-alpha))-th smallest score (inf if too few)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.sort(scores)[k - 1]


def weighted_q(scores, w_cal, w_test, alpha):
    """Weighted split conformal (Tibshirani et al., 2019) for a single test weight."""
    order = np.argsort(scores)
    s = scores[order]
    w = w_cal[order]
    p = np.append(w, w_test) / (w.sum() + w_test)
    cum = np.cumsum(p[:-1])
    idx = np.searchsorted(cum, 1 - alpha)
    return np.inf if idx >= len(s) else s[idx]


def density_ratio(Xa, Xb, Xq, C=1.0):
    """w(x)=p_b(x)/p_a(x) via probabilistic classification on standardized features."""
    sc = StandardScaler().fit(np.vstack([Xa, Xb]))
    clf = LogisticRegression(C=C, max_iter=2000)
    X = sc.transform(np.vstack([Xa, Xb]))
    lab = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf.fit(X, lab)
    p = clf.predict_proba(sc.transform(Xq))[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return (p / (1 - p)) * (len(Xa) / len(Xb))
