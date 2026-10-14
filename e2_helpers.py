"""Helper shared by E2 and E5."""
import numpy as np
from scipy.optimize import minimize_scalar


def fit_rho(r, vv):
    """Least-absolute fit of |r| ~ a(1+rho v) over rho>=0 (a profiled as median ratio)."""
    def loss(rho):
        s = np.abs(r) / (1 + rho * vv)
        a = np.median(s)
        return np.mean(np.abs(np.abs(r) - a * (1 + rho * vv)))
    return minimize_scalar(loss, bounds=(0, 20), method="bounded").x


