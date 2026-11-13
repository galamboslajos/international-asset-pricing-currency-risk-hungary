"""Covariance and score-HAC checks for notebook 09; notebook 08 stays unchanged."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import pricing_robustness as base

MODELS = {"scalar11": ("scalar", 1), "grouped11": ("grouped", 1),
          "diagonal11": ("diagonal", 1), "grouped21": ("grouped", 2)}


def layout(pdesign, name):
    covariance, lags = MODELS[name]
    k = 5 + sum(pdesign[a].shape[1]-1 for a in ["W", "H", "O"])
    groups = max(base.GROUPS[covariance])+1
    return k, groups*(lags+1)


def unpack21(theta, pdesign, scales):
    k, dynamics = layout(pdesign, "grouped21")
    q = len(theta)
    C, dC = np.zeros((5, 5)), np.zeros((q, 5, 5))
    for offset, (i, j) in enumerate(zip(*base.TRI)):
        value = np.exp(theta[k+offset]) if i == j else theta[k+offset]
        C[i, j] = scales[i]*value
        dC[k+offset, i, j] = scales[i]*(value if i == j else 1.)
    omega = C@C.T
    domega = dC@C.T + C@dC.transpose(0, 2, 1)
    weights = softmax(np.c_[np.zeros(3), theta[-dynamics:].reshape(3, 3)], axis=1)
    groups = np.asarray(base.GROUPS["grouped"])
    physical = weights[groups, 1:]
    terms, dterms = [], []
    for component in range(3):
        v = np.sqrt(physical[:, component])
        dv = np.zeros((q, 5))
        for i in range(5):
            for coordinate in range(3):
                j = q-dynamics+3*groups[i]+coordinate
                dv[j, i] = .5*v[i]*((component == coordinate)-physical[i, coordinate])
        terms.append(np.outer(v, v))
        dterms.append(dv[:, :, None]*v[None, None, :]+v[None, :, None]*dv[:, None, :])
    terms, dterms = np.asarray(terms), np.asarray(dterms)
    denominator = 1-terms.sum(axis=0)
    target = omega/denominator
    dtarget = domega/denominator+omega*dterms.sum(axis=0)/denominator**2
    return omega, domega, terms, dterms, target, dtarget, physical


def filter_model(theta, y, pdesign, scales, name):
    covariance, lags = MODELS[name]
    if lags == 1:
        return base.filter_model(theta, y, np.empty((len(y), 0)), pdesign, scales, "priced", covariance)
    omega, domega, terms, dterms, target, dtarget, physical = unpack21(theta, pdesign, scales)
    H, dH = target.copy(), dtarget.copy()
    lag_shock, dlag_shock = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariances, means, residuals = np.empty((n, 5, 5)), np.empty_like(y), np.empty_like(y)
    indices = base.price_indices(pdesign)
    prices, derivatives = {}, {}
    for label, design in pdesign.items():
        prices[label] = base.PRICE_SCALE*(design@theta[indices[label]])
        derivatives[label] = np.zeros((n, q))
        derivatives[label][:, indices[label]] = base.PRICE_SCALE*design
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError("Unstable covariance recursion")
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
        mu, dmu = np.zeros(5), np.zeros((q, 5))
        for label, factor in [("W", 0), ("H", 3), ("O", 4)]:
            value = prices[label][t]
            mu += value*H[:, factor]
            dmu += value*dH[:, :, factor]+derivatives[label][t, :, None]*H[None, :, factor]
        for asset in [1, 2]:
            mu[asset] += base.PRICE_SCALE*theta[asset]*H[asset, asset]
            dmu[:, asset] += base.PRICE_SCALE*theta[asset]*dH[:, asset, asset]
            dmu[asset, asset] += base.PRICE_SCALE*H[asset, asset]
        error, de = y[t]-mu, -dmu
        v = inverse@error
        losses[t] = .5*(5*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error@v)
        scores[t] = .5*np.einsum("ij,kij->k", inverse-np.outer(v, v), dH)+de@v
        covariances[t], means[t], residuals[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None]*error[None, None, :]+error[None, :, None]*de[:, None, :]
        dnext = (domega+dterms[0]*shock+terms[0]*dshock+
                 dterms[1]*lag_shock+terms[1]*dlag_shock+dterms[2]*H+terms[2]*dH)
        H = omega+terms[0]*shock+terms[1]*lag_shock+terms[2]*H
        dH, lag_shock, dlag_shock = dnext, shock, dshock
    return {"losses": losses, "scores": scores, "H": covariances, "mu": means, "errors": residuals,
            "prices": base.PRICE_SCALE*theta[:5], "world_prices": prices["W"],
            "fx_prices": np.c_[prices["H"], prices["O"]], "intercept": omega, "target": target,
            "alpha": physical[:, :2].sum(axis=1), "alpha_lags": physical[:, :2], "beta": physical[:, 2]}


def objective(theta, *args):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            f = filter_model(theta, *args)
        if not np.isfinite(f["losses"]).all() or not np.isfinite(f["scores"]).all():
            raise FloatingPointError("Nonfinite likelihood or gradient")
        return f["losses"].mean(), f["scores"].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def expand_grouped(theta, name, lag2_share=.2):
    if name == "diagonal11":
        return np.r_[theta[:-6], theta[-6:].reshape(3, 2)[base.GROUPS["grouped"]].ravel()]
    if name == "grouped21":
        weights = softmax(np.c_[np.zeros(3), theta[-6:].reshape(3, 2)], axis=1)
        new = np.c_[(1-lag2_share)*weights[:, 1], lag2_share*weights[:, 1], weights[:, 2]]
        return np.r_[theta[:-6], np.log(new/weights[:, :1]).ravel()]
    raise ValueError("Expand grouped11 to diagonal11 or grouped21")


def initial_theta(y, pdesign, scales, name, alpha=.1, beta=.75):
    covariance, lags = MODELS[name]
    theta = base.initial_theta(y, np.empty((len(y), 0)), pdesign, scales, "priced", covariance, alpha, beta)
    return theta if lags == 1 else expand_grouped(theta, name, lag2_share=.5)


def bounds(pdesign, name):
    k, dynamics = layout(pdesign, name)
    return np.asarray([(-100., 100.)]*k+
                      [(-8., 3.) if i == j else (-4., 4.) for i, j in zip(*base.TRI)]+
                      [(-10., 10.)]*dynamics)


def fit_starts(args, starts, maxiter=2000):
    limits = bounds(args[1], args[-1])
    runs, solutions = [], []
    for i, start in enumerate(starts, 1):
        coarse = minimize(objective, start, args=args, method="BFGS", jac=True,
                          options={"maxiter": maxiter, "gtol": 1e-6})
        result = minimize(objective, np.clip(coarse.x, limits[:, 0], limits[:, 1]), args=args,
                          method="L-BFGS-B", jac=True, bounds=limits,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50, "maxcor": 50})
        gradient = float(np.max(np.abs(result.jac)))
        print(f"  start {i}: success={result.success}, NLL={result.fun:.9f}, gradient={gradient:.2e}, iterations={coarse.nit}+{result.nit}", flush=True)
        runs.append({"start": i, "success": bool(result.success), "mean_nll": float(result.fun),
                     "max_abs_gradient": gradient, "iterations": int(coarse.nit+result.nit), "message": str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7 and np.max(np.abs(r.jac)) < 1e-4]
    if not valid:
        raise RuntimeError("No converged fit with a small gradient; inspect starts/MAX_ITER.")
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError("A lower nonconverged candidate remains; increase MAX_ITER.")
    return best, runs


def loss_hessian(theta, *args, step=1e-4):
    result = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j]))
        plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        result[:, j] = (objective(plus, *args)[1]-objective(minus, *args)[1])/(2*delta)
    return (result+result.T)/2


def hac_meat(scores, lags):
    """Bartlett long-run score covariance, divided by T, without a T/(T-q) correction."""
    if not isinstance(lags, int) or not 0 <= lags < len(scores):
        raise ValueError("HAC lags must be an integer from zero to T-1")
    centred = scores-scores.mean(axis=0)
    meat = centred.T@centred
    for lag in range(1, lags+1):
        cross = centred[lag:].T@centred[:-lag]
        meat += (1-lag/(lags+1))*(cross+cross.T)
    return meat/len(scores)


def inference(theta, args, filtered, hac_lags=(0, 3, 6, 12), se_step_tol=.01):
    A = loss_hessian(theta, *args)
    small = loss_hessian(theta, *args, step=5e-5)
    condition = np.linalg.cond(A)
    sensitivity = np.max(np.abs(A-small))/max(1., np.max(np.abs(small)))
    limits = bounds(args[1], args[-1])
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append("parameter bound")
    if min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min()) <= 0: reasons.append("nonpositive curvature")
    if condition >= 1e9: reasons.append("ill-conditioned curvature")
    if sensitivity >= 1e-3: reasons.append("Hessian step sensitivity")
    k = layout(args[1], args[-1])[0]
    estimates = {}
    for lag in hac_lags:
        V = np.full_like(A, np.nan)
        why, se_difference = reasons.copy(), np.nan
        if not why:
            meat = hac_meat(filtered["scores"], lag)
            inverse, inverse_small = np.linalg.inv(A), np.linalg.inv(small)
            V = inverse@meat@inverse.T/len(filtered["scores"])
            Vsmall = inverse_small@meat@inverse_small.T/len(filtered["scores"])
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0:
                why.append("invalid sandwich")
            else:
                se_difference = np.max(np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1))
                if se_difference >= se_step_tol: why.append("price SE step sensitivity")
        estimates[lag] = {"covariance": V if not why else np.full_like(A, np.nan), "ready": not why,
                          "reason": "; ".join(why) or "numerically ready", "price_SE_step_difference": se_difference}
    return {"Hessian": A, "Hessian_condition": condition, "Hessian_step_difference": sensitivity,
            "Hessian_min_eigenvalue": np.linalg.eigvalsh(A).min(), "distance_to_bounds": distance,
            "near_bound": bool(distance.min() < 1e-4), "estimates": estimates}
