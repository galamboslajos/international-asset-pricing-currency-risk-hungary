"""Joint equity pricing-error intercepts for notebook 11; earlier engines stay frozen."""
import numpy as np
from scipy.optimize import minimize
import pricing_robustness as base
import covariance_robustness as covariance
import local_risk_pricing as local


def alpha_indices(theta, free_alpha):
    return np.arange(len(theta)-3, len(theta)) if free_alpha else np.array([], dtype=int)


def expand_zero(theta):
    return np.r_[theta, np.zeros(3)]


def filter_model(theta, y, pdesign, scales, name, free_alpha):
    if not free_alpha:
        f = local.filter_model(theta, y, pdesign, scales, name)
        return {**f, "equity_alphas": np.zeros(3), "pricing_mu": f["mu"]}
    old_theta = theta[:-3]
    omega, domega, terms, dterms, target, dtarget, physical = local.covariance_terms(old_theta, pdesign, scales, name)
    def pad(d):
        return np.concatenate([d, np.zeros((3, *d.shape[1:]))], axis=0)
    domega, dtarget = pad(domega), pad(dtarget)
    dterms = np.array([pad(d) for d in dterms])
    H, dH = target.copy(), dtarget.copy()
    shock_lags = len(terms)-1
    history = [target.copy() for _ in range(shock_lags)]
    dhistory = [dtarget.copy() for _ in range(shock_lags)]
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariances = np.empty((n, 5, 5))
    means, pricing_means, residuals = np.empty_like(y), np.empty_like(y), np.empty_like(y)
    indices = local.price_indices(pdesign)
    prices, derivatives = {}, {}
    for label, design in pdesign.items():
        prices[label] = base.PRICE_SCALE*(design@theta[indices[label]])
        derivatives[label] = np.zeros((n, q))
        derivatives[label][:, indices[label]] = base.PRICE_SCALE*design
    alphas = scales[:3]*theta[-3:]
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError("Unstable covariance recursion")
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
        mu, dmu = np.zeros(5), np.zeros((q, 5))
        for label, factor in [("W", 0), ("H", 3), ("O", 4)]:
            value = prices[label][t]
            mu += value*H[:, factor]
            dmu += value*dH[:, :, factor]+derivatives[label][t, :, None]*H[None, :, factor]
        mu[1] += base.PRICE_SCALE*theta[1]*H[1, 1]
        dmu[:, 1] += base.PRICE_SCALE*theta[1]*dH[:, 1, 1]
        dmu[1, 1] += base.PRICE_SCALE*H[1, 1]
        mu[2] += prices["L"][t]*H[2, 2]
        dmu[:, 2] += prices["L"][t]*dH[:, 2, 2]+derivatives["L"][t]*H[2, 2]
        pricing_means[t] = mu
        mu[:3] += alphas
        dmu[q-3+np.arange(3), np.arange(3)] += scales[:3]
        error, de = y[t]-mu, -dmu
        v = inverse@error
        losses[t] = .5*(5*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error@v)
        scores[t] = .5*np.einsum("ij,kij->k", inverse-np.outer(v, v), dH)+de@v
        covariances[t], means[t], residuals[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None]*error[None, None, :]+error[None, :, None]*de[:, None, :]
        history, dhistory = [shock]+history[:-1], [dshock]+dhistory[:-1]
        dnext = domega+dterms[-1]*H+terms[-1]*dH
        next_H = omega+terms[-1]*H
        for j in range(shock_lags):
            next_H += terms[j]*history[j]
            dnext += dterms[j]*history[j]+terms[j]*dhistory[j]
        H, dH = next_H, dnext
    return {"losses": losses, "scores": scores, "H": covariances, "mu": means,
            "pricing_mu": pricing_means, "errors": residuals, "equity_alphas": alphas,
            "prices": base.PRICE_SCALE*theta[:5], "world_prices": prices["W"], "local_prices": prices["L"],
            "fx_prices": np.c_[prices["H"], prices["O"]], "intercept": omega, "target": target,
            "alpha_lags": physical[:, :-1], "alpha": physical[:, :-1].sum(axis=1), "beta": physical[:, -1]}


def objective(theta, *args):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            f = filter_model(theta, *args)
        if not np.isfinite(f["losses"]).all() or not np.isfinite(f["scores"]).all():
            raise FloatingPointError("Nonfinite likelihood or gradient")
        return f["losses"].mean(), f["scores"].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def bounds(pdesign, name, free_alpha):
    old = local.bounds(pdesign, name)
    return np.r_[old, np.tile([[-np.inf, np.inf]], (3, 1))] if free_alpha else old


def initial_theta(y, pdesign, scales, name, free_alpha):
    theta = local.initial_theta(y, pdesign, scales, name)
    return expand_zero(theta) if free_alpha else theta


def fit_starts(args, starts, maxiter=2600):
    limits = bounds(args[1], args[-2], args[-1])
    runs, solutions = [], []
    for i, start in enumerate(starts, 1):
        coarse = minimize(objective, start, args=args, method="BFGS", jac=True, options={"maxiter": maxiter, "gtol": 1e-6})
        result = minimize(objective, np.clip(coarse.x, limits[:, 0], limits[:, 1]), args=args, method="L-BFGS-B", jac=True,
                          bounds=limits, options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50, "maxcor": 50})
        gradient = float(np.max(np.abs(result.jac)))
        print(f"  start {i}: success={result.success}, NLL={result.fun:.9f}, gradient={gradient:.2e}, iterations={coarse.nit}+{result.nit}", flush=True)
        runs.append({"start": i, "success": bool(result.success), "mean_nll": float(result.fun),
                     "max_abs_gradient": gradient, "iterations": int(coarse.nit+result.nit), "message": str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7 and np.max(np.abs(r.jac)) < 1e-4]
    if not valid: raise RuntimeError("No converged fit with a small gradient; inspect starts/MAX_ITER.")
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError("A lower nonconverged candidate remains; increase MAX_ITER.")
    return best, runs


def loss_hessian(theta, *args, step=1e-4):
    result = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j]));plus, minus = theta.copy(), theta.copy()
        plus[j] += delta;minus[j] -= delta
        result[:, j] = (objective(plus, *args)[1]-objective(minus, *args)[1])/(2*delta)
    return (result+result.T)/2


def inference(theta, args, filtered, hac_lags=(0, 3, 6, 12), se_step_tol=.01):
    A = loss_hessian(theta, *args);small = loss_hessian(theta, *args, step=5e-5)
    condition = np.linalg.cond(A)
    sensitivity = np.max(np.abs(A-small))/max(1., np.max(np.abs(small)))
    limits = bounds(args[1], args[-2], args[-1]);distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append("parameter bound")
    if min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min()) <= 0: reasons.append("nonpositive curvature")
    if condition >= 1e9: reasons.append("ill-conditioned curvature")
    if sensitivity >= 1e-3: reasons.append("Hessian step sensitivity")
    k = local.layout(args[1], args[-2])[0]
    mean_positions = np.r_[np.arange(k), alpha_indices(theta, args[-1])]
    estimates = {}
    for lag in hac_lags:
        V, why, se_difference = np.full_like(A, np.nan), reasons.copy(), np.nan
        if not why:
            meat = covariance.hac_meat(filtered["scores"], lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V, Vsmall = inv@meat@inv.T/len(filtered["scores"]), inv_small@meat@inv_small.T/len(filtered["scores"])
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0: why.append("invalid sandwich")
            else:
                se_difference = np.max(np.abs(np.sqrt(np.diag(Vsmall)[mean_positions])/np.sqrt(np.diag(V)[mean_positions])-1))
                if se_difference >= se_step_tol: why.append("mean-coefficient SE step sensitivity")
        estimates[lag] = {"covariance": V if not why else np.full_like(A, np.nan), "ready": not why,
                          "reason": "; ".join(why) or "numerically ready", "mean_SE_step_difference": se_difference}
    return {"Hessian": A, "Hessian_condition": condition, "Hessian_step_difference": sensitivity,
            "Hessian_min_eigenvalue": np.linalg.eigvalsh(A).min(), "distance_to_bounds": distance,
            "near_bound": bool(distance.min() < 1e-4), "estimates": estimates}
