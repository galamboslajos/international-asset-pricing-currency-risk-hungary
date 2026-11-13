"""Calendar-preserving likelihood stress checks for notebook 14.

The complete frozen return sequence always enters the existing GARCH filter.
Fixed zero/one weights select joint likelihood contributions, not state updates.
"""
import numpy as np
from scipy.optimize import minimize
import local_risk_pricing as pricing
import covariance_robustness as covariance


def objective(theta, y, pdesign, scales, name, weights):
    if weights.shape != (len(y),) or not np.isin(weights, [0., 1.]).all() or weights.sum() == 0:
        raise ValueError("Use a nonempty calendar-aligned zero/one weight vector")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            f = pricing.filter_model(theta, y, pdesign, scales, name)
        if not np.isfinite(f["losses"]).all() or not np.isfinite(f["scores"]).all():
            raise FloatingPointError("Nonfinite likelihood or score")
        return np.mean(weights*f["losses"]), np.mean(weights[:, None]*f["scores"], axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def projected_gradient(theta, gradient, limits):
    result = gradient.copy()
    result[(theta-limits[:, 0] < 1e-6) & (gradient > 0)] = 0
    result[(limits[:, 1]-theta < 1e-6) & (gradient < 0)] = 0
    return result


def fit_starts(args, starts, maxiter=2800):
    limits = pricing.bounds(args[1], args[-2])
    runs, solutions = [], []
    for i, start in enumerate(starts, 1):
        coarse = minimize(objective, start, args=args, method="BFGS", jac=True, options={"maxiter": maxiter, "gtol": 1e-6})
        result = minimize(objective, np.clip(coarse.x, limits[:, 0], limits[:, 1]), args=args, method="L-BFGS-B", jac=True,
                          bounds=limits, options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50, "maxcor": 50})
        raw = float(np.max(np.abs(result.jac)))
        projected = float(np.max(np.abs(projected_gradient(result.x, result.jac, limits))))
        print(f"  start {i}: success={result.success}, weighted NLL/T={result.fun:.9f}, raw/projected gradient={raw:.2e}/{projected:.2e}, iterations={coarse.nit}+{result.nit}", flush=True)
        runs.append({"start": i, "success": bool(result.success), "mean_weighted_nll": float(result.fun), "raw_gradient": raw,
                     "projected_gradient": projected, "iterations": int(coarse.nit+result.nit), "message": str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7 and
             np.max(np.abs(projected_gradient(r.x, r.jac, limits))) < 1e-4]
    if not valid: raise RuntimeError("No converged constrained fit; inspect starts/MAX_ITER.")
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError("A lower unconverged candidate remains; increase MAX_ITER.")
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
    limits = pricing.bounds(args[1], args[-2]);distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append("parameter bound")
    if min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min()) <= 0: reasons.append("nonpositive curvature")
    if condition >= 1e9: reasons.append("ill-conditioned curvature")
    if sensitivity >= 1e-3: reasons.append("Hessian step sensitivity")
    k = pricing.layout(args[1], args[-2])[0]
    scores = args[-1][:, None]*filtered["scores"]
    estimates = {}
    for lag in hac_lags:
        V, why, se_difference = np.full_like(A, np.nan), reasons.copy(), np.nan
        if not why:
            meat = covariance.hac_meat(scores, lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V, Vsmall = inv@meat@inv.T/len(scores), inv_small@meat@inv_small.T/len(scores)
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0: why.append("invalid sandwich")
            else:
                se_difference = np.max(np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1))
                if se_difference >= se_step_tol: why.append("price SE step sensitivity")
        estimates[lag] = {"covariance": V if not why else np.full_like(A, np.nan), "ready": not why,
                          "reason": "; ".join(why) or "numerically ready", "price_SE_step_difference": se_difference}
    return {"Hessian": A, "Hessian_condition": condition, "Hessian_step_difference": sensitivity,
            "Hessian_min_eigenvalue": np.linalg.eigvalsh(A).min(), "distance_to_bounds": distance,
            "near_bound": bool(distance.min() < 1e-4), "weighted_scores": scores, "estimates": estimates}
