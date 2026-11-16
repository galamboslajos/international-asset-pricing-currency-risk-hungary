"""Three-equity constant-price baseline for regional notebook 21.

Pricing/filter functions are copied unchanged from notebook 03.
The optimizer adds a lower-candidate check; inference also reports fixed HAC sensitivities.
World, US and the national equity market; monthly log percentage points.
No data access or execution on import.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

PRICE_SCALE = 0.01
TRI = np.tril_indices(3)

def unpack(theta, scales, partial):
    k = 3 if partial else 1
    q = len(theta)
    prices = np.zeros(3)
    prices[:k] = PRICE_SCALE * theta[:k]
    dp = np.zeros((q, 3))
    dp[np.arange(k), np.arange(k)] = PRICE_SCALE
    L = np.zeros((3, 3))
    dL = np.zeros((q, 3, 3))
    for offset, (i, j) in enumerate(zip(*TRI)):
        index = k + offset
        L[i, j] = np.exp(theta[index]) if i == j else theta[index]
        dL[index, i, j] = L[i, j] if i == j else 1.0
    scaled_L = scales[:, None] * L
    scaled_dL = scales[None, :, None] * dL
    target = scaled_L @ scaled_L.T
    dtarget = scaled_dL @ scaled_L.T + scaled_L @ scaled_dL.transpose(0, 2, 1)
    omega, alpha, beta = softmax([0.0, theta[-2], theta[-1]])
    da, db = np.zeros(q), np.zeros(q)
    da[-2], da[-1] = alpha * (1 - alpha), -alpha * beta
    db[-2], db[-1] = -alpha * beta, beta * (1 - beta)
    return prices, dp, target, dtarget, omega, alpha, beta, da, db

def filter_model(theta, y, scales, partial):
    prices, dp, target, dtarget, omega, alpha, beta, da, db = unpack(theta, scales, partial)
    H, dH = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariance, means, residuals = np.empty((n, 3, 3)), np.empty_like(y), np.empty_like(y)
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError("Unstable covariance recursion")
        chol = np.linalg.cholesky(H)
        inverse = np.linalg.inv(H)
        mu = prices[0] * H[:, 0]
        dmu = prices[0] * dH[:, :, 0] + dp[:, 0, None] * H[None, :, 0]
        for asset in (1, 2):
            mu[asset] += prices[asset] * H[asset, asset]
            dmu[:, asset] += prices[asset] * dH[:, asset, asset] + dp[:, asset] * H[asset, asset]
        error, de = y[t] - mu, -dmu
        v = inverse @ error
        losses[t] = 0.5 * (3 * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + error @ v)
        scores[t] = 0.5 * np.einsum("ij,kij->k", inverse - np.outer(v, v), dH) + de @ v
        covariance[t], means[t], residuals[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None] * error[None, None, :] + error[None, :, None] * de[:, None, :]
        dH = (omega * dtarget - (da + db)[:, None, None] * target
              + da[:, None, None] * shock + alpha * dshock
              + db[:, None, None] * H + beta * dH)
        H = omega * target + alpha * shock + beta * H
    return {"losses": losses, "scores": scores, "H": covariance, "mu": means, "errors": residuals,
            "prices": prices, "target": target, "alpha": alpha, "beta": beta, "omega": omega}

def objective(theta, y, scales, partial):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = filter_model(theta, y, scales, partial)
        if not np.isfinite(result["losses"]).all() or not np.isfinite(result["scores"]).all():
            raise FloatingPointError("Non-finite objective")
        return result["losses"].mean(), result["scores"].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8 + theta @ theta, 2 * theta

def initial_theta(y, scales, partial, alpha=0.05, beta=0.9):
    covariance = np.cov(y, rowvar=False)
    normalised = covariance / np.outer(scales, scales)
    chol = np.linalg.cholesky(normalised)
    chol_params = [np.log(chol[i, j]) if i == j else chol[i, j] for i, j in zip(*TRI)]
    world_price = y[:, 0].mean() / covariance[0, 0]
    prices = [world_price]
    if partial:
        prices += [(y[:, i].mean() - world_price * covariance[i, 0]) / covariance[i, i] for i in (1, 2)]
    omega = 1 - alpha - beta
    return np.r_[np.array(prices) / PRICE_SCALE, chol_params, np.log(alpha / omega), np.log(beta / omega)]

def parameter_bounds(partial):
    k = 3 if partial else 1
    cov_bounds = [(-6.0, 3.0) if i == j else (-4.0, 4.0) for i, j in zip(*TRI)]
    return [(-25.0, 25.0)] * k + cov_bounds + [(-10.0, 10.0)] * 2

def fit_starts(y, scales, partial, starts, maxiter=600):
    runs, solutions = [], []
    for i, start in enumerate(starts, 1):
        result = minimize(objective, start, args=(y, scales, partial), method="L-BFGS-B", jac=True,
                          bounds=parameter_bounds(partial), options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40})
        gradient = np.max(np.abs(result.jac))
        print(f"  start {i}: success={result.success}, mean NLL={result.fun:.9f}, max|gradient|={gradient:.2e}, iterations={result.nit}", flush=True)
        runs.append({"start": i, "success": bool(result.success), "mean_nll": float(result.fun),
                     "max_abs_gradient": float(gradient), "iterations": int(result.nit), "message": str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7 and np.max(np.abs(r.jac)) < 1e-4]
    if not valid:
        raise RuntimeError("No converged fit with a small gradient. Inspect starting-point results and raise MAX_ITER.")
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun - 1e-7 for r in solutions):
        raise RuntimeError("A lower nonconverged candidate remains; inspect starts/MAX_ITER.")
    return best, runs

def numerical_gradient(function, theta, step=1e-5):
    gradient = np.empty_like(theta)
    for j in range(len(theta)):
        delta = step * (1 + abs(theta[j]))
        plus, minus = theta.copy(), theta.copy()
        plus[j] += delta
        minus[j] -= delta
        gradient[j] = (function(plus) - function(minus)) / (2 * delta)
    return gradient

def loss_hessian(theta, y, scales, partial, step=1e-4):
    result = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step * (1 + abs(theta[j]))
        plus, minus = theta.copy(), theta.copy()
        plus[j] += delta
        minus[j] -= delta
        result[:, j] = (objective(plus, y, scales, partial)[1] - objective(minus, y, scales, partial)[1]) / (2 * delta)
    return (result + result.T) / 2


def score_meat(scores, lag):
    """Centered score OPG (lag 0), or Bartlett Newey–West meat at a fixed lag."""
    s = scores - scores.mean(axis=0)
    meat = s.T @ s / len(s)
    for j in range(1, lag + 1):
        moment = s[j:].T @ s[:-j] / len(s)
        meat += (1 - j / (lag + 1)) * (moment + moment.T)
    return meat


def inference(theta, y, scales, partial, filtered, hac_lags=(0, 3, 6, 12)):
    """Suppress inference at bounds, unstable curvature or unstable price SEs."""
    A = loss_hessian(theta, y, scales, partial)
    small = loss_hessian(theta, y, scales, partial, step=5e-5)
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.max(np.abs(A-small)) / max(1., np.max(np.abs(small)))
    limits = np.asarray(parameter_bounds(partial))
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append("parameter bound")
    if minimum <= 0: reasons.append("nonpositive curvature")
    if condition >= 1e9: reasons.append("ill-conditioned curvature")
    if sensitivity >= 1e-3: reasons.append("Hessian step sensitivity")
    k, estimates = (3 if partial else 1), {}
    for lag in hac_lags:
        why, V, difference = reasons.copy(), np.full_like(A, np.nan), np.nan
        if not why:
            meat = score_meat(filtered["scores"], lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V = inv @ meat @ inv.T / len(y)
            Vsmall = inv_small @ meat @ inv_small.T / len(y)
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or not np.isfinite(Vsmall).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0:
                why.append("invalid sandwich")
            else:
                difference = np.max(np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1))
                if difference >= .01: why.append("price SE step sensitivity")
        estimates[lag] = {"covariance": V if not why else np.full_like(A, np.nan),
            "ready": not why, "reason": "; ".join(why) or "numerically ready",
            "price_SE_step_difference": difference}
    return {"Hessian": A, "Hessian_min_eigenvalue": minimum, "Hessian_condition": condition,
        "Hessian_step_difference": sensitivity, "min_distance_to_bounds": distance.min(),
        "estimates": estimates}
