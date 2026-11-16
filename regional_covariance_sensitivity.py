"""Three-equity diagonal BEKK sensitivity for regional notebook 22.

Pricing, covariance and derivative functions match notebook 04 unchanged.
Inference matches notebook 21; old engines and research outputs remain unchanged.
No fetching, data writes or kernel management.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from regional_pricing_baseline import PRICE_SCALE, TRI, numerical_gradient, score_meat


def projected_gradient(theta, gradient, limits):
    """KKT gradient for the box constraints; retain the raw score separately."""
    g = gradient.copy()
    g[(theta <= limits[:, 0]+1e-8) & (g > 0)] = 0.
    g[(theta >= limits[:, 1]-1e-8) & (g < 0)] = 0.
    return g


def fit_starts(y, scales, partial, starts, maxiter=1600):
    limits = np.asarray(parameter_bounds(partial))
    args, runs, solutions = (y, scales, partial), [], []
    for i, start in enumerate(starts, 1):
        if not ((start >= limits[:, 0]-1e-10) & (start <= limits[:, 1]+1e-10)).all():
            raise ValueError("A requested starting point lies outside the model bounds.")
        result = minimize(objective, np.clip(start, limits[:, 0], limits[:, 1]), args=args,
            method="L-BFGS-B", jac=True, bounds=limits,
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50, "maxcor": 50})
        projected = np.max(np.abs(projected_gradient(result.x, result.jac, limits)))
        refined = False
        if not result.success or projected >= 1e-4:
            retry = minimize(objective, result.x, args=args, method="L-BFGS-B", jac=True, bounds=limits,
                options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-7, "maxls": 80, "maxcor": 50})
            if retry.fun <= result.fun+1e-10:
                result, refined = retry, True
                projected = np.max(np.abs(projected_gradient(result.x, result.jac, limits)))
        raw = np.max(np.abs(result.jac))
        print(f"  start {i}: success={result.success}, NLL={result.fun:.9f}, raw={raw:.2e}, projected={projected:.2e}, iterations={result.nit}", flush=True)
        runs.append({"start": i, "success": bool(result.success), "mean_nll": float(result.fun),
            "max_abs_gradient": float(raw), "max_projected_gradient": float(projected),
            "iterations": int(result.nit), "refined": refined, "message": str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7
        and np.max(np.abs(projected_gradient(r.x, r.jac, limits))) < 1e-4]
    if not valid: raise RuntimeError("No converged constrained fit with a small projected gradient; inspect starts.")
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError("A lower nonconverged candidate remains; inspect starts/MAX_ITER.")
    return best, runs

def unpack(theta, scales, partial):
    k, q = (3 if partial else 1), len(theta)
    prices, dp = np.zeros(3), np.zeros((q, 3))
    prices[:k] = PRICE_SCALE * theta[:k]
    dp[np.arange(k), np.arange(k)] = PRICE_SCALE
    L, dL = np.zeros((3, 3)), np.zeros((q, 3, 3))
    for offset, (i, j) in enumerate(zip(*TRI)):
        index = k + offset
        L[i, j] = np.exp(theta[index]) if i == j else theta[index]
        dL[index, i, j] = L[i, j] if i == j else 1.0
    C, dC = scales[:, None] * L, scales[None, :, None] * dL
    intercept = C @ C.T
    dintercept = dC @ C.T + C @ dC.transpose(0, 2, 1)
    weights = softmax(np.c_[np.zeros(3), theta[-6:].reshape(3, 2)], axis=1)
    alpha, beta = weights[:, 1], weights[:, 2]
    da, db = np.zeros((q, 3)), np.zeros((q, 3))
    for i in range(3):
        u, v = k + 6 + 2 * i, k + 7 + 2 * i
        da[u, i], da[v, i] = alpha[i] * (1 - alpha[i]), -alpha[i] * beta[i]
        db[u, i], db[v, i] = -alpha[i] * beta[i], beta[i] * (1 - beta[i])
    a, b = np.sqrt(alpha), np.sqrt(beta)
    da, db = da / (2 * a), db / (2 * b)
    aa, bb = np.outer(a, a), np.outer(b, b)
    daa = da[:, :, None] * a[None, None, :] + a[None, :, None] * da[:, None, :]
    dbb = db[:, :, None] * b[None, None, :] + b[None, :, None] * db[:, None, :]
    denominator = 1 - aa - bb
    target = intercept / denominator
    dtarget = dintercept / denominator + intercept * (daa + dbb) / denominator**2
    return prices, dp, intercept, dintercept, aa, daa, bb, dbb, target, dtarget, alpha, beta

def filter_model(theta, y, scales, partial):
    prices, dp, intercept, dintercept, aa, daa, bb, dbb, target, dtarget, alpha, beta = unpack(theta, scales, partial)
    H, dH = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariance, means, residuals = np.empty((n, 3, 3)), np.empty_like(y), np.empty_like(y)
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError("Unstable covariance recursion")
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
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
        dH = dintercept + daa * shock + aa * dshock + dbb * H + bb * dH
        H = intercept + aa * shock + bb * H
    return {"losses": losses, "scores": scores, "H": covariance, "mu": means, "errors": residuals,
            "prices": prices, "target": target, "intercept": intercept, "alpha": alpha, "beta": beta}

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
    alpha, beta = np.broadcast_to(alpha, 3), np.broadcast_to(beta, 3)
    remaining = 1 - alpha - beta
    covariance = np.cov(y, rowvar=False)
    intercept = covariance * np.sqrt(np.outer(remaining, remaining))
    chol = np.linalg.cholesky(intercept / np.outer(scales, scales))
    chol_params = [np.log(chol[i, j]) if i == j else chol[i, j] for i, j in zip(*TRI)]
    world_price = y[:, 0].mean() / covariance[0, 0]
    prices = [world_price]
    if partial:
        prices += [(y[:, i].mean() - world_price * covariance[i, 0]) / covariance[i, i] for i in (1, 2)]
    dynamics = np.c_[np.log(alpha / remaining), np.log(beta / remaining)].ravel()
    return np.r_[np.array(prices) / PRICE_SCALE, chol_params, dynamics]

def scalar_to_diagonal(theta, partial):
    k = 3 if partial else 1
    remaining = softmax([0.0, theta[-2], theta[-1]])[0]
    converted = theta[:-2].copy()
    for offset, (i, j) in enumerate(zip(*TRI)):
        converted[k + offset] += 0.5 * np.log(remaining) if i == j else 0.0
        if i != j:
            converted[k + offset] *= np.sqrt(remaining)
    return np.r_[converted, np.tile(theta[-2:], 3)]

def parameter_bounds(partial):
    k = 3 if partial else 1
    cov_bounds = [(-6.0, 3.0) if i == j else (-4.0, 4.0) for i, j in zip(*TRI)]
    return [(-25.0, 25.0)] * k + cov_bounds + [(-10.0, 10.0)] * 6

def loss_hessian(theta, y, scales, partial, step=1e-4):
    result = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step * (1 + abs(theta[j]))
        plus, minus = theta.copy(), theta.copy()
        plus[j] += delta
        minus[j] -= delta
        result[:, j] = (objective(plus, y, scales, partial)[1] - objective(minus, y, scales, partial)[1]) / (2 * delta)
    return (result + result.T) / 2

def dynamic_restrictions(theta, scales, partial):
    values = unpack(theta, scales, partial)
    alpha, beta = values[-2:]
    restriction = np.r_[alpha[1:] - alpha[0], beta[1:] - beta[0]]
    # Analytic derivatives of the physical alpha/beta parameters (not their square roots).
    k, q = (3 if partial else 1), len(theta)
    derivatives = np.zeros((2, 3, q))
    for i in range(3):
        u, v = k + 6 + 2 * i, k + 7 + 2 * i
        derivatives[0, i, u], derivatives[0, i, v] = alpha[i] * (1 - alpha[i]), -alpha[i] * beta[i]
        derivatives[1, i, u], derivatives[1, i, v] = -alpha[i] * beta[i], beta[i] * (1 - beta[i])
    jacobian = np.concatenate([d[1:] - d[0] for d in derivatives], axis=0)
    return restriction, jacobian

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
