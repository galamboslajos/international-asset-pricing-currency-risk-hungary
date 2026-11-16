"""Constant currency prices with corrected FX means for regional notebook 23.

Reuse the verified five-series priced-mean filter from pricing_robustness.py.
The no-FX model removes only the two FX prices, in both equity and FX means.
No data access, installation or kernel management occurs on import.
"""
import numpy as np
from scipy.optimize import minimize
import pricing_robustness as core
from regional_pricing_baseline import numerical_gradient, score_meat
from regional_covariance_sensitivity import projected_gradient

PRICE_SCALE = core.PRICE_SCALE
TRI = core.TRI


def expand(theta, with_fx):
    return np.asarray(theta) if with_fx else np.r_[theta[:3], 0., 0., theta[3:]]


def positions(full_length, with_fx):
    return np.arange(full_length) if with_fx else np.r_[np.arange(3), np.arange(5, full_length)]


def arguments(y, scales, covariance):
    return y, np.empty((len(y), 0)), {a: np.ones((len(y), 1)) for a in ['W', 'H', 'O']}, scales, 'priced', covariance


def filter_model(theta, y, scales, with_fx, covariance):
    full = expand(theta, with_fx)
    result = core.filter_model(full, *arguments(y, scales, covariance))
    result['scores'] = result['scores'][:, positions(len(full), with_fx)]
    return result


def objective(theta, y, scales, with_fx, covariance):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, scales, with_fx, covariance)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Nonfinite likelihood or gradient')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def initial_theta(y, scales, with_fx, covariance, alpha=.05, beta=.90):
    full = core.initial_theta(*arguments(y, scales, covariance), alpha=alpha, beta=beta)
    if not with_fx:
        sample = np.cov(y, rowvar=False)
        world = y[:, 0].mean()/sample[0, 0]
        local = [(y[:, i].mean()-world*sample[i, 0])/sample[i, i] for i in [1, 2]]
        full[:5] = np.r_[world, local, 0., 0.]/PRICE_SCALE
    return full[positions(len(full), with_fx)]


def bounds(with_fx, covariance):
    x, pdesign = np.empty((0, 0)), {a: np.ones((0, 1)) for a in ['W', 'H', 'O']}
    full = np.asarray(core.parameter_bounds(x, pdesign, covariance))
    return full[positions(len(full), with_fx)]


def expand_scalar(theta):
    return np.r_[theta[:-2], np.tile(theta[-2:], 5)]


def fit_starts(y, scales, with_fx, covariance, starts, maxiter=1800):
    limits = bounds(with_fx, covariance)
    args, runs, solutions = (y, scales, with_fx, covariance), [], []
    for i, start in enumerate(starts, 1):
        if not np.isfinite(start).all() or not ((start >= limits[:, 0]-1e-10) & (start <= limits[:, 1]+1e-10)).all():
            raise ValueError('Invalid starting point; inspect the moment start or model bounds.')
        result = minimize(objective, np.clip(start, limits[:, 0], limits[:, 1]), args=args,
            method='L-BFGS-B', jac=True, bounds=limits,
            options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-7, 'maxls': 60, 'maxcor': 50})
        projected = np.abs(projected_gradient(result.x, result.jac, limits)).max()
        refined = False
        if not result.success or projected >= 1e-4:
            retry = minimize(objective, result.x, args=args, method='L-BFGS-B', jac=True, bounds=limits,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7, 'maxls': 80, 'maxcor': 50})
            if retry.fun <= result.fun+1e-10:
                result, refined = retry, True
                projected = np.abs(projected_gradient(result.x, result.jac, limits)).max()
        raw = np.abs(result.jac).max()
        print(f'  start {i}: success={result.success}, NLL={result.fun:.9f}, raw={raw:.2e}, projected={projected:.2e}, iterations={result.nit}', flush=True)
        runs.append({'start': i, 'success': bool(result.success), 'mean_nll': float(result.fun),
            'max_abs_gradient': float(raw), 'max_projected_gradient': float(projected),
            'iterations': int(result.nit), 'refined': refined, 'message': str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7
        and np.abs(projected_gradient(r.x, r.jac, limits)).max() < 1e-4]
    if not valid: raise RuntimeError('No converged constrained fit with a small projected gradient.')
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError('A lower nonconverged candidate remains; inspect starts/MAX_ITER.')
    return best, runs


def loss_hessian(theta, y, scales, with_fx, covariance, step=1e-4):
    A = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j])); plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        A[:, j] = (objective(plus, y, scales, with_fx, covariance)[1]-objective(minus, y, scales, with_fx, covariance)[1])/(2*delta)
    return (A+A.T)/2


def inference(theta, y, scales, with_fx, covariance, filtered, hac_lags=(0, 3, 6, 12)):
    A = loss_hessian(theta, y, scales, with_fx, covariance)
    small = loss_hessian(theta, y, scales, with_fx, covariance, step=5e-5)
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.abs(A-small).max()/max(1., np.abs(small).max())
    limits = bounds(with_fx, covariance)
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append('parameter bound')
    if minimum <= 0: reasons.append('nonpositive curvature')
    if condition >= 1e9: reasons.append('ill-conditioned curvature')
    if sensitivity >= 1e-3: reasons.append('Hessian step sensitivity')
    k, estimates = (5 if with_fx else 3), {}
    for lag in hac_lags:
        why, V, difference = reasons.copy(), np.full_like(A, np.nan), np.nan
        if not why:
            meat = score_meat(filtered['scores'], lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V, Vsmall = inv@meat@inv.T/len(y), inv_small@meat@inv_small.T/len(y)
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or not np.isfinite(Vsmall).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0:
                why.append('invalid sandwich')
            else:
                difference = np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1).max()
                if difference >= .01: why.append('price SE step sensitivity')
        estimates[lag] = {'covariance': V if not why else np.full_like(A, np.nan),
            'ready': not why, 'reason': '; '.join(why) or 'numerically ready', 'price_SE_step_difference': difference}
    return {'Hessian': A, 'Hessian_min_eigenvalue': minimum, 'Hessian_condition': condition,
        'Hessian_step_difference': sensitivity, 'min_distance_to_bounds': distance.min(), 'estimates': estimates}
