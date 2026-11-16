"""Time-varying world prices with corrected FX means for regional notebook 24.

Reuse pricing_robustness.py with a varying world design and constant FX designs.
No data access, installation or kernel management occurs on import.
"""
import numpy as np
from scipy.optimize import minimize
import pricing_robustness as core
from regional_pricing_baseline import numerical_gradient, score_meat
from regional_covariance_sensitivity import projected_gradient

PRICE_SCALE = core.PRICE_SCALE
TRI = core.TRI


def arguments(y, z, scales, varying, covariance):
    ones = np.ones((len(y), 1))
    return y, np.empty((len(y), 0)), {'W': z if varying else ones, 'H': ones, 'O': ones}, scales, 'priced', covariance


def world_indices(z, varying):
    return np.r_[0, np.arange(5, 5+z.shape[1]-1)] if varying else np.array([0])


def lift_constant(theta, z):
    return np.r_[theta[:5], np.zeros(z.shape[1]-1), theta[5:]]


def expand_scalar(theta):
    return np.r_[theta[:-2], np.tile(theta[-2:], 5)]


def filter_model(theta, y, z, scales, varying, covariance):
    return core.filter_model(theta, *arguments(y, z, scales, varying, covariance))


def objective(theta, y, z, scales, varying, covariance):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, z, scales, varying, covariance)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Nonfinite likelihood or gradient')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def initial_theta(y, z, scales, varying, covariance, alpha=.05, beta=.90):
    return core.initial_theta(*arguments(y, z, scales, varying, covariance), alpha=alpha, beta=beta)


def bounds(z, varying, covariance):
    x, pdesign = np.empty((0, 0)), {'W': np.ones((0, z.shape[1] if varying else 1)), 'H': np.ones((0, 1)), 'O': np.ones((0, 1))}
    return np.asarray(core.parameter_bounds(x, pdesign, covariance))


def fit_starts(y, z, scales, varying, covariance, starts, maxiter=2000):
    limits = bounds(z, varying, covariance)
    args, runs, solutions = (y, z, scales, varying, covariance), [], []
    for i, start in enumerate(starts, 1):
        if not np.isfinite(start).all() or not ((start >= limits[:, 0]-1e-10) & (start <= limits[:, 1]+1e-10)).all():
            raise ValueError('Invalid starting point; inspect starts or model bounds.')
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


def loss_hessian(theta, y, z, scales, varying, covariance, step=1e-4):
    A = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j])); plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        A[:, j] = (objective(plus, y, z, scales, varying, covariance)[1]-objective(minus, y, z, scales, varying, covariance)[1])/(2*delta)
    return (A+A.T)/2


def inference(theta, y, z, scales, varying, covariance, filtered, hac_lags=(0, 3, 6, 12)):
    A = loss_hessian(theta, y, z, scales, varying, covariance)
    small = loss_hessian(theta, y, z, scales, varying, covariance, step=5e-5)
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.abs(A-small).max()/max(1., np.abs(small).max())
    limits = bounds(z, varying, covariance)
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append('parameter bound')
    if minimum <= 0: reasons.append('nonpositive curvature')
    if condition >= 1e9: reasons.append('ill-conditioned curvature')
    if sensitivity >= 1e-3: reasons.append('Hessian step sensitivity')
    k, estimates = 5+(z.shape[1]-1 if varying else 0), {}
    indices = world_indices(z, varying)
    design = z if varying else np.ones((len(y), 1))
    for lag in hac_lags:
        why, V, difference, path_difference = reasons.copy(), np.full_like(A, np.nan), np.nan, np.nan
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
                variances = [np.einsum('ij,jk,ik->i', design, v[np.ix_(indices, indices)], design) for v in [V, Vsmall]]
                if min(variances[0].min(), variances[1].min()) <= 0:
                    why.append('invalid world-path variance')
                else:
                    path_difference = np.abs(np.sqrt(variances[1]/variances[0])-1).max()
                    if path_difference >= .01: why.append('world-path SE step sensitivity')
        estimates[lag] = {'covariance': V if not why else np.full_like(A, np.nan),
            'ready': not why, 'reason': '; '.join(why) or 'numerically ready',
            'price_SE_step_difference': difference, 'world_path_SE_step_difference': path_difference}
    return {'Hessian': A, 'Hessian_min_eigenvalue': minimum, 'Hessian_condition': condition,
        'Hessian_step_difference': sensitivity, 'min_distance_to_bounds': distance.min(), 'estimates': estimates}
