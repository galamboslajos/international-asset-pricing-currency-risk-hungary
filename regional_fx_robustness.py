"""Calendar-preserving FX-only robustness for regional notebook 30.

Weights apply to joint likelihood terms AFTER the unchanged five-series filter.
All observations and their derivatives remain in the covariance recursion.
"""
import numpy as np
from scipy.optimize import minimize
import regional_local_pricing as pricing
from regional_pricing_baseline import score_meat
from regional_covariance_sensitivity import projected_gradient

PRICE_SCALE = pricing.PRICE_SCALE


def objective(theta, y, designs, scales, covariance, weights):
    if weights.shape != (len(y),) or not np.isin(weights, [0., 1.]).all() or weights.sum() == 0:
        raise ValueError('Use a nonempty, calendar-aligned zero/one weight vector.')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = pricing.filter_model(theta, y, designs, scales, True, covariance)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Nonfinite likelihood or score')
        return (weights*f['losses']).mean(), (weights[:, None]*f['scores']).mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def fit_starts(y, designs, scales, covariance, weights, starts, maxiter=2500):
    limits = pricing.bounds(designs, True, covariance)
    args, runs, solutions = (y, designs, scales, covariance, weights), [], []
    for i, start in enumerate(starts, 1):
        if not np.isfinite(start).all() or not ((start >= limits[:, 0]-1e-10) & (start <= limits[:, 1]+1e-10)).all():
            raise ValueError('Invalid starting point; inspect starts or bounds.')
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


def loss_hessian(theta, y, designs, scales, covariance, weights, step=1e-4):
    A = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j])); plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        A[:, j] = (objective(plus, y, designs, scales, covariance, weights)[1]-objective(minus, y, designs, scales, covariance, weights)[1])/(2*delta)
    return (A+A.T)/2


def inference(theta, y, designs, scales, covariance, weights, filtered, hac_lags=(0, 3, 6, 12), hessians=None):
    if hessians is None:
        A = loss_hessian(theta, y, designs, scales, covariance, weights)
        small = loss_hessian(theta, y, designs, scales, covariance, weights, step=5e-5)
    else:
        A, small = hessians
        if any(a.shape != (len(theta), len(theta)) or not np.isfinite(a).all() or
               not np.allclose(a, a.T, atol=1e-12, rtol=0) for a in (A, small)):
            raise ValueError('Invalid cached Hessians.')
    weighted_scores = weights[:, None]*filtered['scores']
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.abs(A-small).max()/max(1., np.abs(small).max())
    limits = pricing.bounds(designs, True, covariance)
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append('parameter bound')
    if minimum <= 0: reasons.append('nonpositive curvature')
    if condition >= 1e9: reasons.append('ill-conditioned curvature')
    if sensitivity >= 1e-3: reasons.append('Hessian step sensitivity')
    active = designs
    indices = pricing.price_indices(designs, True)
    k = 5+sum(d.shape[1]-1 for d in active.values())
    estimates = {}
    for lag in hac_lags:
        why, V, difference, path_difference = reasons.copy(), np.full_like(A, np.nan), np.nan, np.nan
        if not why:
            meat = score_meat(weighted_scores, lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V, Vsmall = inv@meat@inv.T/len(y), inv_small@meat@inv_small.T/len(y)
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or not np.isfinite(Vsmall).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0:
                why.append('invalid sandwich')
            else:
                difference = np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1).max()
                if difference >= .01: why.append('price SE step sensitivity')
                changes = []
                for label, design in active.items():
                    ids = indices[label]
                    v0 = np.einsum('ij,jk,ik->i', design, V[np.ix_(ids, ids)], design)
                    v1 = np.einsum('ij,jk,ik->i', design, Vsmall[np.ix_(ids, ids)], design)
                    if min(v0.min(), v1.min()) <= 0: why.append(f'invalid {label} path variance')
                    else: changes.append(np.abs(np.sqrt(v1/v0)-1).max())
                path_difference = max(changes) if changes else np.nan
                if changes and path_difference >= .01: why.append('price-path SE step sensitivity')
        estimates[lag] = {'covariance': V if not why else np.full_like(A, np.nan),
            'ready': not why, 'reason': '; '.join(why) or 'numerically ready',
            'price_SE_step_difference': difference, 'price_path_SE_step_difference': path_difference}
    return {'Hessian': A, 'Hessian_small': small, 'weighted_scores': weighted_scores, 'Hessian_min_eigenvalue': minimum, 'Hessian_condition': condition,
        'Hessian_step_difference': sensitivity, 'min_distance_to_bounds': distance.min(), 'estimates': estimates}
