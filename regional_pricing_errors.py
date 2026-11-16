"""Joint regional equity pricing-error intercepts for notebook 27.

Scalar/diagonal BEKK and corrected FX means are preserved. Local designs may be
constant or time varying; only the three equity means receive free intercepts.
"""
import numpy as np
from scipy.optimize import minimize
import regional_local_pricing as previous
from regional_pricing_baseline import numerical_gradient, score_meat
from regional_covariance_sensitivity import projected_gradient

PRICE_SCALE, TRI = previous.PRICE_SCALE, previous.TRI


def alpha_indices(theta, free_alpha):
    return np.arange(len(theta)-3, len(theta)) if free_alpha else np.array([], dtype=int)


def expand_zero(theta):
    return np.r_[theta, np.zeros(3)]


def expand_scalar(theta, free_alpha):
    if free_alpha:
        return np.r_[previous.expand_scalar(theta[:-3]), theta[-3:]]
    return previous.expand_scalar(theta)


def price_indices(designs, free_alpha):
    return previous.price_indices(designs, True)


def filter_model(theta, y, designs, scales, free_alpha, covariance):
    if not free_alpha:
        f = previous.filter_model(theta, y, designs, scales, designs['L'].shape[1] > 1, covariance)
        return {**f, 'pricing_mu': f['mu'], 'equity_alphas': np.zeros(3)}
    omega, domega, aa, daa, bb, dbb, target, dtarget, alpha, beta = previous.covariance_terms(theta[:-3], designs, scales, covariance)
    def pad(d):
        return np.concatenate([d, np.zeros((3, *d.shape[1:]))], axis=0)
    domega, daa, dbb, dtarget = [pad(d) for d in [domega, daa, dbb, dtarget]]
    H, dH = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariances = np.empty((n, 5, 5))
    means, pricing_means, errors = np.empty_like(y), np.empty_like(y), np.empty_like(y)
    indices = price_indices(designs, True)
    prices, derivatives = {}, {}
    for label, design in designs.items():
        prices[label] = PRICE_SCALE*(design @ theta[indices[label]])
        derivatives[label] = np.zeros((n, q))
        derivatives[label][:, indices[label]] = PRICE_SCALE*design
    equity_alphas = scales[:3]*theta[-3:]
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError('Unstable covariance recursion')
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
        mu, dmu = np.zeros(5), np.zeros((q, 5))
        for label, factor in [('W', 0), ('H', 3), ('O', 4)]:
            value = prices[label][t]
            mu += value*H[:, factor]
            dmu += value*dH[:, :, factor]+derivatives[label][t, :, None]*H[None, :, factor]
        mu[1] += PRICE_SCALE*theta[1]*H[1, 1]
        dmu[:, 1] += PRICE_SCALE*theta[1]*dH[:, 1, 1]
        dmu[1, 1] += PRICE_SCALE*H[1, 1]
        mu[2] += prices['L'][t]*H[2, 2]
        dmu[:, 2] += prices['L'][t]*dH[:, 2, 2]+derivatives['L'][t]*H[2, 2]
        pricing_means[t] = mu
        mu[:3] += equity_alphas
        dmu[q-3+np.arange(3), np.arange(3)] += scales[:3]
        error, de = y[t]-mu, -dmu
        v = inverse @ error
        losses[t] = .5*(5*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error @ v)
        scores[t] = .5*np.einsum('ij,kij->k', inverse-np.outer(v, v), dH)+de @ v
        covariances[t], means[t], errors[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None]*error[None, None, :]+error[None, :, None]*de[:, None, :]
        dH = domega+daa*shock+aa*dshock+dbb*H+bb*dH
        H = omega+aa*shock+bb*H
    return {'losses': losses, 'scores': scores, 'H': covariances, 'mu': means, 'pricing_mu': pricing_means, 'errors': errors,
            'equity_alphas': equity_alphas, 'prices': PRICE_SCALE*theta[:5], 'world_prices': prices['W'],
            'fx_prices': np.c_[prices['H'], prices['O']], 'local_prices': prices['L'], 'intercept': omega, 'target': target,
            'alpha': alpha, 'beta': beta}


def objective(theta, y, designs, scales, free_alpha, covariance):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, designs, scales, free_alpha, covariance)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Nonfinite likelihood or gradient')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def initial_theta(y, designs, scales, free_alpha, covariance, alpha=.05, beta=.90):
    theta = previous.initial_theta(y, designs, scales, designs['L'].shape[1] > 1, covariance, alpha, beta)
    return expand_zero(theta) if free_alpha else theta


def bounds(designs, free_alpha, covariance):
    old = previous.bounds(designs, designs['L'].shape[1] > 1, covariance)
    return np.r_[old, np.tile([[-np.inf, np.inf]], (3, 1))] if free_alpha else old


def fit_starts(y, designs, scales, free_alpha, covariance, starts, maxiter=3000):
    limits = bounds(designs, free_alpha, covariance)
    args, runs, solutions = (y, designs, scales, free_alpha, covariance), [], []
    for i, start in enumerate(starts, 1):
        if not np.isfinite(start).all() or not ((start >= limits[:, 0]-1e-10) & (start <= limits[:, 1]+1e-10)).all():
            raise ValueError('Invalid starting point; inspect starts or bounds.')
        coarse = minimize(objective, start, args=args, method='BFGS', jac=True,
            options={'maxiter': min(1000, maxiter), 'gtol': 1e-6})
        seed = coarse.x if np.isfinite(coarse.x).all() and np.isfinite(coarse.fun) and coarse.fun < 1e7 else start
        violations = int(((seed < limits[:, 0]) | (seed > limits[:, 1])).sum())
        result = minimize(objective, np.clip(seed, limits[:, 0], limits[:, 1]), args=args,
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
            'iterations': int(result.nit), 'refined': refined, 'BFGS_success': bool(coarse.success),
            'BFGS_iterations': int(coarse.nit), 'BFGS_mean_nll': float(coarse.fun), 'BFGS_bound_violations': violations, 'message': str(result.message)})
        solutions.append(result)
    valid = [r for r in solutions if r.success and np.isfinite(r.fun) and r.fun < 1e7
        and np.abs(projected_gradient(r.x, r.jac, limits)).max() < 1e-4]
    if not valid: raise RuntimeError('No converged constrained fit with a small projected gradient.')
    best = min(valid, key=lambda r: r.fun)
    if any(np.isfinite(r.fun) and r.fun < best.fun-1e-7 for r in solutions):
        raise RuntimeError('A lower nonconverged candidate remains; inspect starts/MAX_ITER.')
    return best, runs


def loss_hessian(theta, y, designs, scales, free_alpha, covariance, step=1e-4):
    A = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j])); plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        A[:, j] = (objective(plus, y, designs, scales, free_alpha, covariance)[1]-objective(minus, y, designs, scales, free_alpha, covariance)[1])/(2*delta)
    return (A+A.T)/2


def inference(theta, y, designs, scales, free_alpha, covariance, filtered, hac_lags=(0, 3, 6, 12)):
    A = loss_hessian(theta, y, designs, scales, free_alpha, covariance)
    small = loss_hessian(theta, y, designs, scales, free_alpha, covariance, step=5e-5)
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.abs(A-small).max()/max(1., np.abs(small).max())
    limits = bounds(designs, free_alpha, covariance)
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append('parameter bound')
    if minimum <= 0: reasons.append('nonpositive curvature')
    if condition >= 1e9: reasons.append('ill-conditioned curvature')
    if sensitivity >= 1e-3: reasons.append('Hessian step sensitivity')
    active = designs
    indices = price_indices(designs, free_alpha)
    k = 5+sum(d.shape[1]-1 for d in active.values())
    mean_indices = np.r_[np.arange(k), alpha_indices(theta, free_alpha)]
    estimates = {}
    for lag in hac_lags:
        why, V, difference, path_difference, alpha_difference = reasons.copy(), np.full_like(A, np.nan), np.nan, np.nan, np.nan
        if not why:
            meat = score_meat(filtered['scores'], lag)
            inv, inv_small = np.linalg.inv(A), np.linalg.inv(small)
            V, Vsmall = inv@meat@inv.T/len(y), inv_small@meat@inv_small.T/len(y)
            V, Vsmall = (V+V.T)/2, (Vsmall+Vsmall.T)/2
            if not np.isfinite(V).all() or not np.isfinite(Vsmall).all() or min(np.diag(V).min(), np.diag(Vsmall).min()) <= 0:
                why.append('invalid sandwich')
            else:
                difference = np.abs(np.sqrt(np.diag(Vsmall)[mean_indices])/np.sqrt(np.diag(V)[mean_indices])-1).max()
                if difference >= .01: why.append('mean-coefficient SE step sensitivity')
                if free_alpha:
                    a = alpha_indices(theta, True)
                    alpha_difference = np.abs(np.sqrt(np.diag(Vsmall)[a])/np.sqrt(np.diag(V)[a])-1).max()
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
            'mean_SE_step_difference': difference, 'alpha_SE_step_difference': alpha_difference, 'price_path_SE_step_difference': path_difference}
    return {'Hessian': A, 'Hessian_min_eigenvalue': minimum, 'Hessian_condition': condition,
        'Hessian_step_difference': sensitivity, 'min_distance_to_bounds': distance.min(), 'estimates': estimates}
