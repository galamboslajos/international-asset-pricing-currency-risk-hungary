"""National local-price variation for regional notebook 26 (scalar/diagonal BEKK).

Keep world/FX designs fixed; add local slopes only in the national equity mean.
The existing priced-mean/covariance map is reused without modifying old engines.
"""
import numpy as np
from scipy.optimize import minimize
import pricing_robustness as core
import regional_fx_price_variation as previous
from regional_pricing_baseline import numerical_gradient, score_meat
from regional_covariance_sensitivity import projected_gradient

PRICE_SCALE, TRI = core.PRICE_SCALE, core.TRI
expand_scalar = previous.expand_scalar


def active_designs(designs, varying_local):
    return {**{a: designs[a] for a in ['W', 'H', 'O']},
            'L': designs['L'] if varying_local else np.ones((len(designs['W']), 1))}


def price_indices(designs, varying_local):
    active = active_designs(designs, varying_local)
    start = 5+sum(active[a].shape[1]-1 for a in ['W', 'H', 'O'])
    return {**core.price_indices(active), 'L': np.r_[2, np.arange(start, start+active['L'].shape[1]-1)]}


def lift_constant_local(theta, designs):
    start = 5+sum(designs[a].shape[1]-1 for a in ['W', 'H', 'O'])
    return np.r_[theta[:start], np.zeros(designs['L'].shape[1]-1), theta[start:]]


def covariance_terms(theta, designs, scales, covariance):
    old = {a: designs[a] for a in ['W', 'H', 'O']}
    start = 5+sum(old[a].shape[1]-1 for a in old)
    stop = start+designs['L'].shape[1]-1
    positions = np.r_[np.arange(start), np.arange(stop, len(theta))]
    v = core.unpack(theta[positions], scales, np.empty((0, 0)), old, covariance)
    def lift(derivative):
        result = np.zeros((len(theta), *derivative.shape[1:]))
        result[positions] = derivative
        return result
    return v[3], lift(v[4]), v[5], lift(v[6]), v[7], lift(v[8]), v[9], lift(v[10]), v[11], v[12]


def filter_model(theta, y, designs, scales, varying_local, covariance):
    if not varying_local:
        result = previous.filter_model(theta, y, designs, scales, True, covariance)
        return {**result, 'local_prices': np.full(len(y), result['prices'][2])}
    omega, domega, aa, daa, bb, dbb, target, dtarget, alpha, beta = covariance_terms(theta, designs, scales, covariance)
    H, dH = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    losses, scores = np.empty(n), np.empty((n, q))
    covariances, means, errors = np.empty((n, 5, 5)), np.empty_like(y), np.empty_like(y)
    indices = price_indices(designs, True)
    prices, derivatives = {}, {}
    for label, design in designs.items():
        prices[label] = PRICE_SCALE*(design @ theta[indices[label]])
        derivatives[label] = np.zeros((n, q))
        derivatives[label][:, indices[label]] = PRICE_SCALE*design
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
        error, de = y[t]-mu, -dmu
        v = inverse @ error
        losses[t] = .5*(5*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error @ v)
        scores[t] = .5*np.einsum('ij,kij->k', inverse-np.outer(v, v), dH)+de @ v
        covariances[t], means[t], errors[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None]*error[None, None, :]+error[None, :, None]*de[:, None, :]
        dH = domega+daa*shock+aa*dshock+dbb*H+bb*dH
        H = omega+aa*shock+bb*H
    return {'losses': losses, 'scores': scores, 'H': covariances, 'mu': means, 'errors': errors,
            'prices': PRICE_SCALE*theta[:5], 'world_prices': prices['W'], 'fx_prices': np.c_[prices['H'], prices['O']],
            'local_prices': prices['L'], 'intercept': omega, 'target': target, 'alpha': alpha, 'beta': beta}


def objective(theta, y, designs, scales, varying_local, covariance):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, designs, scales, varying_local, covariance)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Nonfinite likelihood or gradient')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def initial_theta(y, designs, scales, varying_local, covariance, alpha=.05, beta=.90):
    theta = previous.initial_theta(y, designs, scales, True, covariance, alpha, beta)
    return lift_constant_local(theta, designs) if varying_local else theta


def bounds(designs, varying_local, covariance):
    old = previous.bounds(designs, True, covariance)
    start = 5+sum(designs[a].shape[1]-1 for a in ['W', 'H', 'O'])
    added = designs['L'].shape[1]-1 if varying_local else 0
    return np.r_[old[:start], np.tile([[-100., 100.]], (added, 1)), old[start:]]


def fit_starts(y, designs, scales, varying_local, covariance, starts, maxiter=2500):
    limits = bounds(designs, varying_local, covariance)
    args, runs, solutions = (y, designs, scales, varying_local, covariance), [], []
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


def loss_hessian(theta, y, designs, scales, varying_local, covariance, step=1e-4):
    A = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        delta = step*(1+abs(theta[j])); plus, minus = theta.copy(), theta.copy()
        plus[j] += delta; minus[j] -= delta
        A[:, j] = (objective(plus, y, designs, scales, varying_local, covariance)[1]-objective(minus, y, designs, scales, varying_local, covariance)[1])/(2*delta)
    return (A+A.T)/2


def inference(theta, y, designs, scales, varying_local, covariance, filtered, hac_lags=(0, 3, 6, 12)):
    A = loss_hessian(theta, y, designs, scales, varying_local, covariance)
    small = loss_hessian(theta, y, designs, scales, varying_local, covariance, step=5e-5)
    minimum = min(np.linalg.eigvalsh(A).min(), np.linalg.eigvalsh(small).min())
    condition = max(np.linalg.cond(A), np.linalg.cond(small))
    sensitivity = np.abs(A-small).max()/max(1., np.abs(small).max())
    limits = bounds(designs, varying_local, covariance)
    distance = np.minimum(theta-limits[:, 0], limits[:, 1]-theta)
    reasons = []
    if distance.min() < 1e-4: reasons.append('parameter bound')
    if minimum <= 0: reasons.append('nonpositive curvature')
    if condition >= 1e9: reasons.append('ill-conditioned curvature')
    if sensitivity >= 1e-3: reasons.append('Hessian step sensitivity')
    active = active_designs(designs, varying_local)
    indices = price_indices(designs, varying_local)
    k = 5+sum(d.shape[1]-1 for d in active.values())
    estimates = {}
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
    return {'Hessian': A, 'Hessian_min_eigenvalue': minimum, 'Hessian_condition': condition,
        'Hessian_step_difference': sensitivity, 'min_distance_to_bounds': distance.min(), 'estimates': estimates}
