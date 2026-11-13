"""Two-step covariance-targeting sensitivity for notebook 15.

Map targeted parameters into the frozen free-intercept engine. All means,
innovations and state updates use that engine; scores follow by chain rule.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import pricing_robustness as base
import covariance_robustness as covariance
import local_risk_pricing as model

TRI = base.TRI


def dynamics(theta, pdesign, name):
    k, count = model.layout(pdesign, name)
    groups = np.array(base.GROUPS['scalar' if name == 'scalar11' else 'grouped'])
    components = 2 if name == 'scalar11' else 3
    physical = softmax(np.c_[np.zeros(count//components), theta[k:].reshape(-1, components)], axis=1)[groups, 1:]
    terms, derivatives = [], []
    for c in range(components):
        v = np.sqrt(physical[:, c]); dv = np.zeros((len(theta), 5))
        for i in range(5):
            for j in range(components):
                dv[k+components*groups[i]+j, i] = .5*v[i]*((c == j)-physical[i, j])
        terms.append(np.outer(v, v))
        derivatives.append(dv[:, :, None]*v[None, None, :]+v[None, :, None]*dv[:, None, :])
    return np.array(terms), np.array(derivatives)


def intercept(theta, pdesign, name, target):
    terms, derivatives = dynamics(theta, pdesign, name)
    return target*(1-terms.sum(axis=0)), -target*derivatives.sum(axis=0)


def constraint(theta, pdesign, name, target, pd_floor=1e-8):
    omega, derivative = intercept(theta, pdesign, name, target)
    values, vectors = np.linalg.eigh(omega); v = vectors[:, 0]
    return values[0]-pd_floor, np.einsum('i,kij,j->k', v, derivative, v)


def free_map(theta, pdesign, name, target):
    """Target is a covariance in fixed, sample-SD-normalized units."""
    k, _ = model.layout(pdesign, name)
    omega, derivative = intercept(theta, pdesign, name, target)
    L = np.linalg.cholesky(omega)
    chol = np.array([np.log(L[i,j]) if i == j else L[i,j] for i,j in zip(*TRI)])
    full = np.r_[theta[:k], chol, theta[k:]]
    J = np.zeros((len(full), len(theta))); J[:k, :k] = np.eye(k); J[k+15:, k:] = np.eye(len(theta)-k)
    for j, d in enumerate(derivative):
        a = np.linalg.solve(L, d); a = np.linalg.solve(L, a.T).T
        a = np.tril(a); a[np.diag_indices(5)] *= .5
        dL = L@a
        J[k:k+15, j] = [dL[i,h]/L[i,h] if i == h else dL[i,h] for i,h in zip(*TRI)]
    return full, J


def filter_model(theta, y, pdesign, scales, name, target):
    full, J = free_map(theta, pdesign, name, target)
    f = model.filter_model(full, y, pdesign, scales, name)
    return {**f, 'scores_free': f['scores'], 'scores': f['scores']@J, 'free_theta': full, 'map_jacobian': J}


def objective(theta, y, pdesign, scales, name, target):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, pdesign, scales, name, target)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all(): raise FloatingPointError('Nonfinite target fit')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def bounds(pdesign, name):
    k, count = model.layout(pdesign, name)
    return np.array([[-100.,100.]]*k+[[-10.,10.]]*count)


def reduce_free(theta, pdesign, name):
    k, _ = model.layout(pdesign, name)
    return np.r_[theta[:k], theta[k+15:]]


def feasible_start(theta, pdesign, name, target, pd_floor=1e-8):
    out = np.clip(theta, *bounds(pdesign, name).T)
    if constraint(out, pdesign, name, target, pd_floor)[0] > 10*pd_floor: return out
    k, count = model.layout(pdesign, name); c = 2 if name == 'scalar11' else 3
    old = out[k:].copy(); common = np.tile(old.reshape(-1,c).mean(axis=0), count//c)
    for fraction in np.linspace(.95,0,20):
        out[k:] = fraction*old+(1-fraction)*common
        if constraint(out,pdesign,name,target,pd_floor)[0] > 10*pd_floor: return out
    raise ValueError('No positive-definite starting intercept')


def projected_gradient(theta, gradient, limits):
    out = gradient.copy()
    out[(theta-limits[:,0]<1e-6)&(gradient>0)] = 0
    out[(limits[:,1]-theta<1e-6)&(gradient<0)] = 0
    return out


def fit_starts(args, starts, maxiter=2800, pd_floor=1e-8):
    y, pdesign, scales, name, target = args
    limits = bounds(pdesign,name); runs, solutions = [], []
    con = {'type':'ineq', 'fun':lambda t:constraint(t,pdesign,name,target,pd_floor)[0],
           'jac':lambda t:constraint(t,pdesign,name,target,pd_floor)[1]}
    for i, start in enumerate(starts,1):
        start = feasible_start(start,pdesign,name,target,pd_floor)
        coarse = minimize(objective,start,args=args,method='BFGS',jac=True,options={'maxiter':maxiter,'gtol':1e-6})
        seed = feasible_start(coarse.x,pdesign,name,target,pd_floor)
        result = minimize(objective,seed,args=args,method='SLSQP',jac=True,bounds=limits,constraints=con,
                          options={'maxiter':maxiter,'ftol':1e-13})
        margin, cg = constraint(result.x,pdesign,name,target,pd_floor)
        multiplier = float(np.asarray(result.get('multipliers',[0.]))[0]) if margin < 1e-6 else 0.
        raw = np.max(np.abs(result.jac)); pg = np.max(np.abs(projected_gradient(result.x,result.jac-multiplier*cg,limits)))
        valid = bool(result.success and np.isfinite(result.fun) and result.fun<1e7 and margin>=-1e-10 and pg<1e-4)
        print(f'  start {i}: success={result.success}, NLL/T={result.fun:.9f}, raw/KKT gradient={raw:.2e}/{pg:.2e}, PD margin={margin:.2e}',flush=True)
        runs.append({'start':i,'success':bool(result.success),'valid':valid,'mean_nll':float(result.fun),'raw_gradient':raw,
                     'KKT_gradient':pg,'PD_margin':margin,'PD_multiplier':multiplier,'iterations':int(coarse.nit+result.nit),'message':str(result.message)})
        solutions.append(result)
    candidates = [r for r,log in zip(solutions,runs) if log['valid']]
    if not candidates: raise RuntimeError('No regular or constrained-converged targeted fit; inspect starts.')
    best = min(candidates,key=lambda r:r.fun)
    if any(np.isfinite(r.fun) and r.fun<best.fun-1e-7 for r in solutions): raise RuntimeError('Lower unconverged target fit remains.')
    return best, runs


def jacobian(function, theta, step=1e-4):
    result = np.empty((len(function(theta)),len(theta)))
    for j in range(len(theta)):
        d = step*(1+abs(theta[j])); plus,minus = theta.copy(),theta.copy();plus[j]+=d;minus[j]-=d
        result[:,j] = (function(plus)-function(minus))/(2*d)
    return result


def target_products(errors, scales):
    z = errors/scales
    return (z[:,:,None]*z[:,None,:])[:,TRI[0],TRI[1]]


def matrix_from_vech(v):
    out = np.zeros((5,5));out[TRI] = v;out[(TRI[1],TRI[0])] = v
    return out


def inference(theta, args, filtered, kind, first_theta, first_filter, hac_lags=(0,3,6,12), se_step_tol=.01, pd_floor=1e-8):
    """Stack first-stage QML, innovation moments and second-stage QML as applicable."""
    y,pdesign,scales,name,target = args; n=len(y); q=len(theta); k=model.layout(pdesign,name)[0]
    normal = target[TRI]; moment = normal-target_products(first_filter['errors'] if kind=='innovations' else y-y.mean(axis=0),scales)
    def pieces(step):
        A = jacobian(lambda t:objective(t,*args)[1],theta,step); A=(A+A.T)/2
        B = jacobian(lambda s:objective(theta,y,pdesign,scales,name,matrix_from_vech(s))[1],normal,step)
        if kind=='innovations':
            fa=(y,pdesign,scales,name)
            A0=jacobian(lambda t:model.objective(t,*fa)[1],first_theta,step);A0=(A0+A0.T)/2
            M=-jacobian(lambda t:target_products(model.filter_model(t,*fa)['errors'],scales).mean(axis=0),first_theta,step)
            q0=len(first_theta);D=np.zeros((q0+15+q,q0+15+q))
            D[:q0,:q0]=A0;D[q0:q0+15,:q0]=M;D[q0:q0+15,q0:q0+15]=np.eye(15)
            D[-q:,q0:q0+15]=B;D[-q:,-q:]=A
        else:
            A0=None;D=np.block([[np.eye(15),np.zeros((15,q))],[B,A]])
        return A,A0,D
    margin=constraint(theta,pdesign,name,target,pd_floor)[0];limits=bounds(pdesign,name)
    distance=np.minimum(theta-limits[:,0],limits[:,1]-theta)
    reasons=[]
    if distance.min()<1e-4:reasons.append('parameter bound')
    if margin<1e-6:reasons.append('target intercept PD boundary')
    if kind=='innovations':
        old_limits=model.bounds(pdesign,name)
        if np.minimum(first_theta-old_limits[:,0],old_limits[:,1]-first_theta).min()<1e-4:reasons.append('first-stage parameter bound')
    A,A0,D=pieces(1e-4);small,small0,Dsmall=pieces(5e-5)
    condition=np.linalg.cond(D)
    sensitivity=np.max(np.abs(D-Dsmall))/max(1.,np.max(np.abs(Dsmall)))
    hessians=[A,small]+([A0,small0] if A0 is not None else [])
    if min(np.linalg.eigvalsh(a).min() for a in hessians)<=0:reasons.append('nonpositive curvature')
    if condition>=1e9:reasons.append('ill-conditioned stacked Jacobian')
    if sensitivity>=1e-3:reasons.append('Jacobian step sensitivity')
    stacked=np.c_[first_filter['scores'],moment,filtered['scores']] if kind=='innovations' else np.c_[moment,filtered['scores']]
    estimates={}
    for lag in hac_lags:
        for method in ['target_adjusted','target_fixed']:
            why=reasons.copy();V=np.full((q,q),np.nan);diff=np.nan
            if not why:
                if method=='target_adjusted':
                    inv=np.linalg.inv(D)[-q:];inv_small=np.linalg.inv(Dsmall)[-q:];scores=stacked
                else:
                    inv=np.linalg.inv(A);inv_small=np.linalg.inv(small);scores=filtered['scores']
                meat=covariance.hac_meat(scores,lag)
                V=inv@meat@inv.T/n;Vs=inv_small@meat@inv_small.T/n;V=(V+V.T)/2;Vs=(Vs+Vs.T)/2
                if not np.isfinite(V).all() or min(np.diag(V).min(),np.diag(Vs).min())<=0:why.append('invalid sandwich')
                else:
                    diff=np.max(np.abs(np.sqrt(np.diag(Vs)[:k])/np.sqrt(np.diag(V)[:k])-1))
                    if diff>=se_step_tol:why.append('price SE step sensitivity')
            estimates[(method,lag)]={'covariance':V if not why else np.full((q,q),np.nan),'ready':not why,
                                     'reason':'; '.join(why) or 'numerically ready','price_SE_step_difference':diff}
    return {'Hessian':A,'stack_Jacobian':D,'stack_scores':stacked,'Jacobian_condition':condition,'Jacobian_step_difference':sensitivity,
            'PD_margin':margin,'near_bound':bool(distance.min()<1e-4 or margin<1e-6),'estimates':estimates}
