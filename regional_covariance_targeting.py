"""Two-step covariance-targeting sensitivity for regional notebook 31.

Map targeted parameters into the frozen free-intercept engine. All means,
innovations and state updates use that engine; scores follow by chain rule.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import pricing_robustness as base
from regional_pricing_baseline import score_meat
import regional_local_pricing as model

TRI = base.TRI


def layout(pdesign, name):
    return 5+sum(d.shape[1]-1 for d in pdesign.values()), (2 if name == 'scalar' else 10)


def dynamics(theta, pdesign, name):
    k, count = layout(pdesign, name)
    groups = np.zeros(5, dtype=int) if name == 'scalar' else np.arange(5)
    components = 2
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


def eigen_constraints(theta, pdesign, name, target, pd_floor=1e-8):
    omega, derivative = intercept(theta, pdesign, name, target)
    values, vectors = np.linalg.eigh(omega)
    return values-pd_floor, np.einsum('ij,kil,lj->jk', vectors, derivative, vectors)


def kkt_certificate(theta, gradient, pdesign, name, target, pd_floor=1e-8, dual=None):
    """PSD-cone dual: a single minimum-eigenvalue multiplier misses corners.

    A repeated active eigenvalue allows a full PSD dual on its eigenspace.
    Recover its least-squares certificate; validate saved duals without optimizing.
    """
    omega, derivative = intercept(theta,pdesign,name,target)
    shifted = omega-pd_floor*np.eye(5)
    values, vectors = np.linalg.eigh(shifted)
    limits = bounds(pdesign,name)
    if dual is None:
        Q = vectors[:,values<1e-6]; rank = Q.shape[1]
        dual = np.zeros((5,5))
        if rank:
            tri = np.tril_indices(rank)
            projected = np.einsum('ia,kij,jb->kab',Q,derivative,Q)
            X = np.column_stack([projected[:,i,j]*(1 if i==j else 2) for i,j in zip(*tri)])
            free = (theta>limits[:,0]+1e-6)&(theta<limits[:,1]-1e-6)
            m = np.linalg.lstsq(X[free],gradient[free],rcond=None)[0]
            def matrix(v):
                M=np.zeros((rank,rank));M[tri]=v[:len(tri[0])];M[(tri[1],tri[0])]=v[:len(tri[0])];return M
            M=matrix(m);dual=Q@M@Q.T
            residual=projected_gradient(theta,gradient-X@m,limits)
            if np.linalg.eigvalsh(M).min()<-1e-8 or np.abs(residual).max()>=1e-4:
                lower=np.flatnonzero(theta<=limits[:,0]+1e-6);upper=np.flatnonzero(theta>=limits[:,1]-1e-6)
                Z=np.c_[X,np.eye(len(theta))[:,lower],-np.eye(len(theta))[:,upper]]
                ev,V=np.linalg.eigh(M);M=(V*np.maximum(ev,1e-10))@V.T
                seed=np.r_[M[tri],np.zeros(len(lower)+len(upper))]
                def eigjac(v):
                    _,E=np.linalg.eigh(matrix(v));a=E[:,0]
                    return np.r_[[a[i]*a[j]*(1 if i==j else 2) for i,j in zip(*tri)],np.zeros(len(lower)+len(upper))]
                result=minimize(lambda v:(.5*np.sum((Z@v-gradient)**2),Z.T@(Z@v-gradient)),seed,jac=True,method='SLSQP',
                    bounds=[(None,None)]*len(tri[0])+[(0,None)]*(len(lower)+len(upper)),
                    constraints={'type':'ineq','fun':lambda v:np.linalg.eigvalsh(matrix(v)).min(),'jac':eigjac},
                    options={'maxiter':500,'ftol':1e-18})
                dual=Q@matrix(result.x)@Q.T
    dual=np.asarray(dual)
    if dual.shape!=(5,5) or not np.isfinite(dual).all():raise ValueError('Invalid PSD dual certificate')
    stationarity=gradient-np.einsum('ij,kij->k',dual,derivative)
    return {'dual':dual,'KKT_gradient':float(np.abs(projected_gradient(theta,stationarity,limits)).max()),
        'PD_margin':float(values.min()),'dual_min_eigenvalue':float(np.linalg.eigvalsh((dual+dual.T)/2).min()),
        'complementarity':float(abs(np.trace(dual@shifted)))}


def free_map(theta, pdesign, name, target):
    """Target is a covariance in fixed, sample-SD-normalized units."""
    k, _ = layout(pdesign, name)
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
    f = model.filter_model(full, y, pdesign, scales, True, name)
    return {**f, 'scores_free': f['scores'], 'scores': f['scores']@J, 'free_theta': full, 'map_jacobian': J}


def direct_filter(theta, y, designs, scales, name, target):
    """Evaluate targeted states directly, including trial steps outside the PSD cone.

    SLSQP may evaluate infeasible trial intercepts. Their conditional covariances
    can remain PD, so the likelihood has a smooth extension there. Factoring the
    intercept itself creates an artificial discontinuity at zero and prevents
    the optimizer from returning to its strictly positive feasible boundary.
    Accepted estimates still satisfy the original normalized intercept floor.
    This recursion and its derivatives are checked against the mapped engine.
    """
    normal, dnormal = intercept(theta, designs, name, target)
    unit = np.outer(scales, scales)
    omega, domega = normal*unit, dnormal*unit
    terms, dterms = dynamics(theta, designs, name)
    aa, bb = terms; daa, dbb = dterms
    H = target*unit; n, q = len(y), len(theta)
    dH = np.zeros((q,5,5))
    losses, scores = np.empty(n), np.empty((n,q))
    covariances, means, errors = np.empty((n,5,5)), np.empty_like(y), np.empty_like(y)
    indices = model.price_indices(designs, True)
    prices, derivatives = {}, {}
    for label, design in designs.items():
        prices[label] = model.PRICE_SCALE*(design@theta[indices[label]])
        derivatives[label] = np.zeros((n,q))
        derivatives[label][:,indices[label]] = model.PRICE_SCALE*design
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H))>1e8:
            raise FloatingPointError('Unstable targeted covariance recursion')
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
        mu, dmu = np.zeros(5), np.zeros((q,5))
        for label, factor in [('W',0),('H',3),('O',4)]:
            value = prices[label][t]
            mu += value*H[:,factor]
            dmu += value*dH[:,:,factor]+derivatives[label][t,:,None]*H[None,:,factor]
        mu[1] += model.PRICE_SCALE*theta[1]*H[1,1]
        dmu[:,1] += model.PRICE_SCALE*theta[1]*dH[:,1,1]
        dmu[1,1] += model.PRICE_SCALE*H[1,1]
        mu[2] += prices['L'][t]*H[2,2]
        dmu[:,2] += prices['L'][t]*dH[:,2,2]+derivatives['L'][t]*H[2,2]
        error, de = y[t]-mu, -dmu
        v = inverse@error
        losses[t] = .5*(5*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error@v)
        scores[t] = .5*np.einsum('ij,kij->k',inverse-np.outer(v,v),dH)+de@v
        covariances[t],means[t],errors[t] = H,mu,error
        shock = np.outer(error,error)
        dshock = de[:,:,None]*error[None,None,:]+error[None,:,None]*de[:,None,:]
        dH = domega+daa*shock+aa*dshock+dbb*H+bb*dH
        H = omega+aa*shock+bb*H
    return {'losses':losses,'scores':scores,'H':covariances,'mu':means,'errors':errors}


def objective(theta, y, pdesign, scales, name, target):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = direct_filter(theta, y, pdesign, scales, name, target)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all(): raise FloatingPointError('Nonfinite target fit')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8+theta@theta, 2*theta


def bounds(pdesign, name):
    k, count = layout(pdesign, name)
    return np.array([[-100.,100.]]*k+[[-10.,10.]]*count)


def reduce_free(theta, pdesign, name):
    k, _ = layout(pdesign, name)
    return np.r_[theta[:k], theta[k+15:]]


def feasible_start(theta, pdesign, name, target, pd_floor=1e-8):
    out = np.clip(theta, *bounds(pdesign, name).T)
    if constraint(out, pdesign, name, target, pd_floor)[0] > 10*pd_floor: return out
    k, count = layout(pdesign, name); c = 2
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


def fit_starts(args, starts, maxiter=3000, pd_floor=1e-8):
    y,pdesign,scales,name,target=args;limits=bounds(pdesign,name);runs,solutions=[],[]
    con={'type':'ineq','fun':lambda t:eigen_constraints(t,pdesign,name,target,pd_floor)[0],
        'jac':lambda t:eigen_constraints(t,pdesign,name,target,pd_floor)[1]}
    for i,start in enumerate(starts,1):
        start=feasible_start(start,pdesign,name,target,pd_floor)
        for attempt in ['direct','warm retry']:
            seed=start
            if attempt=='warm retry':
                coarse=minimize(objective,start,args=args,method='BFGS',jac=True,options={'maxiter':min(maxiter,1000),'gtol':1e-6})
                seed=feasible_start(coarse.x,pdesign,name,target,pd_floor)
            result=minimize(objective,seed,args=args,method='SLSQP',jac=True,bounds=limits,constraints=con,options={'maxiter':maxiter,'ftol':1e-13})
            value,g=objective(result.x,*args);cert=kkt_certificate(result.x,g,pdesign,name,target,pd_floor)
            feasible=bool(cert['PD_margin']>=-1e-10 and ((result.x>=limits[:,0]-1e-10)&(result.x<=limits[:,1]+1e-10)).all())
            valid=bool(result.success and np.isfinite(value) and value<1e7 and feasible and cert['KKT_gradient']<1e-4
                and cert['dual_min_eigenvalue']>=-1e-8 and cert['complementarity']<1e-6)
            result['PD_dual_matrix']=cert['dual'];result.fun=value
            print(f"  start {i} {attempt}: success={result.success}, NLL/T={value:.9f}, KKT={cert['KKT_gradient']:.2e}, PD margin={cert['PD_margin']:.2e}",flush=True)
            runs.append({'start':i,'attempt':attempt,'success':bool(result.success),'valid':valid,'feasible':feasible,'mean_nll':float(value),
                'raw_gradient':float(np.abs(g).max()),'KKT_gradient':cert['KKT_gradient'],'PD_margin':cert['PD_margin'],
                'PD_multiplier':float(np.trace(cert['dual'])),'dual_min_eigenvalue':cert['dual_min_eigenvalue'],'complementarity':cert['complementarity'],
                'iterations':int(result.nit),'message':str(result.message)})
            solutions.append(result)
            if valid:break
    candidates=[r for r,log in zip(solutions,runs) if log['valid']]
    if not candidates:raise RuntimeError('No converged targeted fit with a valid PSD-cone KKT certificate.')
    best=min(candidates,key=lambda r:r.fun)
    if any(log['feasible'] and np.isfinite(r.fun) and r.fun<best.fun-1e-7 for r,log in zip(solutions,runs)):
        raise RuntimeError('A lower feasible unconverged target candidate remains.')
    return best,runs


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


def inference(theta, args, filtered, kind, first_theta, first_filter, hac_lags=(0,3,6,12), se_step_tol=.01, pd_floor=1e-8, derivative_cache=None, first_hessians=None):
    """Stack first-stage QML, innovation moments and second-stage QML as applicable."""
    y,pdesign,scales,name,target = args; n=len(y); q=len(theta); k=layout(pdesign,name)[0]
    normal = target[TRI]; moment = normal-target_products(first_filter['errors'] if kind=='innovations' else y-y.mean(axis=0),scales)
    def pieces(step):
        A = jacobian(lambda t:objective(t,*args)[1],theta,step); A=(A+A.T)/2
        B = jacobian(lambda s:objective(theta,y,pdesign,scales,name,matrix_from_vech(s))[1],normal,step)
        if kind=='innovations':
            fa=(y,pdesign,scales,True,name)
            if first_hessians is None:
                A0=jacobian(lambda t:model.objective(t,*fa)[1],first_theta,step);A0=(A0+A0.T)/2
            else: A0=first_hessians[0 if step == 1e-4 else 1]
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
        old_limits=model.bounds(pdesign,True,name)
        if np.minimum(first_theta-old_limits[:,0],old_limits[:,1]-first_theta).min()<1e-4:reasons.append('first-stage parameter bound')
    if derivative_cache is None:
        A,A0,D=pieces(1e-4);small,small0,Dsmall=pieces(5e-5)
    else:
        A,A0,D,small,small0,Dsmall=derivative_cache
        if any(not np.isfinite(x).all() for x in [A,D,small,Dsmall]+([A0,small0] if A0 is not None else [])):
            raise ValueError('Invalid cached stacked derivatives.')
    condition=max(np.linalg.cond(D),np.linalg.cond(Dsmall))
    sensitivity=np.max(np.abs(D-Dsmall))/max(1.,np.max(np.abs(Dsmall)))
    hessians=[A,small]+([A0,small0] if A0 is not None else [])
    if min(np.linalg.eigvalsh(a).min() for a in hessians)<=0:reasons.append('nonpositive curvature')
    if condition>=1e9:reasons.append('ill-conditioned stacked Jacobian')
    if sensitivity>=1e-3:reasons.append('Jacobian step sensitivity')
    stacked=np.c_[first_filter['scores'],moment,filtered['scores']] if kind=='innovations' else np.c_[moment,filtered['scores']]
    estimates={}
    for lag in hac_lags:
        for method in ['target_adjusted','target_fixed']:
            why=reasons.copy();V=np.full((q,q),np.nan);diff=np.nan;path_diff=np.nan
            if not why:
                if method=='target_adjusted':
                    inv=np.linalg.inv(D)[-q:];inv_small=np.linalg.inv(Dsmall)[-q:];scores=stacked
                else:
                    inv=np.linalg.inv(A);inv_small=np.linalg.inv(small);scores=filtered['scores']
                meat=score_meat(scores,lag)
                V=inv@meat@inv.T/n;Vs=inv_small@meat@inv_small.T/n;V=(V+V.T)/2;Vs=(Vs+Vs.T)/2
                if not np.isfinite(V).all() or not np.isfinite(Vs).all() or min(np.diag(V).min(),np.diag(Vs).min())<=0:why.append('invalid sandwich')
                else:
                    diff=np.max(np.abs(np.sqrt(np.diag(Vs)[:k])/np.sqrt(np.diag(V)[:k])-1))
                    if diff>=se_step_tol:why.append('price SE step sensitivity')
                    ids=model.price_indices(pdesign,True);changes=[]
                    for a,design in pdesign.items():
                        ix=ids[a];v=np.einsum('ij,jk,ik->i',design,V[np.ix_(ix,ix)],design);vs=np.einsum('ij,jk,ik->i',design,Vs[np.ix_(ix,ix)],design)
                        if min(v.min(),vs.min())<=0:why.append('invalid '+a+' path variance')
                        else:changes.append(np.abs(np.sqrt(vs/v)-1).max())
                    path_diff=max(changes) if changes else np.nan
                    if changes and path_diff>=se_step_tol:why.append('price-path SE step sensitivity')
            estimates[(method,lag)]={'covariance':V if not why else np.full((q,q),np.nan),'ready':not why,
                                     'reason':'; '.join(why) or 'numerically ready','price_SE_step_difference':diff,'price_path_SE_step_difference':path_diff}
    return {'Hessian':A,'Hessian_small':small,'first_Hessian':A0,'first_Hessian_small':small0,'stack_Jacobian_small':Dsmall,'derivative_cache':(A,A0,D,small,small0,Dsmall),'stack_Jacobian':D,'stack_scores':stacked,'Jacobian_condition':condition,'Jacobian_step_difference':sensitivity,
            'PD_margin':margin,'near_bound':bool(distance.min()<1e-4 or margin<1e-6),'estimates':estimates}
