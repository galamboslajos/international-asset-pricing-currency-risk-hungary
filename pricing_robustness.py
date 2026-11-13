"""Joint GARCH-in-mean specification checks for the Hungary research notebooks.

All returns are monthly log percentage points. No downloads, data edits or kernel management.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

GROUPS={"scalar":[0,0,0,0,0], "grouped":[0,0,1,2,2], "diagonal":[0,1,2,3,4]}

PRICE_SCALE = 0.01
N = 5
TRI = np.tril_indices(N)

def layout(x, pdesign, covariance_model):
    k = 5 + sum(pdesign[a].shape[1]-1 for a in ["W","H","O"])
    g = 2 * x.shape[1]
    dynamics = 2*(max(GROUPS[covariance_model])+1)
    return k, g, dynamics

def unpack(theta, scales, x, pdesign, covariance_model):
    k, g, dynamics = layout(x, pdesign, covariance_model)
    q = len(theta)
    prices, dp = np.zeros(5), np.zeros((q, 5))
    prices[:] = PRICE_SCALE * theta[:5]
    dp[np.arange(5), np.arange(5)] = PRICE_SCALE
    gamma = theta[k:k+g].reshape(2, x.shape[1])
    C, dC = np.zeros((N, N)), np.zeros((q, N, N))
    for offset, (i, j) in enumerate(zip(*TRI)):
        index = k + g + offset
        value = np.exp(theta[index]) if i == j else theta[index]
        C[i, j] = scales[i] * value
        dC[index, i, j] = scales[i] * (value if i == j else 1.0)
    intercept = C @ C.T
    dintercept = dC @ C.T + C @ dC.transpose(0, 2, 1)
    weights = softmax(np.c_[np.zeros(dynamics // 2), theta[-dynamics:].reshape(-1, 2)], axis=1)
    group_ids = np.asarray(GROUPS[covariance_model])
    alpha, beta = weights[group_ids,1], weights[group_ids,2]
    da, db = np.zeros((q, N)), np.zeros((q, N))
    for i in range(N):
        offset = group_ids[i]
        u, v = q - dynamics + 2 * offset, q - dynamics + 2 * offset + 1
        da[u, i], da[v, i] = alpha[i] * (1-alpha[i]), -alpha[i]*beta[i]
        db[u, i], db[v, i] = -alpha[i]*beta[i], beta[i]*(1-beta[i])
    a, b = np.sqrt(alpha), np.sqrt(beta)
    da, db = da / (2*a), db / (2*b)
    aa, bb = np.outer(a, a), np.outer(b, b)
    daa = da[:, :, None]*a[None, None, :] + a[None, :, None]*da[:, None, :]
    dbb = db[:, :, None]*b[None, None, :] + b[None, :, None]*db[:, None, :]
    denominator = 1-aa-bb
    target = intercept / denominator
    dtarget = dintercept / denominator + intercept*(daa+dbb)/denominator**2
    return prices, dp, gamma, intercept, dintercept, aa, daa, bb, dbb, target, dtarget, alpha, beta

def price_indices(pdesign):
    world_end = 5 + pdesign["W"].shape[1]-1
    huf_end = world_end + (pdesign["H"].shape[1]-1)
    other_end = huf_end + (pdesign["O"].shape[1]-1)
    return {"W":np.r_[0,np.arange(5,world_end)],
            "H":np.r_[3,np.arange(world_end,huf_end)],
            "O":np.r_[4,np.arange(huf_end,other_end)]}

def filter_model(theta, y, x, pdesign, scales, mean_mode, covariance_model):
    if mean_mode not in ["var", "priced"]:
        raise ValueError("mean_mode must be var or priced")
    prices, dp, gamma, intercept, dintercept, aa, daa, bb, dbb, target, dtarget, alpha, beta = unpack(theta, scales, x, pdesign, covariance_model)
    H, dH = target.copy(), dtarget.copy()
    n, q = len(y), len(theta)
    k, g, _ = layout(x, pdesign, covariance_model)
    losses, scores = np.empty(n), np.empty((n, q))
    covariances, means, residuals = np.empty((n, N, N)), np.empty_like(y), np.empty_like(y)
    indices = price_indices(pdesign)
    price_values, price_gradients = {}, {}
    for label in ["W","H","O"]:
        design = pdesign[label]
        price_values[label] = PRICE_SCALE * (design @ theta[indices[label]])
        price_gradients[label] = np.zeros((n,q))
        price_gradients[label][:,indices[label]] = PRICE_SCALE * design
    priced_rows = 3 if mean_mode=="var" else 5
    for t in range(n):
        if not np.isfinite(H).all() or np.max(np.diag(H)) > 1e8:
            raise FloatingPointError('Unstable covariance recursion')
        chol, inverse = np.linalg.cholesky(H), np.linalg.inv(H)
        mu, dmu = np.zeros(N), np.zeros((q, N))
        for label, factor in [("W",0),("H",3),("O",4)]:
            value = price_values[label][t]
            derivative = price_gradients[label][t]
            mu[:priced_rows] += value * H[:priced_rows, factor]
            dmu[:, :priced_rows] += value * dH[:, :priced_rows, factor] + derivative[:, None] * H[None, :priced_rows, factor]
        for asset in (1, 2):
            mu[asset] += prices[asset] * H[asset, asset]
            dmu[:, asset] += prices[asset]*dH[:, asset, asset] + dp[:, asset]*H[asset, asset]
        mu[3:] += scales[3:] * (gamma @ x[t])
        for j in range(2):
            start = k + j*x.shape[1]
            dmu[start:start+x.shape[1], 3+j] += scales[3+j]*x[t]
        error, de = y[t]-mu, -dmu
        v = inverse @ error
        losses[t] = .5*(N*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+error@v)
        scores[t] = .5*np.einsum('ij,kij->k', inverse-np.outer(v,v), dH) + de@v
        covariances[t], means[t], residuals[t] = H, mu, error
        shock = np.outer(error, error)
        dshock = de[:, :, None]*error[None, None, :] + error[None, :, None]*de[:, None, :]
        dH = dintercept + daa*shock + aa*dshock + dbb*H + bb*dH
        H = intercept + aa*shock + bb*H
    return {'losses':losses, 'scores':scores, 'H':covariances, 'mu':means, 'errors':residuals,
            'prices':prices, 'world_prices':price_values['W'], 'fx_prices':np.c_[price_values['H'],price_values['O']], 'price_coefficients':{a:PRICE_SCALE*theta[i] for a,i in indices.items()}, 'gamma':gamma, 'intercept':intercept, 'target':target, 'alpha':alpha, 'beta':beta}

def objective(theta, y, x, pdesign, scales, mean_mode, covariance_model):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            f = filter_model(theta, y, x, pdesign, scales, mean_mode, covariance_model)
        if not np.isfinite(f['losses']).all() or not np.isfinite(f['scores']).all():
            raise FloatingPointError('Non-finite likelihood or gradient')
        return f['losses'].mean(), f['scores'].mean(axis=0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return 1e8 + theta@theta, 2*theta


def initial_theta(y,x,pdesign,scales,mean_mode,covariance_model,alpha=.05,beta=.90):
    k,g,dynamics=layout(x,pdesign,covariance_model)
    gamma=np.linalg.lstsq(x,y[:,3:]/scales[3:],rcond=None)[0].T if x.shape[1] else np.zeros((2,0))
    centred=y.copy();centred[:,3:]-=(x@gamma.T)*scales[3:]
    covariance=np.cov(centred,rowvar=False)
    if mean_mode=="priced":
        factors=[0,3,4]
        world,huf,other=np.linalg.solve(covariance[np.ix_(factors,factors)],centred[:,factors].mean(axis=0))
    else:world,huf,other=y[:,0].mean()/covariance[0,0],0.,0.
    means=world*covariance[:,0]+huf*covariance[:,3]+other*covariance[:,4]
    local=[(y[:,i].mean()-means[i])/covariance[i,i] for i in [1,2]]
    prices=np.array([world,*local,huf,other])/PRICE_SCALE
    remaining=1-alpha-beta
    C=np.linalg.cholesky(remaining*covariance/np.outer(scales,scales))
    chol=[np.log(C[i,j]) if i==j else C[i,j] for i,j in zip(*TRI)]
    return np.r_[prices,np.zeros(k-5),gamma.ravel(),chol,
                 np.tile([np.log(alpha/remaining),np.log(beta/remaining)],dynamics//2)]

def parameter_bounds(x,pdesign,covariance_model):
    k,g,dynamics=layout(x,pdesign,covariance_model)
    covariance=[(-8.,3.) if i==j else (-4.,4.) for i,j in zip(*TRI)]
    return [(-100.,100.)]*k+[(-3.,3.)]*g+covariance+[(-10.,10.)]*dynamics

def change_mean(theta,old_x,new_x,pdesign):
    k=5+sum(pdesign[a].shape[1]-1 for a in ["W","H","O"])
    return np.r_[theta[:k],np.zeros(2*new_x.shape[1]),theta[k+2*old_x.shape[1]:]]

def expand_scalar(theta,covariance_model):
    return np.r_[theta[:-2],np.tile(theta[-2:],max(GROUPS[covariance_model])+1)]

def fit_starts(y,x,pdesign,scales,mean_mode,covariance_model,starts,maxiter=1800):
    bounds=np.asarray(parameter_bounds(x,pdesign,covariance_model));args=(y,x,pdesign,scales,mean_mode,covariance_model)
    runs,solutions=[],[]
    for i,start in enumerate(starts,1):
        coarse=minimize(objective,start,args=args,method="BFGS",jac=True,options={"maxiter":maxiter,"gtol":1e-6})
        result=minimize(objective,np.clip(coarse.x,bounds[:,0],bounds[:,1]),args=args,method="L-BFGS-B",jac=True,bounds=bounds,
                        options={"maxiter":maxiter,"ftol":1e-13,"gtol":1e-7,"maxls":50,"maxcor":50})
        gradient=float(np.max(np.abs(result.jac)))
        print(f"  start {i}: success={result.success}, NLL={result.fun:.9f}, gradient={gradient:.2e}, iterations={coarse.nit}+{result.nit}",flush=True)
        runs.append({"start":i,"success":bool(result.success),"mean_nll":float(result.fun),"max_abs_gradient":gradient,
                     "iterations":int(coarse.nit+result.nit),"BFGS_success":bool(coarse.success),"message":str(result.message)})
        solutions.append(result)
    valid=[r for r in solutions if r.success and np.isfinite(r.fun) and r.fun<1e7 and np.max(np.abs(r.jac))<1e-4]
    if not valid:raise RuntimeError("No converged fit with a small gradient; inspect starts/MAX_ITER.")
    best=min(valid,key=lambda r:r.fun)
    if any(np.isfinite(r.fun) and r.fun<best.fun-1e-7 for r in solutions):
        raise RuntimeError("A lower nonconverged candidate remains. Increase MAX_ITER.")
    return best,runs

def numerical_gradient(function,theta,step=1e-5):
    result=np.empty_like(theta)
    for j in range(len(theta)):
        delta=step*(1+abs(theta[j]));plus=theta.copy();minus=theta.copy()
        plus[j]+=delta;minus[j]-=delta
        result[j]=(function(plus)-function(minus))/(2*delta)
    return result

def loss_hessian(theta,y,x,pdesign,scales,mean_mode,covariance_model,step=1e-4):
    result=np.empty((len(theta),len(theta)))
    for j in range(len(theta)):
        delta=step*(1+abs(theta[j]));plus=theta.copy();minus=theta.copy()
        plus[j]+=delta;minus[j]-=delta
        result[:,j]=(objective(plus,y,x,pdesign,scales,mean_mode,covariance_model)[1]-objective(minus,y,x,pdesign,scales,mean_mode,covariance_model)[1])/(2*delta)
    return (result+result.T)/2

def var_radius(gamma):
    if gamma.shape[1]<=1:return 0.
    lags=(gamma.shape[1]-1)//2
    top=np.concatenate([gamma[:,1+2*j:3+2*j] for j in range(lags)],axis=1)
    companion=np.r_[top,np.c_[np.eye(2*(lags-1)),np.zeros((2*(lags-1),2))]]
    return float(np.max(np.abs(np.linalg.eigvals(companion))))

def inference(theta,y,x,pdesign,scales,mean_mode,covariance_model,filtered,se_step_tol=.01):
    args=(y,x,pdesign,scales,mean_mode,covariance_model)
    A=loss_hessian(theta,*args);small=loss_hessian(theta,*args,step=5e-5)
    eigenvalues=np.linalg.eigvalsh(A);condition=np.linalg.cond(A)
    sensitivity=np.max(np.abs(A-small))/max(1.,np.max(np.abs(small)))
    bounds=np.asarray(parameter_bounds(x,pdesign,covariance_model))
    distance=np.minimum(theta-bounds[:,0],bounds[:,1]-theta)
    boundary=bool(distance.min()<1e-4);radius=var_radius(filtered["gamma"])
    reasons=[]
    if boundary:reasons.append("parameter bound")
    if eigenvalues.min()<=0:reasons.append("nonpositive curvature")
    if condition>=1e9:reasons.append("ill-conditioned curvature")
    if sensitivity>=1e-3:reasons.append("Hessian step sensitivity")
    if radius>=1:reasons.append("unstable FX AR correction")
    V=np.full_like(A,np.nan);se_difference=np.nan
    if not reasons:
        scores=filtered["scores"]-filtered["scores"].mean(axis=0);meat=scores.T@scores/len(y)
        def sandwich(H):
            inverse=np.linalg.inv(H);v=inverse@meat@inverse.T/len(y)
            return (v+v.T)/2
        V,Vsmall=sandwich(A),sandwich(small)
        if (not np.isfinite(V).all() or not (np.diag(V)>0).all() or np.linalg.eigvalsh(small).min()<=0 or not (np.diag(Vsmall)>0).all()):
            reasons.append("invalid sandwich")
        else:
            k=layout(x,pdesign,covariance_model)[0]
            se_difference=np.max(np.abs(np.sqrt(np.diag(Vsmall)[:k])/np.sqrt(np.diag(V)[:k])-1))
            if se_difference>=se_step_tol:reasons.append("price SE step sensitivity")
    ready=not reasons
    return {"covariance":V if ready else np.full_like(A,np.nan),"Hessian":A,"ready":ready,"reason":"; ".join(reasons) or "numerically ready",
            "Hessian_min_eigenvalue":eigenvalues.min(),"Hessian_condition":condition,"Hessian_step_difference":sensitivity,
            "price_SE_step_difference":se_difference,"near_bound":boundary,"FX_AR_radius":radius,"distance_to_bounds":distance}

def wald(theta,covariance,indices):
    idx=np.asarray(indices,dtype=int);v=covariance[np.ix_(idx,idx)]
    valid=len(idx)>0 and np.isfinite(v).all() and np.linalg.eigvalsh(v).min()>0 and np.linalg.cond(v)<1e9
    statistic=theta[idx]@np.linalg.solve(v,theta[idx]) if valid else np.nan
    return {"Wald_statistic":statistic,"df":len(idx),"robust_p_value":chi2.sf(statistic,len(idx)) if valid else np.nan}

def diagnostics(filtered,labels,lags=12):
    z=filtered["errors"]/np.sqrt(np.diagonal(filtered["H"],axis1=1,axis2=2))
    white=np.linalg.solve(np.linalg.cholesky(filtered["H"]),filtered["errors"][...,None])[...,0]
    marginal,cross=[],[]
    for j,label in enumerate(labels):
        marginal.append({"series":label,"z_mean":z[:,j].mean(),"z_sd":z[:,j].std(ddof=1),
            "LB_z_p":acorr_ljungbox(z[:,j],lags=[lags],return_df=True)["lb_pvalue"].iloc[0],
            "LB_z_squared_p":acorr_ljungbox((z[:,j]-z[:,j].mean())**2,lags=[lags],return_df=True)["lb_pvalue"].iloc[0],
            "ARCH_LM_p":het_arch(z[:,j],nlags=lags,ddof=0,result_object=False)[1]})
    for i in range(5):
        for j in range(i+1,5):
            product=white[:,i]*white[:,j]
            cross.append({"pair":f"{labels[i]} / {labels[j]}","LB_product_p":acorr_ljungbox(product,lags=[lags],return_df=True)["lb_pvalue"].iloc[0]})
    return pd.DataFrame(marginal).set_index("series"),pd.DataFrame(cross).set_index("pair")
