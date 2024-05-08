import numpy as np
from colorsys import hls_to_rgb
from numpy.random import uniform
import scipy.stats as stats

import matplotlib.pyplot as plt

def norm_inner(v1,v2):
    return v1.conj() @ v2/(np.linalg.norm(v1)*np.linalg.norm(v2))

def tr_prod(A,B):
    prod = np.abs(A @ np.swapaxes(B, axis1=-2, axis2=-1).conjugate())**2
    return np.trace(prod, axis1=-2, axis2=-1)

def fidelity(A,B):
    return tr_prod(A,B)/(np.sqrt(tr_prod(A,A)*tr_prod(B,B)))

def cpx_corr(Y1,Y2):
    corr = np.sum((Y1)*(Y2).conj())
    norm_fac = np.sqrt(np.sum(np.abs(Y1)**2)*np.sum(np.abs(Y2)**2))  
    return corr/norm_fac

def rnd_cpx_disk(size=None):
    rho = np.sqrt(uniform(0, 1, size))
    phi = uniform(0, 2*np.pi, size)
    return rho * np.exp(1j*phi)

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = np.transpose(c, (1,2,0))  
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
    
    
def norm_qqplot(x, ax=None, lgd=True, **kwargs):
    """Quantile-Quantile plot."""
    
    if ax is None:
        ax = plt.gca()
    
    # Extract quantiles
    quantiles = stats.probplot(x,  dist='norm', fit=False)
    theor = quantiles[0]
    observed = quantiles[1]
    # Rescale to default normal distribution:
    fit_params = stats.norm.fit(x)
    observed = (np.sort(observed) - fit_params[-2]) / fit_params[-1]

    mq = ax.scatter(theor, observed, **kwargs)
    
    lim_pts = [-5.8,5.8] 
    ax.set_xlim(lim_pts)
    ax.set_ylim(lim_pts)
    lp, = ax.plot(lim_pts, lim_pts, color="r", lw=1.5, ls='-')

    ax.set_aspect("equal")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered quantiles")
    if lgd:
        ax.legend([mq,lp],['Quantiles', 'Perfect match curve'], 
                  loc=0, facecolor='white',framealpha=0.9)
        
    return ax