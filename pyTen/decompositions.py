import numpy as np
import gc
import scipy
import functools
from numba import jit

rng = np.random.default_rng(12345)

from .operations import unfold, nmode_prod
from .functions import complexdisk_rand

import numpy as np



def get_HOSvec(x, n_mode, n_vec=None):
    """Computes the higher-order singular vectors of a tensor along a given mode.

    Computes the HOSVD via succesisve computations of the SVD of the 
    various matrix unfoldings of the tensor.

    Parameters
    ----------
    x : np.array
    rank : tuple or list of int (optional)

    Returns
    -------
    us : list of np.array
        List on matrices containing the higher order singular vectors
    s : np.array
        Core tensor
    """
    
    if n_vec is None:
        n_vec = x.shape[n_mode]

    unfold_x = unfold(x,n_mode)
    if unfold_x.shape[0] < unfold_x.shape[1]:
        fl=False
    else:
        fl=True
    gc.collect()
    u,_,_ = np.linalg.svd(unfold_x,full_matrices=fl)

    return u[:,:n_vec]

# HOSVD
def hosvd(x, rank=()):
    """Computes the higher-order singular decomposition of a tensor.

    Computes the HOSVD via succesisve computations of the SVD of the 
    various matrix unfoldings of the tensor.

    Parameters
    ----------
    x : np.array
    rank : tuple or list of int (optional)

    Returns
    -------
    us : list of np.array
        List on matrices containing the higher order singular vectors
    s : np.array
        Core tensor
    """
    order = len(x.shape)
    us = []
    if not rank:
        rank = x.shape
    for n in range(order):
        unfold_x = unfold(x,n)
        if unfold_x.shape[0] < unfold_x.shape[1]:
            fl=False
        else:
            fl=True
        gc.collect()
        u,_,_ = np.linalg.svd(unfold_x,full_matrices=fl)
        us += [u[:,:rank[n]]]
    s=x
    for n in range(order):
        s=nmode_prod(s,us[n].T.conj(),n)
    return us, s




# ALS

def outer_tenvec(ten,vec):
    return np.array(ten)[...,None,:]*np.array(vec)[:,:]

def sum_outers(mats):
    return np.sum(functools.reduce(outer_tenvec, mats), axis=-1) 

def _init_fact_matrices(x, rank, init_fact_mat=None):
    order = len(x.shape)
    if init_fact_mat is None:
        fact_mat = []
        for i in range(order):
            fact_mat += [complexdisk_rand(size=(x.shape[i], rank))]
        
    else:
        fact_mat = init_fact_mat
    return fact_mat.copy()

def _cum_khatri_rao(mats, mat_inds=None):
    if mat_inds is not None:
        matrices = [mats[i] for i in mat_inds]
    else:
        matrices = mats
    return functools.reduce(scipy.linalg.khatri_rao, matrices)

def _cum_hada2(mats, mat_inds=None):
    if mat_inds is not None:
        matrices = [mats[i] for i in mat_inds]
    else:
        matrices = mats
    f = lambda x : x.T @ x
    return np.prod(list(map(f,matrices)), axis=0)

def _als_mode_loop(x, fact_mat, special_form):
    order = len(x.shape)
    for mode in range(order):
        hr_ind = np.arange(mode+1, mode+order)%order
        kr_prod = _cum_khatri_rao(fact_mat, mat_inds=hr_ind)
        if special_form:            
            hada_prod = _cum_hada2(fact_mat, mat_inds=hr_ind)
            fact_mat[mode] = unfold(x, mode) @ kr_prod @ np.linalg.pinv(hada_prod)
        else:
            fact_mat[mode] = unfold(x, mode) @  np.linalg.pinv(kr_prod.T)
    return fact_mat


def als(x, rank, max_iter, init_fact_mat=None, evol=False, special_form=False):
    """Computes the canonical polyadic decomposition via ALS algorithm"""
    
    norm_x = np.linalg.norm(x)
    fact_mat =_init_fact_matrices(x, rank, init_fact_mat=init_fact_mat)
    
    cost_evol = []
    for n_iter in range(max_iter):
        fact_mat = _als_mode_loop(x, fact_mat, special_form)

        if evol:
            x_approx = sum_outers(fact_mat)#np.einsum('ij,kj,lj',fact_mat[0],fact_mat[1],fact_mat[2])
            cost_evol += [np.linalg.norm(x-x_approx)/norm_x]

    return fact_mat, cost_evol


# Symmetric ALSs

def _symmetrize_fact_mats(fact_mats, anti=False):
    norm_fm0 = _norm_fact_mat(fact_mats[0])
    norm_fm1 = _norm_fact_mat(fact_mats[1])
    sign = 1
    if anti: 
        sign=-1
    fact_mat0 = norm_fm0*(fact_mats[0]/norm_fm0 + sign*(fact_mats[1]/norm_fm1).conj())/2
    fact_mat1 = sign*norm_fm1*fact_mat0.conj()/norm_fm0
    return [fact_mat0, fact_mat1]



def _norm_fact_mat(fact_mat):
    return np.linalg.norm(fact_mat, axis=0, keepdims=True)



def als3herm(x, rank, max_iter, init_fact_mat=None, evol=False, special_form=False):
    """Computes the canonical polyadic decomposition of partially Hermitian 
    3rd-order tensors via ALS algorithm"""

    norm_x = np.linalg.norm(x)
    fact_mat =_init_fact_matrices(x, rank, init_fact_mat=init_fact_mat)
    
    cost_evol = []
    for n_iter in range(max_iter):
        fact_mat = _als_mode_loop(x, fact_mat, special_form)

        fact_mat[1:] = _symmetrize_fact_mats(fact_mat[1:])

        if evol:
            x_approx = sum_outers(fact_mat)
            cost_evol += [np.linalg.norm(x-x_approx)/norm_x]

    return fact_mat, cost_evol

def als4herm2(x, rank, max_iter, init_fact_mat=None, evol=False, special_form=False,anti=False):
    norm_x = np.linalg.norm(x)
    fact_mat =_init_fact_matrices(x, rank, init_fact_mat=init_fact_mat)
    
    cost_evol = []
    for n_iter in range(max_iter):

        fact_mat = _als_mode_loop(x, fact_mat, special_form)
            
        fact_mat[0:2] = _symmetrize_fact_mats(fact_mat[0:2], anti=anti)
        fact_mat[2:] = _symmetrize_fact_mats(fact_mat[2:])
        
        if evol:
            x_approx = sum_outers(fact_mat)
            cost_evol += [np.linalg.norm(x-x_approx)/norm_x]

    return fact_mat, cost_evol



# HOOI

def hooi(x, rank, max_iter=50, entries=(), evol=False):
    """Computes Tucker deomposition via HOOI algorithm"""
    if not entries:
        entries = tuple(range(len(x.shape)))
    
    norm_x = np.linalg.norm(x)
    us, _ = hosvd(x, rank)
    cost_evol = []
    for n_iter in range(max_iter):

        # Update unitary matrices
        for entry in entries:
            for mode, u in enumerate(us):
                if mode == entry:
                    continue
                y = nmode_prod(x,u.T.conj(),mode)
            gc.collect()
            u_current, _, _ = np.linalg.svd(unfold(y,entry),full_matrices=False)
            us[entry] = u_current[:,:rank[entry]]
            
        # Update core tensor
        if evol:
            core = x.copy()
            for mode, u in enumerate(us):
                core = nmode_prod(core, u.T.conj(), mode)
            
            x_approx = core
            for mode, u in enumerate(us):
                x_approx = nmode_prod(x_approx, u, mode)
            
            cost_evol += [np.linalg.norm(x-x_approx)/norm_x]
            
    core = x.copy()
    for mode, u in enumerate(us):
        core = nmode_prod(core, u.T, mode)

    return us, core, cost_evol 

# Power method 

def _init_power(x, init_eigvec=None):
    
    if init_eigvec is None:
        eigvec = complexdisk_rand(size=(x.shape[-1]))        
    else:
        eigvec = init_eigvec
    return eigvec.copy()

def power3herm(x, max_iter, init_eigvec=None, evol=False):
    norm_x = np.linalg.norm(x)
    eigvec =_init_power(x, init_eigvec=init_eigvec)
    
    cost_evol = []
    ortovec = np.sum(x * eigvec[:,None] * eigvec[None,:].conj(), axis=(1,2))
    for n_iter in range(max_iter):
        eigvec = np.sum(x * ortovec[:,None,None] * eigvec[:,None], axis=(0,1))
        # norm_eigvec = np.linalg.norm(eigvec)
        eigvec /= np.linalg.norm(eigvec) 
        ortovec = np.sum(x * eigvec[:,None] * eigvec[None,:].conj(), axis=(1,2))
        
        if evol:
            x_approx = ortovec[:,None,None]*eigvec[None,:,None].conj()*eigvec[None,None,:]
            cost_evol += [np.linalg.norm(x-x_approx)/norm_x]
    return ortovec, eigvec, cost_evol

# @jit(nopython=True)
def power4herm(x, max_iter, init_vecs, evol=False):
    norm_x = np.sum(np.abs(x)**2)**(1/2)
    outvec = init_vecs[0]
    invec = init_vecs[1]
    
    cost_evol = []
    for n_iter in range(max_iter):
        invec = np.sum(np.sum(np.sum(x * outvec[:,None,None,None].conj() * outvec[:,None,None] * invec[:,None]
            , axis=0), axis=0), axis=0)
        invec /= np.sum(np.abs(invec)**2)**(1/2)

        outvec = np.sum(np.sum(np.sum(x * outvec[:,None,None] * invec[:,None] * invec.conj()
            , axis=1), axis=1), axis=1)
        lam = np.sum(np.abs(outvec)**2)**(1/2)
        outvec /= lam
        
        if evol:
            x_approx = lam*outvec[:,None,None,None]*outvec[:,None,None].conj()*invec[:,None].conj()*invec
            cost_evol += [np.sum(np.abs(x-x_approx)**2)**(1/2)/norm_x]
    return [outvec, invec], cost_evol