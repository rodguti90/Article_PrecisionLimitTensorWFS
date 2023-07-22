import numpy as np
import scipy
import functools
rng = np.random.default_rng(12345)


# def normalize(x):
#     return x/np.linalg.norm(x)

def outer_tenvec(ten,vec):
    return np.array(ten)[...,None,:]*np.array(vec)[:,:]

def sum_outers(mats):
    return np.sum(functools.reduce(outer_tenvec, mats), axis=-1) 

def cum_khatri_rao(mats, mat_inds=None):
    if mat_inds is not None:
        matrices = [mats[i] for i in mat_inds]
    else:
        matrices = mats
    return functools.reduce(scipy.linalg.khatri_rao, matrices)

def _tensor_roll_indices(x,n):
    inds = np.arange(len(np.shape(x)))
    return np.moveaxis(x, inds, np.roll(inds,n))

def unfold(x, n):
    dims = x.shape
    return _tensor_roll_indices(x,n).reshape(dims[n],-1)

def fold(x,n,shape):
    x = x.reshape(np.roll(shape,-n))
    return _tensor_roll_indices(x,-n)

def nmode_prod(x,u,n):
    shape = np.array(x.shape)
    shape[n] = u.shape[0]
    return fold(u@unfold(x,n),n,shape)


# def cum_hada2(mats, mat_inds=None):
#     if mat_inds is not None:
#         matrices = [mats[i] for i in mat_inds]
#     else:
#         matrices = mats
#     f = lambda x : x.T @ x
#     return np.prod(list(map(f,matrices)), axis=0)