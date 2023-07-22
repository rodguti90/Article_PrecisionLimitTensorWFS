import numpy as np
# import gc
# import scipy
# import functools
rng = np.random.default_rng(12345)


# def normalize(x):
#     return x/np.linalg.norm(x)

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