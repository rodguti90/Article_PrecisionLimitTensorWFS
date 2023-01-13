import numpy as np
# import gc
# import scipy
# import functools
rng = np.random.default_rng(12345)

def complexdisk_rand(size=1):
    amp= np.sqrt(np.random.uniform(0,1,size))
    phase = np.random.uniform(0,2*np.pi,size)
    return amp*np.exp(1j*phase)
