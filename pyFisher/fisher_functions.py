import numpy as np


#############################################################################
# Tensor defs
#############################################################################

def sym_exp(H):
    return H[...,None]*np.conj(H[:,None,:])

def get_W3(Hs):
    return sym_exp(Hs[-1])-sym_exp(Hs[0])

def sym_2exp(H1,H2):
    return H1[...,None]*np.conj(H2[:,None,:])    

def inout_tensor(H1,H2=None):
    if H2 is None:
        H2 = H1
    return H1[:,None,:,None]*np.conj(H2[None,:,None,:])

def get_W4(Hs):
    return inout_tensor(Hs[-1]) - inout_tensor(Hs[0])
    
#############################################################################
# Get output fields
#############################################################################

def normalizevecs(X):
    return X/(np.sum(np.abs(X)**2,axis=-1, keepdims=True))**(1/2)

def getOutputFields(X, H, normalize=True):
    Hs = np.array(H)  
    if normalize:  
        Xnorm = normalizevecs(X)
    else:
        Xnorm = X.copy()
    if (len(np.shape(Hs)) == 2) or (len(np.shape(X)) == 1):
        Ys = Hs @ Xnorm[...,None]
    elif len(np.shape(Hs)) == 3:
        Ys = Hs[:,None,...] @ Xnorm[...,None]
    else:
        raise ValueError('Dimension of X is invalid')
    return np.squeeze(Ys)


#############################################################################
# Get Fisher information 
#############################################################################

def fisherPerMode(X, Hs, dx=1):
    Ys = getOutputFields(X, Hs)
    return (np.abs(Ys[-1])**2 - np.abs(Ys[0])**2)**2/dx**2

def fisher(X, Hs, dx=1):
    return np.sum(fisherPerMode(X, Hs, dx=dx), axis=-1)

#############################################################################
# Get Fisher information 
#############################################################################

def derYop(X, H):
    Ys = getOutputFields(X, H)
    Youter = Ys[...,None] * np.expand_dims(Ys, axis=-2).conj()
    return Youter[-1] - Youter[0]

def get_oopms(X, H, all=False):
    X=normalizevecs(X)
    Yder = derYop(X, H)
    if len(X.shape) == 1:
        lam, v = np.linalg.eigh(Yder)
        sort_ind = np.argsort(lam**2)
        if all:
            return v[:,sort_ind], lam[sort_ind]
        else:
            return v[:,sort_ind[-2:]], lam[sort_ind[-2:]]
    else:
        nxs = X.shape[0]
        vecs = np.empty((nxs,H[0].shape[0],2), complex)
        vals = np.empty((nxs,2), complex)
        for indx in range(nxs):
            lam, v = np.linalg.eigh(Yder[indx])
            sort_ind = np.argsort(lam**2)
            vecs[indx] = v[:,sort_ind[-2:]]
            vals[indx] = lam[sort_ind[-2:]]

        return vecs, vals

def get_fish_oopms(X, H):
    _, vals = get_oopms(X, H)
    return np.sum(np.abs(vals)**2, axis=-1)

