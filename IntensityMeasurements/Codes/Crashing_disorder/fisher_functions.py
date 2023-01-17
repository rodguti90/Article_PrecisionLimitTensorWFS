import numpy as np

def normalize(x):
    return x/np.linalg.norm(x)

#############################################################################
# Tensor defs
#############################################################################

def sym_exp(H):
    return H[...,None]*np.conj(H[:,None,:])

def sym_2exp(H1,H2):
    return H1[...,None]*np.conj(H2[:,None,:])    

def get_W3(Hs, method='2pts'):
    if method == '2pts':
        return sym_exp(Hs[-1])-sym_exp(Hs[0])
    elif '3pts':
        der = Hs[-1] - Hs[0]
        return sym_2exp(der,Hs[1])+sym_2exp(Hs[1],der)
    else:
        raise ValueError('Invalid option for method use 2pts or 3pts')

def inout_tensor(H1,H2=None):
    if H2 is None:
        H2 = H1
    return H1[:,None,:,None]*np.conj(H2[None,:,None,:])

def get_W4(Hs, method='2pts'):
    if method == '2pts':
        return inout_tensor(Hs[-1]) - inout_tensor(Hs[0])
    elif '3pts':
        der = Hs[-1] - Hs[0]
        return inout_tensor(der,Hs[1])+inout_tensor(Hs[1],der)
    else:
        raise ValueError('Invalid option for method use 2pts or 3pts')
    

def get_WS(Hs):
    '''
    Get the Wigner-Smith operator from 3 matrices at Dx-dx, Dx and Dx+dx
    '''
    dH = (Hs[-1] - Hs[0])*0.5
    WS = -1j*np.linalg.pinv(Hs[1])@dH
    return WS

def get_WS_symm(Hs):
    '''
    Get the Hermitian part of the Wigner-Smith operator from 3 matrices at Dx-dx, Dx and Dx+dx
    '''
    WS = get_WS(Hs)
    WS_symm = WS + WS.T.conj()
    return WS_symm

def getFisherOperator(Hs):
    der = Hs[-1] - Hs[0]
    return 4*der.T.conj() @ der

#############################################################################
# Get output fields
#############################################################################

def normalizevecs(X):
    return X/(np.sum(np.abs(X)**2,axis=-1, keepdims=True))**(1/2)

# Give option to project on another basis (pass from modes to pixels)
def getOutputFields(X, H):
    Hs = np.array(H)    
    Xnorm = normalizevecs(X)
    if (len(np.shape(Hs)) == 2) or (len(np.shape(X)) == 1):
        Ys = Hs @ Xnorm[...,None]
    elif len(np.shape(Hs)) == 3:
        Ys = Hs[:,None,...] @ Xnorm[...,None]
    else:
        raise ValueError('Dimension of X is invalid')
    return np.squeeze(Ys)

def getYpix(X,H,modes_out):
    Y = H @ X
    return np.sum(Y[:,None]*modes_out,axis=0)

def getPixAmpChange(X,H0,H1,modes_out):
    Y0 = getYpix(X,H0,modes_out)
    Y1 = getYpix(X,H1,modes_out)
    return np.abs(Y1)-np.abs(Y0)


# def getOutputFields(X, Hs):    
#     Xnorm = X/(np.sum(np.abs(X)**2,axis=-1, keepdims=True))**(1/2)
#     if len(np.shape(X)) == 1:
#         Ys = Hs @ Xnorm
#     elif len(np.shape(X)) == 2:
#         Ys = np.squeeze(Hs @ Xnorm[:,None,:,None])
#     else:
#         raise ValueError('Dimension of X is invalid')
#     return np.squeeze(Ys)

#############################################################################
# Get Fisher information 
#############################################################################

# Implement 3pts derivative
def fisherPerMode(X, Hs, noise='poisson', method='2pts'):
    Ys = getOutputFields(X, Hs)
    if noise == 'poisson':
        if  len(np.shape(Ys)) == 2:
            return 4*(np.abs(Ys[1]) - np.abs(Ys[0]))**2        
        elif  len(np.shape(Ys)) == 3:
            return 4*(np.abs(Ys[:,1]) - np.abs(Ys[:,0]))**2
        else:
            raise ValueError('Dimension of Ys is invalid')
    elif noise =='gaussian':
        if method=='2pts':
            return (np.abs(Ys[-1])**2 - np.abs(Ys[0])**2)**2
        elif method=='3pts':
            return (2*np.real((Ys[-1]-Ys[0])*Ys[1].conj()))**2
        else:
            raise ValueError('Invalid option for method use 2pts or 3pts')
    else:
        raise ValueError('Invalid noise option')

# def fisherPerMode(X, Hs, noise='poisson', method='2pts'):
#     Ys = getOutputFields(X, Hs)
#     if noise == 'poisson':
#         if  len(np.shape(Ys)) == 2:
#             return 4*(np.abs(Ys[1]) - np.abs(Ys[0]))**2        
#         elif  len(np.shape(Ys)) == 3:
#             return 4*(np.abs(Ys[:,1]) - np.abs(Ys[:,0]))**2
#         else:
#             raise ValueError('Dimension of Ys is invalid')
#     elif noise =='gaussian':
#         if  len(np.shape(Ys)) == 2:
#             return (np.abs(Ys[1])**2 - np.abs(Ys[0])**2)**2        
#         elif  len(np.shape(Ys)) == 3:
#             return (np.abs(Ys[:,1])**2 - np.abs(Ys[:,0])**2)**2
#         else:
#             raise ValueError('Dimension of Ys is invalid')
#     else:
#         raise ValueError('Invalid noise option')

def fisher(X, Hs, noise='poisson'):
    return np.sum(fisherPerMode(X, Hs, noise=noise), axis=-1)

#############################################################################
# Get Fisher information 
#############################################################################

def derYop(X, H):
    Ys = getOutputFields(X, H)
    # y1 = Hs[-1] @ x
    # y0 = Hs[0] @ x
    # Y1 = Ys[-1,...,None] * np.expand_dims(Ys[-1], axis=-2).conj()
    # Y0 = Ys[0,...,None] * np.expand_dims(Ys[0], axis=-2).conj()
    Youter = Ys[...,None] * np.expand_dims(Ys, axis=-2).conj()
    # Y1 =  y1[:,None] * y1[None,:].conj()
    # Y0 =  y0[:,None] * y0[None,:].conj()
    return Youter[-1] - Youter[0]

def get_moim(X, H, all=False):
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