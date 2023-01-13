import numpy as np

# Metrics

def complex_correlation(Y1,Y2):
    Y1 = Y1-Y1.mean()
    Y2 = Y2-Y2.mean()
    return np.abs(np.sum(Y1.ravel() * Y2.ravel().conj())) \
           / np.sqrt(np.sum(np.abs(Y1.ravel())**2) *np.sum(np.abs(Y2.ravel())**2))


tr = lambda A,B: np.trace(np.abs(A@B.transpose().conjugate())**2)

fidelity = lambda A,B: tr(A,B)/(np.sqrt(tr(A,A)*tr(B,B)))

def cpx_corr_seq(Ys,Yref=None):
    if Yref is None:
        Yref = Ys[0]
    Ys = Ys - np.mean(Ys, axis=-1, keepdims=True)
    Yref = Yref - np.mean(Yref)

    np.abs(np.sum(Ys * Yref.conj())) \
           / np.sqrt(np.sum(np.abs(Ys)**2,axis=-1,keepdims=True) *np.sum(np.abs(Yref)**2))
# 