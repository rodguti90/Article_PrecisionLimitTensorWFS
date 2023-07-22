import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(42)

rng = np.random.default_rng(12345)


class OutputFields(nn.Module):
    def __init__(self, H0, H1, Xinit=None):
        super().__init__()
        
        n_out, n_in = H0.shape
        if Xinit is None:
            X = torch.randn(n_in, dtype=torch.cfloat)
            X = X / torch.sum(torch.abs(X)**2)
        else:
            X = torch.tensor(Xinit, dtype=torch.cfloat)
        self.X = nn.Parameter(X, requires_grad=True)
        self.H0 = torch.tensor(H0, dtype=torch.cfloat)
        self.H1 = torch.tensor(H1, dtype=torch.cfloat)
       
    def forward(self):
        Xnorm = self.X/(torch.sum(torch.abs(self.X)**2))**(1/2)
        Y0 = self.H0 @ Xnorm
        Y1 = self.H1 @ Xnorm
        return Y0, Y1


def fisher_opt_opm(Y0, Y1):
    Ymat1 =  Y1[:,None] * Y1[None,:].conj()
    Ymat0 =  Y0[:,None] * Y0[None,:].conj()
    L, _ = torch.linalg.eigh(Ymat1-Ymat0)
    return 1/torch.sum(L**2)

def optimize_inputNopms(Xinit,TMs, n_epochs=200, lr=1e-2): 
    model = OutputFields(TMs[0], TMs[1], Xinit=Xinit)

    loss_fn = fisher_opt_opm

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_evol = []
    for epoch in range(n_epochs):
        
        Y0, Y1 = model()
        loss = loss_fn(Y0, Y1)
        loss_evol += [loss.item()]
        loss.backward()    
        optimizer.step()
        optimizer.zero_grad()
    X_final = model.state_dict()['X'].numpy()
    X_final /= (np.sum(np.abs(X_final)**2))**(1/2)

    return X_final, loss_evol

