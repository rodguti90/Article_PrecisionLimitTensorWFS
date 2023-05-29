import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(42)

rng = np.random.default_rng(12345)

def complexdisk_rand(size=1):
    amp= np.sqrt(np.random.uniform(0,1,size))
    phase = np.random.uniform(0,2*np.pi,size)
    return amp*np.exp(1j*phase)

class OutputFields(nn.Module):
    def __init__(self, H0, H1, Xinit=None, n_opt_output=False):
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
        if type(n_opt_output) == int:
            self.opt_output = True
            self.transf = nn.utils.parametrizations.orthogonal(
                nn.Linear(n_out, n_opt_output, bias=False, dtype=torch.cfloat))
        else:
            self.opt_output = False

    def forward(self):
        Xnorm = self.X/(torch.sum(torch.abs(self.X)**2))**(1/2)
        Y0 = self.H0 @ Xnorm
        Y1 = self.H1 @ Xnorm
        if self.opt_output:
            return self.transf(Y0), self.transf(Y1)
        else:
            return Y0, Y1

def fisher_gauss_moim(Y0, Y1):
    Ymat1 =  Y1[:,None] * Y1[None,:].conj()
    Ymat0 =  Y0[:,None] * Y0[None,:].conj()
    L, _ = torch.linalg.eigh(Ymat1-Ymat0)
    return 1/torch.sum(L**2)

def fisher_gauss_b1o(Y0, Y1):
    Ymat1 =  Y1[:,None] * Y1[None,:].conj()
    Ymat0 =  Y0[:,None] * Y0[None,:].conj()
    L, _ = torch.linalg.eigh(Ymat1-Ymat0)
    return 1/torch.max(L**2)

def poisson_fisher(Y0, Y1):
    fisher = 4*torch.sum((torch.abs(Y1) - torch.abs(Y0))**2)#/torch.abs(Y1)**2)
    return 1/fisher

def gaussian_fisher(Y0, Y1):
    fisher = torch.sum((torch.abs(Y1)**2 - torch.abs(Y0)**2)**2)#/torch.abs(Y1)**2)
    return 1/fisher

def optimize_input(Xinit,TMs, n_epochs=200, lr=1e-2, noise='gaussian', n_opt_output=False): 
    model = OutputFields(TMs[0],
                    TMs[1],
                    Xinit=Xinit,
                    n_opt_output=n_opt_output)
    if noise == 'poisson':
        loss_fn = poisson_fisher
    elif noise == 'gaussian':
        loss_fn = gaussian_fisher
    elif noise == 'moim':
        loss_fn = fisher_gauss_moim
    elif noise == 'b1o':
        loss_fn = fisher_gauss_b1o
    else:
        raise ValueError('Invalid noise option')
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

    return 1/loss.item(), X_final, loss_evol

def optimize_input_list(inputs, TMs, n_epochs=200, lr=1e-2, noise='gaussian', n_opt_output=False):
    Xfinal_list = np.empty_like(inputs)
    fisherfinal_list = np.empty((len(inputs)))
    for k in range(len(inputs)):
        Xinit = inputs[k]
        fisherfinal_list[k], Xfinal_list[k],_= optimize_input(
            Xinit,TMs, n_epochs=n_epochs, lr=lr, noise=noise,n_opt_output=n_opt_output)
    
    return fisherfinal_list, Xfinal_list