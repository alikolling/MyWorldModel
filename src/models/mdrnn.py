import torch 
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
import numpy as np
from lib.consts import *

def gmm_loss(y, out_mu, out_sigma, out_pi): # pylint: disable=too-many-arguments
    '''
    Calcula a perda de gmm.
    
    Calcule menos o log de probabilidade do lote sob o modelo GMM descrito
    por mus, sigmas, pi. Precisamente, com bs1, bs2, ... os tamanhos do lote
    dimensões (várias dimensões de lote são úteis quando você tem um lote
    eixo e um eixo de passo de tempo), gs o número de misturas e fs o número de
    recursos. 
     
    Computes the gmm loss.
    
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    
    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    '''

    ok = Normal.arg_constraints["loc"].check(out_mu)
    bad_elements = out_mu[~ok]
    if bad_elements.data.size(dim=0) > 0:
        print(bad_elements)

    result = Normal(loc=out_mu, scale=out_sigma)
    y = y.view(-1, SEQ_LEN, 1, LATENT_VEC)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log(EPSILON + result)
    return torch.mean(result)
    


class _MDRNNBase(nn.Module):
    '''
    Classe base para a rede recorrente.
    Base class for recurring network .
    ''' 
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        
        self.tanh = nn.Tanh()
        self.z_pi = nn.Linear(self.hiddens, self.gaussians * self.latents)
        self.z_sigma = nn.Linear(self.hiddens, self.gaussians * self.latents)
        self.z_mu = nn.Linear(self.hiddens, self.gaussians * self.latents)
        self.z_rs = nn.Linear(self.hiddens, 1)
        self.z_ds = nn.Linear(self.hiddens, 1)

    def forward(self, *inputs):
        pass
        
        
class MDRNN(_MDRNNBase):
    '''
    Modelo MDRNN para vários passos à frente.  
    MDRNN model for multi steps forward. 
    '''
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(input_size=latents + actions, hidden_size=hiddens,batch_first=True)        
    
    def forward(self, actions, latents): # pylint: disable=arguments-differ
        ''' MULTI STEPS forward.
        :args actions: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args latents: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - pi_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - rs: (BSIZE, SEQ_LEN) torch tensor
            - ds: (BSIZE, SEQ_LEN) torch tensor
        '''
        
        sequence, bs = actions.size(1), actions.size(0)
        
        
        ins = torch.cat([actions, latents], dim=-1)
        z, _ = self.rnn(ins)
        z = self.tanh(z)
        
        pi = self.z_pi(z).view(-1, sequence, self.gaussians, self.latents)
        pi = f.softmax(pi, dim=2)
        pi = pi / TEMPERATURE

        sigmas = torch.exp(self.z_sigma(z)).view(-1, sequence, self.gaussians, self.latents)
        sigmas = sigmas * (TEMPERATURE ** 0.5)
        
        mus = self.z_mu(z).view(-1, sequence, self.gaussians, self.latents)
        
        rs = self.z_rs(z).view(-1, sequence)
        
        ds = self.z_ds(z).view(-1, sequence)
        
        return mus, sigmas, pi, rs, ds
        
        
class MDRNNCell(_MDRNNBase):
    '''
    Modelo MDRNN para um passo à frente. 
    MDRNN model for one step forward.'''
    def __init__(self, latents, actions, hidden, gaussians):
        super().__init__(latents, actions, hidden, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hidden)
        
    def forward(self, actions, latents, hidden):# pylint: disable=arguments-differ
        '''ONE STEP forward.
        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - pi_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        '''
        sequence = actions.size(0)
        in_al = torch.cat([actions, latents], dim=1)
        
        next_hidden = self.rnn(in_al, hidden)
        z = next_hidden[0]
        z = self.tanh(z)

        pi = self.z_pi(z).view(-1, sequence, self.gaussians, self.latents)
        pi = f.softmax(pi, dim=2)
        pi = pi / TEMPERATURE

        sigmas = torch.exp(self.z_sigma(z)).view(-1, sequence, self.gaussians, self.latents)
        sigmas = sigmas * (TEMPERATURE ** 0.5)
        
        mus = self.z_mu(z).view(-1, sequence, self.gaussians, self.latents)
        
        rs = self.z_rs(z).view(-1, sequence)
        
        ds = self.z_ds(z).view(-1, sequence)

        return mus, sigmas, pi, rs, ds, next_hidden
        
