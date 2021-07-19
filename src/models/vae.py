import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.consts import *


class Encoder(nn.Module):
    '''
    Classe Encoder. Entrada observacao 64x64x3 e saida media e desvio padrao do tamanho do estado latente.
    Encoder Class. Input 64x64x3 observation and  output is the mean and standard deviation of latent state  
    size. 
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        
        self.fc_mu = nn.Linear(in_features=256*2*2, out_features=32)
        self.fc_logvar = nn.Linear(in_features=256*2*2, out_features=32)
            
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) 

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar



class Decoder(nn.Module):
    ''' 
    Classe Decoder. Entrada e o estado latente e a saida é a reconstrução da observação 64x64x3.
    Decoder Class. Input is the latent state and output is the 64x64x3 observation reconstruction. 
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(in_features=32, out_features=256*2*2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))
        return x
    


class VariationalAutoencoder(nn.Module):
    '''
    Classe VariationalAutoencoder. Utiliza as classes Encoder e Decoder. 
    VariationalAutoencoder class. It uses the Encoder and Decoder classes. 
    '''
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, encode=False):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        if encode:
            return latent
            
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
    #if self.training:
        # the reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)
    #else:
        #return mu
            
            

    

