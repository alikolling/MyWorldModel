'''
Arquivo com os hiperparametros.
File with hyperparameters. 
'''
import torch
import math


##### CONFIG

torch.set_printoptions(precision=10)
## CUDA variable from Torch
CUDA = torch.cuda.is_available()
#torch.backends.cudnn.deterministic = True
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


## VAE
LATENT_VEC = 32
KLD_TOLERANCE = 0.5

## RNN
HIDDEN_UNITS = 256
HIDDEN_DIM = 256
TEMPERATURE = 1.
GAUSSIANS = 5
NUM_LAYERS = 1
SEQ_LEN = 500
EPSILON = 1e-6

HEIGHT = 64
WIDTH = 64
SIZE = 64
RED_SIZE = 64

ACTION_SPACE = 3

## Training
BATCH_SIZE_VAE = 32
BATCH_SIZE_LSTM = 2


