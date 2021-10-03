from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.vae import VariationalAutoencoder

from lib.consts import *
from utils.learning import EarlyStopping
from utils.learning import LRScheduler

from torch.utils.data import DataLoader
from data.loaders import RolloutObservationDataset

from utils.misc import save_checkpoint
import matplotlib.pyplot as plt
import numpy as np

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.ToTensor()]) 

dataset_test = RolloutObservationDataset('datasets/carracing', transform_test, train=False)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=2)

model = VariationalAutoencoder()

model = model.to(DEVICE) 

vae_dir = join('exp_dir', 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'teste'))


reload_file = join(vae_dir, 'best.tar')
if not False and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])

#test = next(iter(test_loader)).to(DEVICE) 
#print(test.shape)   
#sample,_,_ = model(test)
#print(sample.shape)
#save_image(sample,'reconstrução.png')
#save_image(test,'original.png')
rnd = torch.randn(2,LATENT_VEC).to(DEVICE)
rnd_sample = model.decoder(rnd).cpu()
np.savetxt("rnd.csv", rnd.cpu().numpy() ,delimiter =", ", fmt ='% s')
save_image(rnd_sample,'random.png')
rnd1 = torch.randn(2,LATENT_VEC).to(DEVICE)
rnd_sample1 = model.decoder(rnd1).cpu()
np.savetxt("rnd1.csv", rnd1.cpu().numpy() ,delimiter =", ", fmt ='% s')
save_image(rnd_sample1,'random1.png')
