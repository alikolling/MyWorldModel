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

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.ToTensor()]) 

dataset_test = RolloutObservationDataset('datasets/carracing', transform_test, train=False)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=True, num_workers=2)

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

test = next(iter(test_loader)).to(DEVICE) 
print(test.shape)   
sample,_,_ = model(test)
print(sample.shape)
save_image(sample,'teste.png')
save_image(test,'a.png')
