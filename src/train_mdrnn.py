''' Recurrent model training '''
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.learning import EarlyStopping
from utils.learning import LRScheduler
from lib.consts import *
from data.loaders import RolloutSequenceDataset
from models.vae import VariationalAutoencoder
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

writer = SummaryWriter("logs/rnn")

# constants
epochs = 400


# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
vae_state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          vae_state['epoch'], vae_state['precision']))
          
vae = VariationalAutoencoder().to(DEVICE)
vae.load_state_dict(vae_state['state_dict'])


# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)



mdrnn = MDRNN(LATENT_VEC, ACTION_SPACE, HIDDEN_UNITS, GAUSSIANS)
mdrnn.to(DEVICE)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping(patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    lr_scheduler.load_state_dict(rnn_state["scheduler"])
    early_stopping.load_state_dict(rnn_state["early_stopping"])
    e = rnn_state["epoch"]
else:
    e = 0    
  
# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    batch_size=BATCH_SIZE_LSTM, num_workers=4, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BATCH_SIZE_LSTM, num_workers=4)

def to_latent(obs, next_obs):
    '''
    Transforme as observações em espaço latente. 
    
    Transform observations to latent space.
    :args obs: 5D torch tensor (BATCH_SIZE_LSTM, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BATCH_SIZE_LSTM, SEQ_LEN, ASIZE, SIZE, SIZE)
    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BATCH_SIZE_LSTM, SEQ_LEN, LATENT_VEC)
        - next_latent_obs: 4D torch tensor (BATCH_SIZE_LSTM, SEQ_LEN, LATENT_VEC)
    '''
    
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]
    
        latent_obs, latent_next_obs = [vae(x, encode=True).view(BATCH_SIZE_LSTM, SEQ_LEN, LATENT_VEC)
         for x in (obs, next_obs)]
    
    return latent_obs, latent_next_obs
    
def get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool):
    '''
    Calcule as perdas.
    
     A perda calculada é:
     (GMMLoss (latent_next_obs, GMMPredicted) + MSE (recompensa, predicted_reward) +
          BCE (terminal, logit_terminal)) / (LSIZE + 2)
     O fator LSIZE + 2 está aqui para neutralizar o fato de que as escalas do GMMLoss
     aproximadamente linearmente com LSIZE. Todas as perdas são calculadas em média tanto no
     lote e as dimensões da sequência (as duas primeiras dimensões). 
      
    Compute losses.
    
    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).
    
    :args latent_obs: (BATCH_SIZE_LSTM, SEQ_LEN, LATENT_VEC) torch tensor
    :args action: (BATCH_SIZE_LSTM, SEQ_LEN, ACTION_SPACE) torch tensor
    :args reward: (BATCH_SIZE_LSTM, SEQ_LEN) torch tensor
    :args latent_next_obs: (BATCH_SIZE_LSTM, SEQ_LEN, LATENT_VEC) torch tensor
    
    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    '''

    '''latent_obs, action, reward, terminal, latent_next_obs = [arr.transpose(1, 0)
                                for arr in [latent_obs, action,
                                    reward, terminal,
                                    latent_next_obs]]'''

    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)

    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LATENT_VEC + 2
    else:
        mse = 0
        scale = LATENT_VEC + 1
    
    loss = (gmm + bce + mse) 
    
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)
  
  
def data_pass(epoch, train, include_reward):# pylint: disable=too-many-locals
    ''' One pass through the data'''
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader
        
    loader.dataset.load_next_buffer()
    
    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0
    
    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(DEVICE) for arr in data]
        
        # transform obs to latent
        latent_obs, latent_next_obs = to_latent(obs, next_obs)
        
        if train:
            losses = get_loss(latent_obs, action, reward, 
                            terminal, latent_next_obs, include_reward)
            
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward, 
                                terminal, latent_next_obs, include_reward)    
    
        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']
        
        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BATCH_SIZE_LSTM)
    
       
    pbar.close()
    return cum_loss * BATCH_SIZE_LSTM / len(loader.dataset)
    
train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None

for ep in range(epochs-e):
    e = e + 1
    train(e)
    test_loss = test(e)
    writer.add_scalar('test_loss', test_loss, e)
    writer.flush()
    lr_scheduler(test_loss)
    early_stopping(test_loss)
    
    is_best = not cur_best or test_loss < cur_best
    
    if is_best:
        cur_best = test_loss
        
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'early_stopping': early_stopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

    if early_stopping.early_stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
