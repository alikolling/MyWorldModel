import argparse
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

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')

args = parser.parse_args()


torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.ToTensor()]) 

dataset_train = RolloutObservationDataset('datasets/carracing', transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing', transform_test, train=False)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

writer = SummaryWriter("logs/vae")

model = VariationalAutoencoder()

model = model.to(DEVICE)

learning_rate = 1e-3

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping(patience=100)


def vae_loss(recon_x, x, mu, logsigma):
    
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    kld_min = torch.zeros_like(kld) + KLD_TOLERANCE * LATENT_VEC
    kld_loss = torch.max(kld, kld_min)
    
    return BCE + kld_loss, BCE, kld_loss

def train(epoch):
    
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        image_batch_recon, latent_mu, latent_logvar = model(data)
        loss, bce, kld = vae_loss(image_batch_recon, data, latent_mu, latent_logvar)
        loss.backward()
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    
    train_loss /= len(train_loader.dataset)
    train_bce /= len(train_loader.dataset)
    train_kld /= len(train_loader.dataset)    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    
    writer.add_scalar('avg train loss', train_loss, epoch)
    writer.add_scalar('avg train bce loss', train_bce, epoch)
    writer.add_scalar('avg train kld loss', train_kld, epoch)         
    writer.flush() 
    
def test():
    
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    bce_test = 0
    kld_test = 0
                
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            image_batch_recon, latent_mu, latent_logvar = model(data)
            loss, bce, kld = vae_loss(image_batch_recon, data, latent_mu, latent_logvar)
            test_loss += loss.item()
            bce_test += bce.item()
            kld_test += kld.item()
                
    test_loss /= len(test_loader.dataset)
    bce_test /= len(test_loader.dataset)
    kld_test /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, bce_test , kld_test

vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))


reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.load_state_dict(state['scheduler'])
    early_stopping.load_state_dict(state['early_stopping'])

cur_best = None

for epoch in range(1, args.epochs + 1):

    train(epoch)
    writer.flush()
    test_loss, test_bce, test_kld = test()
    lr_scheduler(test_loss)
    early_stopping(test_loss)
    
    
    writer.add_scalar('avg test loss', test_loss, epoch)
    writer.add_scalar('avg test bce loss', test_bce, epoch)
    writer.add_scalar('avg test kld loss', test_kld, epoch)         
    writer.flush()
    
    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'early_stopping': early_stopping.state_dict()
    }, is_best, filename, best_filename)


    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LATENT_VEC).to(DEVICE)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if early_stopping.early_stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

