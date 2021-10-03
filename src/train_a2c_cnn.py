from os.path import join, exists
from os import mkdir
import argparse
import gym
import math
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from collections import deque
from models.a2c_cnn import Actor,Critic
from models.vae import VariationalAutoencoder
from lib.consts import *
from utils.misc import save_checkpoint
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser("A2C training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')                    
args = parser.parse_args()

env_name = "CarRacing-v0" #"MountainCarContinuous-v0"  #Pendulum-v0 LunarLanderContinuous-v2
env = gym.make(env_name)

writer = SummaryWriter("logs/a2c_cnn")

print("action space: ", env.action_space.shape[0])
print("observation space ", env.env.observation_space.shape[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

GAMMA = 0.9
ENTROPY_BETA = 0.001  
CLIP_GRAD = .1
LR_c = 1e-3
LR_a = 1e-3

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.ToTensor()])

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#np.random.seed(0)
#env.seed(0)


output_shape = env.action_space.shape[0]

critic = Critic().to(device)
actor = Actor(output_shape).to(device)
c_optimizer = optim.RMSprop(params = critic.parameters(),lr = LR_c)
a_optimizer = optim.RMSprop(params = actor.parameters(),lr = LR_a)

max_episodes = 5000

steps = 0
max_steps = 200000

def test_net(count = 5):
    rewards = 0.0
    steps = 0
    entropys = 0.0
    for _ in range(count):
        obs = env.reset()

        while True:
            
            obs_v = transform(obs)
            mean_v, var_v = actor(obs_v.unsqueeze(0).to(device))
            action, _, entropy = sample(mean_v.cpu(), var_v.cpu()) #[0]
            obs, reward, done, info = env.step(action[0].numpy())
            env.env.viewer.window.dispatch_events()
            rewards += reward
            entropys += np.mean(entropy.detach().numpy())
            steps += 1
            if done:
                break

    return rewards/count, entropys/count, steps/count

def compute_returns(rewards,masks, gamma=GAMMA):
    R = 0 #pred.detach()
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.FloatTensor(returns).reshape(-1).unsqueeze(1)
    
def sample(mean, variance):
  '''
  Calcula as ações, log probs e entropia com base em uma distribuição normal por uma dada média e variância. 
  Calculates the actions, log probs and entropy based on a normal distribution by a given mean and variance.
  
  ====================================================
  
  calculate log prob:
  log_prob = -((action - mean) ** 2) / (2 * var) - log(sigma) - log(sqrt(2 *pi))
  
  calculate entropy:
  entropy =  0.5 + 0.5 * log(2 *pi * sigma) 
  entropy here relates directly to the unpredictability of the actions which an agent takes in a given policy.
  The greater the entropy, the more random the actions an agent takes.
  
  '''
  sigma = torch.sqrt(variance)
  m = Normal(mean, sigma)
  actions = m.sample()
  actions = np.clip(actions, env.action_space.low, env.action_space.high) # usually clipping between -1,1 but pendulum env has action range of -2,2
  logprobs = m.log_prob(actions)
  entropy = m.entropy()  # Equation: 0.5 + 0.5 * log(2 *pi * sigma)
    
  return actions, logprobs, entropy

def run_optimization(logprob_batch, entropy_batch, values_batch, rewards_batch, masks):
    '''
    Calcula a perda de ator e a perda de crítico e retrata-a através da Rede 
    Calculates the actor loss and the critic loss and backpropagates it through the Network
    
    ============================================
    Critic loss:
    c_loss = 0.5 * F.mse_loss(value_v, discounted_rewards)
    
    a_loss = (-log_prob_v * advantage).mean() + ENTROPY_BETA * entropy
    
    '''
    
    log_prob_v = torch.cat(logprob_batch).to(device)
    entropy_v = torch.cat(entropy_batch).to(device)
    value_v = torch.cat(values_batch).to(device)
    
    

    rewards_batch = torch.FloatTensor(rewards_batch)
    masks = torch.FloatTensor(masks)
    discounted_rewards = compute_returns(rewards_batch, masks).to(device)
    
    # critic_loss
    c_optimizer.zero_grad()
    critic_loss = 0.5 * F.mse_loss(value_v, discounted_rewards) #+ ENTROPY_BETA * entropy.detach().mean()
    critic_loss.backward()
    clip_grad_norm_(critic.parameters(),CLIP_GRAD)
    c_optimizer.step()
    
    # A(s,a) = Q(s,a) - V(s)
    advantage = discounted_rewards - value_v.detach() 

    #actor_loss
    a_optimizer.zero_grad()
    actor_loss = (-log_prob_v * advantage).mean() + ENTROPY_BETA * entropy.detach().mean()
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(),CLIP_GRAD)
    a_optimizer.step()
    
    
    
    return actor_loss, critic_loss, advantage
    
    

a2c_dir = join(args.logdir, 'a2c_cnn')
if not exists(a2c_dir):
    mkdir(a2c_dir)


reload_file = join(a2c_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}".format(state['epoch'],))
    actor.load_state_dict(state['actor_dict'])
    critic.load_state_dict(state['critic_dict'])
    c_optimizer.load_state_dict(state['c_optimizer'])
    a_optimizer.load_state_dict(state['a_optimizer'])
    
ep = 0    
cur_best = None
  
for _ in range(max_episodes):
    
    ep = ep + 1
    state = env.reset()

    done = False

    logprob_batch = []
    entropy_batch = []
    values_batch = []
    rewards_batch = []
    masks = []
    for step in range(max_steps):

        #env.render()
        state = transform(state)
        mean, variance = actor(state.unsqueeze(0).to(device))   
        action, logprob, entropy = sample(mean.cpu(), variance.cpu())
        value = critic(state.unsqueeze(0).to(device))

        next_state, reward, done, _ = env.step(action[0].numpy())
        env.env.viewer.window.dispatch_events()
        logprob_batch.append(logprob)
        entropy_batch.append(entropy)
        values_batch.append(value)
        rewards_batch.append(reward)  
        masks.append(1 - done)
        state = next_state

        if done:
          break
    
    print("\rEpisode: {} | Training Reward: {:.2f}\n".format(ep, sum(rewards_batch)))
    actor_loss, critic_loss, advantage = run_optimization(logprob_batch, entropy_batch, values_batch, rewards_batch, masks)
    writer.add_scalar('training reward', sum(rewards_batch), ep)
    writer.add_scalar('actor loss', actor_loss, ep)
    writer.add_scalar('critic loss', critic_loss, ep)
    writer.add_scalar('advantage', advantage.mean(), ep)         
    writer.flush()
    
    if ep != 0 and ep % 10 == 0:
        test_rewards, test_entropy, test_steps = test_net()
        print("\rEpisode: {} | Test Reward: {:.2f}".format(ep, test_rewards), end = "", flush = True)
        
        # checkpointing
        best_filename = join(a2c_dir, 'best.tar')
        filename = join(a2c_dir, 'checkpoint.tar')
        is_best = not cur_best or test_rewards < cur_best
        if is_best:
            cur_best = test_rewards

            save_checkpoint({
                'epoch': ep,
                'actor_dict': actor.state_dict(),
                'critic_dict': critic.state_dict(),
                'rewards': test_rewards,
                'c_optimizer': c_optimizer.state_dict(),
                'a_optimizer': a_optimizer.state_dict()
            }, is_best, filename, best_filename)

        
        writer.add_scalar('test steps', test_steps, ep)
        writer.add_scalar('test reward', test_rewards, ep)
        writer.add_scalar('test entropy', test_entropy, ep)         
        writer.flush()
        
        
