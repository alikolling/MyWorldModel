'''
Programa que roda o ambiente OpenAI Gym CarRacing-v0 
e armazena as observacoes, acoes, recompensas e estados finais.
As acoes de entrada do ambiente seguem uma politica browniana.

Program that runs the OpenAI Gym CarRacing-v0 environment
and stores the observations, actions, rewards and final states.
The input actions of the environment follow a Brownian policy.  
'''    

import sys
sys.path.append('/home/alisson/Dev/integrador/projeto_V2/src')

import gym_turtlebot3
import numpy as np
import torch
import time
import gym
import os
import rospy
from os.path import join, exists
import argparse
from utils.misc import sample_continuous_policy



def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914

    time.sleep(5)
    os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
    rospy.init_node('TurtleBot3_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
    env = gym.make('TurtleBot3_Circuit_Simple-v0', observation_mode=1, continuous=True)
    time.sleep(5)

    seq_len = 1000
    
    for i in range(rollouts):
        env.reset()
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seqlen)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1./50)
        
        
        s_rollout = []
        r_rollout = []
        d_rollout = []
        
        
        t = 0
        while True:
            
            action = a_rollout[t]
            t += 1
            
            s, r, done, _ = env.step(action)
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    
    os.system('gnome-terminal --tab --working-directory=WORK_DIR -- zsh -c "export '
                  'ROS_MASTER_URI=http://localhost:{}; export GAZEBO_MASTER_URI=http://localhost:{}; roslaunch '
                  'turtlebot3_gazebo turtlebot3_stage_4_with_camera.launch"'.format(11310 + 1, 11340 + 1))
    time.sleep(5)
    
    
    
    generate_data(args.rollouts, args.dir, args.policy)
