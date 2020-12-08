import gym
import modelB
import exp_replayB
import random
from wrappers import *
import torch
import torch.nn.functional as F
from collections import namedtuple
from tqdm import tqdm
from itertools import count
import gym_carla
import glob
import sys
import glob



def torchify_state_dim(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state).float()
    return state.unsqueeze(0)

def test(env, n_episodes, policy,device, render=False):
    #env = gym.wrappers.Monitor(env, './videos/' + 'qdrive_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state1 = torchify_state_dim(obs['camera'])
        state2 = torchify_state_dim(obs['lidar'])
        vec = torch.tensor([obs['state']])
        state1 = state1.view(1,4,84,84)
        state2 = state2.view(1,4,84,84)
        total_reward = 0.0
        for t in count():
            action = policy(state1.float().to(device),state2.float().to(device),vec.float().to(device)).max(1)[1].view(1,1)
            # if action==0:
            #     action_=[-3.0,0]
            # elif action==1:
            #     action_=[-1.0,0]
            # elif action==2:
            #     action_=[0.5,0]
            # elif action==3:
            #     action_=[2.0,0.0]
            # elif action==4:
            #     action_=[3.0,0.0]
            # elif action==5:
            #     action_=[0.20,-0.3]
            # elif action==6:
            #     action_=[0.25,-0.2]
            # elif action==7:
            #     action_=[0.35,-0.1]
            #
            # elif action==8:
            #     action_=[0.35,0.1]
            # elif action==9:
            #     action_=[0.25,0.2]
            # elif action==10:
            #     action_=[0.20,0.3]
            #
            # if render:
            #     env.render()
            #     time.sleep(0.02)

            next_state, reward, done, info = env.step(action[0][0].item())
            next_state1 = next_state['camera']
            next_state2 = next_state['lidar']
            new_vec = next_state['state']
            total_reward += reward

            if done:
                next_state1 = torch.zeros(state1.size())
                next_state2 = torch.zeros(state2.size())
                new_vec = torch.tensor([new_vec])
                done_flag=1
            else:
                next_state1 = torchify_state_dim(next_state1)
                next_state2 = torchify_state_dim(next_state2)
                new_vec = torch.tensor([new_vec])
                next_state1 = next_state1.view(1,4,84,84)
                next_state2 = next_state2.view(1,4,84,84)
                done_flag=0

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
      'number_of_vehicles': 100,
      'number_of_walkers': 0,
      'display_size': 480,  # screen size of bird-eye render
      'max_past_step': 1,  # the number of past steps to draw
      'dt': 0.1,  # time interval between two frames
      'discrete': True,  # whether to use discrete control space
      'discrete_acc': [-1.5,-1.0, 1.0,1.5,2.0],  # discrete value of accelerations
      'discrete_steer': [-0.25,-0.2, -0.15,-0.1,-0.05, 0.0,0.05, 0.1,0.15, 0.2,0.25],  # discrete value of steering angles
      'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
      'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
      'ego_vehicle_filter': 'model3',  # filter for defining ego vehicle
      'port': 2000,  # connection port
      'town': 'Town03',  # which town to simulate
      'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
      'max_time_episode': 1000,  # maximum timesteps per episode
      'max_waypt': 12,  # maximum number of waypoints
      'obs_range': 32,  # observation range (meter)
      'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
      'd_behind': 12,  # distance behind the ego vehicle (meter)
      'out_lane_thres': 2.0,  # threshold for out of lane
      'desired_speed': 7,  # desired speed (m/s)
      'max_ego_spawn_times': 1500,  # maximum times to spawn ego vehicle
      'display_route': True,  # whether to render the desired route
      'pixor_size': 64,  # size of the pixor labels
      'pixor': False,  # whether to output PIXOR observation
    }
    env = gym.make("carla-v0",params=params)
    env = make_env(env)
    agent = torch.load("final_saved_model")
    test(env, 5, agent,device)
