from agent import Agent
import gym
from utils import plotLearning
from wrappers import *
from tqdm import tqdm
import gym_carla
import sys
import glob
import matplotlib.pyplot as plt
import wrappers as w
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from itertools import count

def torchify_state_dim(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state).float()
    return state.unsqueeze(0)


params = {
  'number_of_vehicles': 100,
  'number_of_walkers': 10,
  'display_size': 256,  # screen size of bird-eye render
  'max_past_step': 1,  # the number of past steps to draw
  'dt': 0.1,  # time interval between two frames
  'discrete': False,  # whether to use discrete control space
  'discrete_acc': [-3.0,-2.0, 0.0,2.0, 3.0],  # discrete value of accelerations
  'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
  'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
  'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
  'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
  'port': 2000,  # connection port
  'town': 'Town03',  # which town to simulate
  'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
  'max_time_episode': 1000,  # maximum timesteps per episode
  'max_waypt': 12,  # maximum number of waypoints
  'obs_range': 32,  # observation range (meter)
  'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
  'd_behind': 12,  # distance behind the ego vehicle (meter)
  'out_lane_thres': 2.0,  # threshold for out of lane
  'desired_speed': 8,  # desired speed (m/s)
  'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
  'display_route': True,  # whether to render the desired route
  'pixor_size': 64,  # size of the pixor labels
  'pixor': False,  # whether to output PIXOR observation
}
writer = SummaryWriter()
env = gym.make('carla-v0', params=params)
env = w.make_env(env)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[56448], tau=0.01, env=env,
              batch_size=64,  layer1_size=404, layer2_size=300, n_actions=2)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(500):

    obs = env.reset()
    state1 = torchify_state_dim(obs['camera'])
    state2 = torchify_state_dim(obs['lidar'])
    vec = torch.tensor([obs['state']])
    state = torch.cat((state1,state2)).view(1,2,100,50)
    done = False
    score = 0
    for t in count():
        act = agent.choose_action(state,vec)
        if t%10==0:
            print("chosen action",act[0])
        new_state, reward, done, info = env.step(act[0])
        next_state1 = new_state['camera']
        next_state2 = new_state['lidar']
        new_vec = new_state['state']
        if done:
            new_state = torch.zeros(state.size())
            new_vec = torch.tensor([new_vec])
        else:
            next_state1 = torchify_state_dim(next_state1)
            next_state2 = torchify_state_dim(next_state2)
            new_vec = torch.tensor([new_vec])
            new_state = torch.cat((next_state1,next_state2)).view(1,2,100,50)
        agent.remember(state, vec, act, reward, new_state, new_vec, int(done))
        agent.learn()
        score += reward
        state = new_state
        if done:
            writer.add_scalar("Reward", score)
            break
        #env.render()
    if i % 100 == 0:
        agent.save_models()
