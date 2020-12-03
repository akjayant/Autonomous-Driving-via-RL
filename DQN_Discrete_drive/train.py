import gym
import model
import exp_replay
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
import matplotlib.pyplot as plt
from wrappers import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Agent():
    def __init__(self,action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size):
        self.action_space = action_space
        self.frame_history_len = frame_history_len
        self.env = env
        self.device = device
        self.policy_qnet = model.DQN(self.frame_history_len,168,84,11,1e-4).to(self.device)
        self.target_qnet = model.DQN(self.frame_history_len,168,84,11,1e-4).to(self.device)
        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
        self.optimizer = self.policy_qnet.optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = exp_replay.ExperienceReplay(buffer_size)
        self.update_every = update_every
    def epsilon_greedy_act(self, state, eps=0.0):
         #-----epsilon greedy-------------------------------------
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_space)
        else:
            #---set the network into evaluation mode(
            self.policy_qnet.eval()
            with torch.no_grad():
                action_values = self.policy_qnet(state.to(self.device))
            #----choose best action
            action = np.argmax(action_values.cpu().data.numpy())
            #----We need switch it back to training mode
            self.policy_qnet.train()
            return action

    def torchify_state_dim(self,obs):
        state = np.array(obs)
        #print(state.shape)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state).float()
        return state.unsqueeze(0)

    def update_gradients(self):
        gamma = 0.99
        if self.buffer.__len__()<self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        #Preparing batch
        experience = namedtuple('experience',
                        ('state', 'action', 'next_state', 'reward','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,device='cuda'),batch.state))
        states = torch.cat(batch.state).to(self.device)
        #print(states.size())
        actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),batch.action))
        actions = torch.cat(actions).to(self.device)
        #print(actions.size())
        rewards = list(map(lambda a: torch.tensor([a],device='cuda'),batch.reward))
        rewards = torch.cat(rewards).to(self.device)
        #print(rewards.size())
        next_states = list(map(lambda a: torch.as_tensor(a,device='cuda'),batch.next_state))
        next_states = torch.cat(next_states).to(self.device)
        dones = list(map(lambda a: torch.tensor([a],device='cuda'),batch.done))
        dones = torch.cat(dones).to(self.device)

        # Target = r + gamma*(max_a Q_target[next_state])
        action_values = self.target_qnet(next_states).detach()

        max_action_values = action_values.max(1)[0].detach()
        target = rewards + gamma*max_action_values*(1-dones)
        current = self.policy_qnet(states).gather(1,actions)
        target = target.reshape(32,1)

        loss = F.smooth_l1_loss(target, current)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_qnet.parameters():
            param.grad.data.clamp_(-1.2, 1.2)
        self.optimizer.step()


    def train(self,max_epsiodes):
        global steps
        reward_track = []
        eps = self.epsilon_start
        for episode in tqdm(range(max_epsiodes)):
            obs = self.env.reset()
            state1 = self.torchify_state_dim(obs['camera'])
            state2 = self.torchify_state_dim(obs['lidar'])
            state = torch.cat((state1,state2)).view(1,4,168,84)
            #print(state.shape)
            total_reward = 0
            for t in  range(10000):
                action = self.epsilon_greedy_act(state,eps)
                #print(action)
                if action==0:
                    action_=[-3.0,0]
                elif action==1:
                    action_=[-1.0,0]
                elif action==2:
                    action_=[0.5,0]
                elif action==3:
                    action_=[2.0,0.0]
                elif action==4:
                    action_=[3.0,0.0]
                elif action==5:
                    action_=[0.20,-0.3]
                elif action==6:
                    action_=[0.25,-0.2]
                elif action==7:
                    action_=[0.35,-0.1]

                elif action==8:
                    action_=[0.35,0.1]
                elif action==9:
                    action_=[0.25,0.2]
                elif action==10:
                    action_=[0.20,0.3]


                next_state,reward,done,_ = self.env.step(action_)
                next_state1 = next_state['camera']
                next_state2 = next_state['lidar']

                if done:
                    next_state = torch.zeros(state.size())
                    done_flag=1
                else:
                    next_state1 = self.torchify_state_dim(next_state1)
                    next_state2 = self.torchify_state_dim(next_state2)
                    next_state = torch.cat((next_state1,next_state2)).view(1,4,168,84)
                    done_flag=0
                total_reward += reward
                reward = torch.tensor([reward],device = self.device)

                self.buffer.add(state,action,next_state,reward.to('cpu'),done_flag)
                eps = max(eps * self.epsilon_decay, self.epsilon_min)
                #print("epsilon",eps)
                steps += 1

                if steps > 1000:
                    print("Gradient updating....")
                    print(self.buffer.__len__())
                    self.update_gradients()
                    print("done")
                    if steps%self.update_every==0:
                        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
                state = next_state
                if done:
                    reward_track.append(total_reward)
                    writer.add_scalar("Reward", total_reward)
                    break
            if episode%10 == 0:
                print("Episode no "+str(episode)+" reward = "+str(total_reward))
            if episode%100 == 0:
                torch.save(self.policy_qnet, "saved_model_checkpoint")


if __name__ == "__main__":
    params = {
      'number_of_vehicles': 100,
      'number_of_walkers': 0,
      'display_size': 480,  # screen size of bird-eye render
      'max_past_step': 1,  # the number of past steps to draw
      'dt': 0.1,  # time interval between two frames
      'discrete': False,  # whether to use discrete control space
      'discrete_acc': [-3.0,-2.0, 0.0,2.0, 3.0],  # discrete value of accelerations
      'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
      'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
      'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
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
      'desired_speed': 5,  # desired speed (m/s)
      'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
      'display_route': True,  # whether to render the desired route
      'pixor_size': 64,  # size of the pixor labels
      'pixor': False,  # whether to output PIXOR observation
    }
    writer = SummaryWriter()
    env = gym.make('carla-v0', params=params)
    env = make_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = 11
    frame_history_len = 4
    steps = 0
    buffer_size = 70000
    epsilon_start = 1
    epsilon_decay = 0.9999
    epsilon_min = 0.01
    update_every = 500
    batch_size = 32
    myagent = Agent(action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size)
    myagent.train(400)
    torch.save(myagent.policy_qnet, "final_saved_model")
