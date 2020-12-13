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
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from safety_model import SafetyModel
import math
import warnings
warnings.filterwarnings('ignore')

class Agent():
    def __init__(self,action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size):
        self.action_space = action_space
        self.frame_history_len = frame_history_len
        self.env = env
        self.device = device
        self.policy_qnet = model.DQN(self.frame_history_len,168,84,action_space,1e-4).to(self.device)
        self.target_qnet = model.DQN(self.frame_history_len,168,84,action_space,1e-4).to(self.device)
        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
        self.optimizer = self.policy_qnet.optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = exp_replay.ExperienceReplay(buffer_size)
        self.update_every = update_every
    def epsilon_greedy_act(self, state,vec, eps=0.0):
         #-----epsilon greedy-------------------------------------
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_space)
        else:
            #---set the network into evaluation mode(
            self.policy_qnet.eval()
            with torch.no_grad():
                action_values = self.policy_qnet(state.float().to(self.device),vec.float().to(self.device))
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
                        ('state','vec' ,'action', 'next_state','new_vec', 'reward','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.state))
        states = torch.cat(states).to(self.device)
        #print(states.size())
        vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.vec))
        vecs = torch.cat(vecs).to(self.device)



        actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),batch.action))
        actions = torch.cat(actions).to(self.device)
        #print(actions.size())
        rewards = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),batch.reward))
        rewards = torch.cat(rewards).to(self.device)
        #print(rewards.size())
        next_states = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.next_state))
        next_states = torch.cat(next_states).to(self.device)

        new_vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.new_vec))
        new_vecs = torch.cat(new_vecs).to(self.device)
        dones = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),batch.done))
        dones = torch.cat(dones).to(self.device)

        # Target = r + gamma*(max_a Q_target[next_state])
        action_values = self.target_qnet.forward(next_states,new_vecs).detach()

        max_action_values = action_values.max(1)[0].detach()
        target = rewards + gamma*max_action_values*(1-dones)
        current = self.policy_qnet.forward(states,vecs).gather(1,actions)
        target = target.reshape(32,1)

        loss = F.smooth_l1_loss(target, current)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_qnet.parameters():
            param.grad.data.clamp_(-1.2, 1.2)
        self.optimizer.step()


    def train(self,max_epsiodes,safety_model):
        global steps
        reward_track = []
        eps = self.epsilon_start
        for episode in tqdm(range(max_epsiodes)):
            print("-----------EPSISODE----",episode)
            obs = self.env.reset()
            state1 = self.torchify_state_dim(obs['camera'])
            state2 = self.torchify_state_dim(obs['lidar'])
            vec = torch.tensor([obs['state']])
            state = torch.cat((state1,state2)).view(1,4,168,84)
            #print(state.shape)
            total_reward = 0
            count_unsafe = 0
            for t in  range(100000):
                action = self.epsilon_greedy_act(state,vec,eps)
                #print(vec)
                safety_tag = safety_model(state.float(),vec.float(),torch.tensor([[action]],dtype=torch.float))
                #Safety Tag calibration
                if safety_tag[0][0].item() > 1e-2:
                    safety_tag=1
                else:
                    safety_tag=0

                if safety_tag==0:
                    next_state,reward,done,_ = self.env.step(action)
                elif safety_tag==1 or vec[0][3].item()==1.0:
                    print("unsafe action being suggested by model")
                    if vec[0][3].item()==1.0:
                        print("reason : maybe a vehicle ahead")
                        next_state,reward,done,_ = self.env.step(0)
                        reward = -20
                        count_unsafe+=1
                    else:
                        print("reason : maybe out of lane unsafe")
                        next_state,reward,done,_ = self.env.step(0)
                        count_unsafe += 1
                        reward = -10


                    if count_unsafe>6:
                        print("episode stopped due to safety issue!")
                        break
                next_state1 = next_state['camera']
                next_state2 = next_state['lidar']
                new_vec = next_state['state']
                if done:
                    next_state = torch.zeros(state.size())
                    new_vec = torch.tensor([new_vec])
                    done_flag=1
                else:
                    next_state1 = self.torchify_state_dim(next_state1)
                    next_state2 = self.torchify_state_dim(next_state2)
                    new_vec = torch.tensor([new_vec])
                    next_state = torch.cat((next_state1,next_state2)).view(1,4,168,84)
                    done_flag=0
                total_reward += reward
                reward = torch.tensor([reward],device = self.device)

                self.buffer.add(state,vec,action,next_state,new_vec,reward,done_flag)
                eps = max(eps * self.epsilon_decay, self.epsilon_min)
                #print("epsilon",eps)
                steps += 1

                if steps > 200:
                    #print("Gradient updating....")
                    #print(self.buffer.__len__())
                    self.update_gradients()
                    #print("done")
                    if steps%self.update_every==0:
                        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
                state = next_state
                if done:
                    print("episode completed normally")
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
      'display_size': 360,  # screen size of bird-eye render
      'max_past_step': 1,  # the number of past steps to draw
      'dt': 0.1,  # time interval between two frames
      'discrete': True,  # whether to use discrete control space
      'discrete_acc': [-3.0, 0.0,1.0,2.0],  # discrete value of accelerations
      'discrete_steer': [-0.2, -0.15,-0.1,-0.05, 0.0,0.05, 0.1,0.15, 0.2],  # discrete value of steering angles
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
      'out_lane_thres': 2.5,  # threshold for out of lane
      'desired_speed': 6,  # desired speed (m/s)
      'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
      'display_route': True,  # whether to render the desired route
      'pixor_size': 64,  # size of the pixor labels
      'pixor': False,  # whether to output PIXOR observation
    }
    writer = SummaryWriter()
    env = gym.make('carla-v0', params=params)
    env = make_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = 19
    frame_history_len = 4
    steps = 0
    buffer_size = 60000
    epsilon_start = 1
    epsilon_decay = 0.9999
    epsilon_min = 0.01
    update_every = 500
    batch_size = 32
    myagent = Agent(action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size)
    safety_model = SafetyModel(frame_history_len,168,84,action_space,1e-4)
    safety_model.load_state_dict(torch.load("SCM_model"))
    myagent.train(600,safety_model)
    torch.save(myagent.policy_qnet, "final_saved_model")
