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
        self.policy_qnet = modelB.DQN(self.frame_history_len,84,84,action_space,1e-4).to(self.device)
        self.target_qnet = modelB.DQN(self.frame_history_len,84,84,action_space,1e-4).to(self.device)
        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
        self.optimizer = self.policy_qnet.optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = exp_replayB.ExperienceReplay(buffer_size)
        self.update_every = update_every
    def epsilon_greedy_act(self, state1,state2,vec, eps=0.0):
         #-----epsilon greedy-------------------------------------
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_space)
        else:
            #---set the network into evaluation mode(
            self.policy_qnet.eval()
            with torch.no_grad():
                action_values = self.policy_qnet(state1.float().to(self.device),state2.float().to(self.device),vec.float().to(self.device))
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
                        ('state1','state2','vec' ,'action', 'next_state1','next_state2','new_vec', 'reward','done'))
        batch = experience(*zip(*batch))
        states1 = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.state1))
        states1 = torch.cat(states1).to(self.device)


        states2 = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.state2))
        states2 = torch.cat(states2).to(self.device)
        #print(states.size())
        vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.vec))
        vecs = torch.cat(vecs).to(self.device)



        actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),batch.action))
        actions = torch.cat(actions).to(self.device)
        #print(actions.size())
        rewards = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),batch.reward))
        rewards = torch.cat(rewards).to(self.device)
        #print(rewards.size())
        next_states1 = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.next_state1))
        next_states1 = torch.cat(next_states1).to(self.device)

        next_states2 = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.next_state2))
        next_states2 = torch.cat(next_states2).to(self.device)

        new_vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.new_vec))
        new_vecs = torch.cat(new_vecs).to(self.device)
        dones = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),batch.done))
        dones = torch.cat(dones).to(self.device)

        # Target = r + gamma*(max_a Q_target[next_state])
        action_values = self.target_qnet.forward(next_states1,next_states2,new_vecs).detach()

        max_action_values = action_values.max(1)[0].detach()
        target = rewards + gamma*max_action_values*(1-dones)
        current = self.policy_qnet.forward(states1,states2,vecs).gather(1,actions)
        target = target.reshape(64,1)

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
            vec = torch.tensor([obs['state']])
            state1 = state1.view(1,4,84,84)
            state2 = state2.view(1,4,84,84)
            #print(state.shape)
            total_reward = 0
            for t in  range(100000):
                action = self.epsilon_greedy_act(state1,state2,vec,eps)
                next_state,reward,done,_ = self.env.step(action)
                next_state1 = next_state['camera']
                next_state2 = next_state['lidar']
                new_vec = next_state['state']
                if done:
                    next_state1 = torch.zeros(state1.size())
                    next_state2 = torch.zeros(state2.size())
                    new_vec = torch.tensor([new_vec])
                    done_flag=1
                else:
                    next_state1 = self.torchify_state_dim(next_state1)
                    next_state2 = self.torchify_state_dim(next_state2)
                    new_vec = torch.tensor([new_vec])
                    next_state1 = next_state1.view(1,4,84,84)
                    next_state2 = next_state2.view(1,4,84,84)
                    done_flag=0
                total_reward += reward
                reward = torch.tensor([reward],device = self.device)

                self.buffer.add(state1,state2,vec,action,next_state1,next_state2,new_vec,reward,done_flag)
                eps = max(eps * self.epsilon_decay, self.epsilon_min)
                #print("epsilon",eps)
                steps += 1

                if steps > 200:
                    print("Gradient updating....")
                    #print(self.buffer.__len__())
                    self.update_gradients()
                    print("done")
                    if steps%self.update_every==0:
                        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
                state = next_state
                if done:
                    reward_track.append(total_reward)
                    writer.add_scalar("Reward MULTIPATH", total_reward)
                    break
            if episode%10 == 0:
                print("Episode no "+str(episode)+" reward = "+str(total_reward))
            if episode%100 == 0:
                print("Episode ",episode)
                torch.save(self.policy_qnet, "saved_model_checkpoint")


if __name__ == "__main__":
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
    writer = SummaryWriter()
    env = gym.make('carla-v0', params=params)
    env = make_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = 35
    frame_history_len = 4
    steps = 0
    buffer_size = 80000
    epsilon_start = 1
    epsilon_decay = 0.9999
    epsilon_min = 0.01
    update_every = 500
    batch_size = 64
    myagent = Agent(action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size)
    myagent.train(1500)
    torch.save(myagent.policy_qnet, "final_saved_model")
