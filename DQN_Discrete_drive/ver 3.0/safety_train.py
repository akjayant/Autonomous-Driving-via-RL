import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from safety_model import SafetyModel
import exp_replay_safety
import torch.optim as optim
import gym
from wrappers import *
from collections import namedtuple
import gym_carla
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import sys

class CustomDataset(Dataset):
    def __init__(self,buffer,batch_size,device):
        self.buffer = buffer
        batch = self.buffer.sample(batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         #Preparing batch
        experience = namedtuple('experience',
                         ('state','vec' ,'action','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,dtype=torch.float),batch.state))
        self.states = torch.cat(states)
         #print(states.size())
        vecs = list(map(lambda a: torch.as_tensor(a,dtype=torch.float),batch.vec))
        self.vecs = torch.cat(vecs)



        actions = list(map(lambda a: torch.tensor([[a]]),batch.action))
        self.actions = torch.cat(actions)

        dones = list(map(lambda a: torch.tensor([a],dtype=torch.float),batch.done))
        self.dones = torch.cat(dones)
    def __len__(self):
        return self.buffer.__len__()
    def getlabel(self,idx):
        return self.dones[idx].item()
    def __getitem__(self,idx):
        #print("work!")
        return self.states[idx],self.vecs[idx],self.actions[idx],self.dones[idx]





def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def torchify_state_dim(obs):
    state = np.array(obs)
    #print(state.shape)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state).float()
    return state.unsqueeze(0)

def random_act(action_space):
    rnd = random.random()
    return np.random.randint(action_space)

def createbuffer(max_epsiodes,env,buffer):
    for episode in tqdm(range(max_epsiodes)):
        obs = env.reset()
        state1 = torchify_state_dim(obs['camera'])
        state2 = torchify_state_dim(obs['lidar'])
        vec = torch.tensor([obs['state']])
        state = torch.cat((state1,state2)).view(1,4,168,84)
        #print(state.shape)
        total_reward = 0
        for t in  range(100000):
            action = random_act(19)
            next_state,reward,done,_ = env.step(action)
            next_state1 = next_state['camera']
            next_state2 = next_state['lidar']
            new_vec = next_state['state']
            if done:
                next_state = torch.zeros(state.size())
                new_vec = torch.tensor([new_vec])
                done_flag=1
            else:
                next_state1 = torchify_state_dim(next_state1)
                next_state2 = torchify_state_dim(next_state2)
                new_vec = torch.tensor([new_vec])
                next_state = torch.cat((next_state1,next_state2)).view(1,4,168,84)
                done_flag=0


            buffer.add(state,vec,action,done_flag)
            state = next_state

            if done:
                break
        if episode%10 == 0:
            print("Buffer saved, episode=",episode)
            torch.save(buffer,"Buffer_dataset")
def train(epochs,SCM,dataloader_object,device):
    #criterion = nn.BCELoss()
    optimizer = SCM.optimizer
    epoch_acc = 0
    print("Gradient updating....")

    for ep in tqdm(range(epochs)):
        loss_ep = 0
        acc_ep = 0
        count=0
        for states,vecs,actions,dones in dataloader_object:
            states = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),states))

            states = torch.cat(states)
            batch_p = int(states.shape[0]/4)
            states = states.view(batch_p,4,168,84).to(device)


            vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),vecs))
            vecs = torch.cat(vecs).view(batch_p,4).to(device)

            actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),actions))
            actions = torch.cat(actions).view(batch_p,1).to(device)

            dones = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),dones))
            dones = torch.cat(dones).view(batch_p,1).to(device)

            logits = SCM.forward(states,vecs,actions).view(batch_p,1)
            dones = dones.view(batch_p,1)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(logits,dones)
            acc = binary_acc(logits,dones)
            loss_ep+=loss.item()
            acc_ep+=acc
            #print(batch_p)
            #print("Loss =", loss)
        #writer.add_scalar("Loss", loss.item())
            loss.backward()
            optimizer.step()
            count+=1
        print("loss :",loss_ep/count)
        print("accuracy :",acc_ep/count)
        if ep%50==0:
            print("model checkpont =",ep)
            torch.save(SCM.state_dict(), "SCM_model")



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
      'out_lane_thres': 2.0,  # threshold for out of lane
      'desired_speed': 5,  # desired speed (m/s)
      'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
      'display_route': True,  # whether to render the desired route
      'pixor_size': 64,  # size of the pixor labels
      'pixor': False,  # whether to output PIXOR observation
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = 19
    frame_history_len = 4
    steps = 0
    buffer_size = 10000
    batch_size = 32
    SCM = SafetyModel(frame_history_len,168,84,action_space,1e-4).to(device)
    print(SCM)
    train_flag=sys.argv[1]
    if train_flag==True:
        epochs=600
        buffer = torch.load('Buffer_dataset')
        dataset_size=buffer.__len__()
        dataset_object = CustomDataset(buffer,dataset_size,device)
        dataloader_object = DataLoader(dataset_object, batch_size=32,
                                num_workers=1,sampler=ImbalancedDatasetSampler(dataset_object))
        train(epochs,SCM,dataloader_object,device)
    else:
        env = gym.make('carla-v0', params=params)
        env = make_env(env)
        buffer = exp_replay_safety.ExperienceReplay(buffer_size)
        createbuffer(200,env,buffer)
