import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import model as m
from collections import namedtuple
import random
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

# class ReplayBuffer(object):
#     def __init__(self, max_size, input_shape, n_actions):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, *input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
#
#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size
#         print(state)
#         print(type(state))
#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = 1 - done
#         self.mem_cntr += 1
#
#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)
#
#         batch = np.random.choice(max_mem, batch_size)
#
#         states = self.state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         states_ = self.new_state_memory[batch]
#         terminal = self.terminal_memory[batch]
#
#         return states, actions, rewards, states_, terminal


experience = namedtuple("experience",('state','vec','action','reward','new_state','new_vec','done'))

class ExperienceReplay():
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self,*args):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience(*args)
        #circular queue kinda thing, old transitions will get replaced
        self.position = (self.position+1)%self.capacity
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)




class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=40000, layer1_size=100,
                 layer2_size=100, batch_size=32):
        self.gamma = gamma
        self.tau = tau
        self.memory = ExperienceReplay(max_size)
        self.batch_size = batch_size

        self.actor = m.ActorNetwork(alpha, 100,50,2, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='Actor')
        print(self.actor)
        self.critic = m.CriticNetwork(beta,100,50,2, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic')

        self.target_actor = m.ActorNetwork(alpha, 100,50,2, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor')
        self.target_critic = m.CriticNetwork(beta, 100,50,2, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=self.tau)

    def choose_action(self, observation,vec):
        self.actor.eval()

        mu = self.actor.forward(observation.float().to(self.actor.device),vec.float().to(self.actor.device))
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, vec, action, reward, new_state, new_vec, done):
        self.memory.add(state, vec,action, reward, new_state,new_vec, done)

    def learn(self):
        # if self.memory.mem_cntr < self.batch_size:
        #     return
        # state, action, reward, new_state, done = \
        #                               self.memory.sample_buffer(self.batch_size)
        #
        # reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        # done = T.tensor(done).to(self.critic.device)
        # new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        # action = T.tensor(action, dtype=T.float).to(self.critic.device)
        # state = T.tensor(state, dtype=T.float).to(self.critic.device)
        if self.memory.__len__()<self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        #Preparing batch
        experience = namedtuple('experience',
                        ('state', 'vec','action', 'reward', 'new_state','new_vec','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,dtype=torch.float,device='cuda'),batch.state))
        states = torch.cat(states).to(self.critic.device)

        vecs = list(map(lambda a: torch.as_tensor(a,dtype=torch.float,device='cuda'),batch.vec))
        vecs = torch.cat(vecs).to(self.critic.device)

        #print(states.size())
        actions = list(map(lambda a: torch.tensor(a,dtype=torch.float,device='cuda'),batch.action))
        actions = torch.cat(actions).to(self.critic.device)
        #print(actions.size())

        rewards = list(map(lambda a: torch.tensor([a],dtype=torch.float,device='cuda'),batch.reward))
        rewards = torch.cat(rewards).to(self.critic.device)
        #print(rewards.size())
        next_states = list(map(lambda a: torch.as_tensor(a,dtype=torch.float,device='cuda'),batch.new_state))
        next_states = torch.cat(next_states).to(self.critic.device)


        new_vecs = list(map(lambda a: torch.as_tensor(a,dtype=torch.float,device='cuda'),batch.new_vec))
        #print(new_vecs)
        #print(torch.tensor(new_vecs).size())

        new_vecs = torch.cat(new_vecs).to(self.critic.device)

        dones = list(map(lambda a: torch.tensor([a],device='cuda'),batch.done))
        dones = torch.cat(dones).to(self.critic.device)


        #renaming
        state=states
        new_state = next_states
        reward = rewards
        action = actions
        done = dones
        vec = vecs
        new_vec = new_vecs

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()


        target_actions = self.target_actor.forward(new_state,new_vec)

        critic_value_ = self.target_critic.forward(new_state,new_vec, target_actions)
        critic_value = self.critic.forward(state,vec, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)

        target = target.view(self.batch_size, 1)


        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state,vec)
        self.actor.train()
        actor_loss = -self.critic.forward(state,vec, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()
