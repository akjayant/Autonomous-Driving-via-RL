import exp_replay_safety
import torch
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

buffer = torch.load('Buffer_dataset')
# batch_size = 3
# print(buffer.__len__())
# batch = buffer.sample(batch_size)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #Preparing batch
# experience = namedtuple('experience',
#                 ('state','vec' ,'action','done'))
# batch = experience(*zip(*batch))
# states = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.state))
# states = torch.cat(states).to(device)
# #print(states.size())
# vecs = list(map(lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float),batch.vec))
# vecs = torch.cat(vecs).to(device)
#
#
#
# actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),batch.action))
# actions = torch.cat(actions).to(device)
#
# dones = list(map(lambda a: torch.tensor([a],device='cuda',dtype=torch.float),batch.done))
# dones = torch.cat(dones).to(device)
#
# print(states.size())
# print(actions.size())
# print(dones)
#
# #a = torch.cat([states,vecs,actions,dones])
# #print(a)#
# #print(a.size())
#
# idx = 1
# print(batch)
# print(batch.state[idx])
# print(states[idx])
# print(vecs[idx])
# print(actions[idx])
# print(dones[idx])

class CustomDataset(Dataset):
    def __init__(self,buffer,batch_size):
        self.buffer = buffer
        batch = self.buffer.sample(batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         #Preparing batch
        experience = namedtuple('experience',
                         ('state','vec' ,'action','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,dtype=torch.float),batch.state))
        self.states = torch.cat(states)#.to(device)
         #print(states.size())
        vecs = list(map(lambda a: torch.as_tensor(a,dtype=torch.float),batch.vec))
        self.vecs = torch.cat(vecs)#.to(device)



        actions = list(map(lambda a: torch.tensor([[a]]),batch.action))
        self.actions = torch.cat(actions)#.to(device)

        dones = list(map(lambda a: torch.tensor([a],dtype=torch.float),batch.done))
        self.dones = torch.cat(dones)#.to(device)
    def __len__(self):
        return self.buffer.__len__()
    def getlabel(self,idx):
        return self.dones[idx].item()
    def __getitem__(self,idx):
        #print("work!")
        return self.states[idx],self.vecs[idx],self.actions[idx],self.dones[idx]


batch_size=buffer.__len__()
dataset_object = CustomDataset(buffer,batch_size)



dataloader_object = DataLoader(dataset_object, batch_size=4,
                        num_workers=1,sampler=ImbalancedDatasetSampler(dataset_object))
#
count=0
for i,a,b,c in dataloader_object:
     print("States",i)
     print("dones",c)
     print(i.size(),c.size())
     count+=1
     if count==1:
         break
