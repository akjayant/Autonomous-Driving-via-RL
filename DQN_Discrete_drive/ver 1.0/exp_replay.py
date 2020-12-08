import numpy as np
import random
from collections import namedtuple


experience = namedtuple("experience",('state','action','next_state','reward','done'))

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
