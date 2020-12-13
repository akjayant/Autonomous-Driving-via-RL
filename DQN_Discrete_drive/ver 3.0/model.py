import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,frame_history_len,height_image,width_image,n_actions,lr):
        super(DQN,self).__init__()
        self.frame_history_len = frame_history_len
        self.conv1 = nn.Conv2d(self.frame_history_len,32,5)   # 4 images, 32 out channels, 5*5 krnel default stride=1
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)


        x = torch.randn(self.frame_history_len,height_image,width_image).view(-1,frame_history_len,height_image,width_image)
        #print("Expected dimensions of input- ")
        #print(x.size())
        self.to_linear = None   #auxillary variable to calculate shape of output of conv+max_pool
        self.convs(x)

        self.fc1 = nn.Linear(self.to_linear+4,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,n_actions)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(),lr = self.lr)

    def convs(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        #x = self.model(x)
        if self.to_linear is None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x,vec):
        x = self.convs(x)
        x = x.view(-1,self.to_linear)
        x = torch.cat([x,vec],dim=1)
        x = x.view(-1,self.to_linear+4)
          #flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
