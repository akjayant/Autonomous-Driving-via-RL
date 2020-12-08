import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,frame_history_len,height_image,width_image,n_actions,lr):
        super(DQN,self).__init__()
        self.frame_history_len = frame_history_len
        self.convA1 = nn.Conv2d(self.frame_history_len,32,5)   # 4 images, 32 out channels, 5*5 krnel default stride=1
        self.convA2 = nn.Conv2d(32,64,5)
        self.convA3 = nn.Conv2d(64,128,5)


        x1 = torch.randn(self.frame_history_len,height_image,width_image).view(-1,frame_history_len,height_image,width_image)
        #print("Expected dimensions of input- ")
        #print(x.size())
        self.to_linearA = None   #auxillary variable to calculate shape of output of conv+max_pool
        self.convs(x1,'A')



        self.convB1 = nn.Conv2d(self.frame_history_len,32,5)   # 4 images, 32 out channels, 5*5 krnel default stride=1
        self.convB2 = nn.Conv2d(32,64,5)
        self.convB3 = nn.Conv2d(64,128,5)


        x2 = torch.randn(self.frame_history_len,height_image,width_image).view(-1,frame_history_len,height_image,width_image)
        #print("Expected dimensions of input- ")
        #print(x.size())
        self.to_linearB = None   #auxillary variable to calculate shape of output of conv+max_pool
        self.convs(x2,'B')

        self.fc1 = nn.Linear(self.to_linearA+self.to_linearB+4,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,n_actions)










        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(),lr = self.lr)

    def convs(self,x,flag):
        if flag=='A':
            x = F.max_pool2d(F.relu(self.convA1(x)),(2,2))
            x = F.max_pool2d(F.relu(self.convA2(x)),(2,2))
            x = F.max_pool2d(F.relu(self.convA3(x)),(2,2))
            #x = self.model(x)
            if self.to_linearA is None:
                self.to_linearA = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        else:
            x = F.max_pool2d(F.relu(self.convB1(x)),(2,2))
            x = F.max_pool2d(F.relu(self.convB2(x)),(2,2))
            x = F.max_pool2d(F.relu(self.convB3(x)),(2,2))
            if self.to_linearB is None:
                self.to_linearB = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x1,x2,vec):
        x1 = self.convs(x1,'A')
        x1 = x1.view(-1,self.to_linearA)

        x2 = self.convs(x2,'B')
        x2 = x2.view(-1,self.to_linearA)

        x = torch.cat([x1,x2,vec],dim=1)
        x = x.view(-1,self.to_linearA+self.to_linearA+4)
          #flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
