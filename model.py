import torch 
import torch.nn as nn


#define a normal neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.my_model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.my_model(x)
        return logits
    

tuan = Net()
print(tuan)
    