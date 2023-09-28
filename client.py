import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net

class Client(object):
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = Net()
        self.opt = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.1)

        self.uploader_params = []
        self.uploader_grads = []

    def train(self, epochs, train_loader):
        self.model.train()
        for epoch in range(epochs):
            size = len(train_loader.dataset)
            self.model.train()
            for batch, (X, Y) in enumerate(train_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                self.opt.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss.backward()
                self.opt.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            self.scheduler.step()

    def upload(self):
        for param in self.model.parameters():
            self.uploader_params.append(param.data.clone())
            self.uploader_grads.append(param.grad.data.clone())
        self.uploader_grads = torch.cat([g.view(-1) for g in self.uploader_grads])

        self.uploader_params = torch.cat([p.view(-1) for p in self.uploader_params])

    
    def print_client(self):
        print(self.uploader_params)
        print(self.uploader_grads)