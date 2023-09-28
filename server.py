
import torch
from client import Client

class Server():
    def __init__(self, clients):
        self.grads = []
        self.params = []
        self.G = []
        self.clients = clients
    
    def receive(self):
        for client in self.clients:
            self.grads.append(client.uploader_grads)
            self.params.append(client.uploader_params)
    
    def print_server(self):
        print(self.params)
        print(self.grads)

    def aggregate_grad(self):
        self.G = torch.cat(self.grads, dim=0)
        print(self.G.shape)

    