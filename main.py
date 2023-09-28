import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from client import Client
from server import Server


training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False,
    transform = ToTensor(),
)

bs = 64

train_dataloader = DataLoader(training_data, batch_size = bs)
test_dataloader = DataLoader(test_data, batch_size = bs)



client1 = Client()
client2 = Client()
client3 = Client()
server_sample = Server([client1, client2, client3])

client1.train(1, train_dataloader)
client1.upload()
client2.train(1, train_dataloader)
client2.upload()
client3.train(1, train_dataloader)
client3.upload()

server_sample.receive()

# server_sample.print_server()

# client1.print_client()
# client2.print_client()
# client3.print_client()

server_sample.print_server()
server_sample.aggregate_grad()






