
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


from typing import Dict, List, Tuple

class Net(nn.Module):
  
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4* 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x
    

def train_pfedme(
        model: nn.Module,
        trainloader: DataLoader,
        new: bool,
        local_rounds: int,
        local_iterations: int,
        device: torch.device,
        lambda_reg: int,
        lr: float,
        mu: float):
    # Copy the parameters obtained from the server(global model)
    global_params = [val.detach().clone() for val in model.parameters()]
    model.train()
    # Local update on client 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for r in range(local_rounds):
        if new==False:
            data_iterator = iter(trainloader)
            data, target = next(data_iterator)
            data, target = data.to(device), target.to(device) #sample a batch
            for i in range(local_iterations):
                optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss = criterion(model(data), target) + (lambda_reg/2) * proximal_term
                loss.backward()
                optimizer.step()
        else:
            for i in range(local_iterations):  #for each local iteration, sample a batch
                data_iterator = iter(trainloader)
                data, target = next(data_iterator)
                data, target = data.to(device), target.to(device) #sample a batch
                optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss = criterion(model(data), target) + (lambda_reg/2) * proximal_term
                loss.backward()
                optimizer.step()
            
        #at the end each round after local_iterations, update the global model according to the personalized
        with torch.no_grad():
            for param, global_param in zip(model.parameters(), global_params):
                global_param -= mu * lambda_reg * (global_param - param)
    return global_params

def train_fedavg(model: nn.Module,
        trainloader: DataLoader,
        local_epochs: int,
        device: torch.device,
        lr: float):


    model.train()
    #local update on the client
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    for epoch in range(local_epochs):  # loop over the dataset multiple times

        for  data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
def test(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
   # model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
