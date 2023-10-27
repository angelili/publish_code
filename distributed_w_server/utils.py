from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split, DataLoader
from torchvision.datasets import FashionMNIST


import random




def load_part_of_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(),  transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST("/home/s124m21/publish_code/dataset", train=True, download=True, transform=transform)
    
    testset = FashionMNIST("/home/s124m21/publish_code/dataset", train=False, download=True, transform=transform)

    sample_size_train = random.randint(10000, 16000)
    sample_size_test =  int(sample_size_train*0.1)


    indices_train = random.sample(range(len(trainset)), sample_size_train)
    sampler_train= torch.utils.data.SubsetRandomSampler(indices_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, sampler=sampler_train)

    indices_test = random.sample(range(len(testset)), sample_size_test)
    sampler_test = torch.utils.data.SubsetRandomSampler(indices_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, sampler=sampler_test)

 
    return trainloader, testloader, DataLoader(testset, batch_size=32)

