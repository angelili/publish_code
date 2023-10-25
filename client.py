from torch.utils.data import DataLoader
import model
import torch
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import flwr as fl
import numpy as np
# Flower Client
class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
    PyTorch."""

    def __init__(
        self,
        model: model.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device=device

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]        

    def fit(self, parameters, config):
        new: bool = config['new']
        lambda_reg: int = config["lambda_reg"]
        local_rounds: int = config["local_rounds"]
        local_iterations: int= config["local_iterations"]
        lr: float= config["learning_rate"]
        mu: float= config["global_learning_rate"]

    
        self.set_parameters(parameters)
        
        global_params=model.train_pfedme(model=self.model, trainloader=self.DataLoader,
        new=new, device=self.device, local_rounds=local_rounds, local_iterations=local_iterations, lambda_reg=lambda_reg,
        lr=lr, mu=mu)

        loss_person, accuracy_person = model.test(local_model=self.model, testloader=self.testloader, device=self.device)
        with torch.no_grad():     
          for param, global_param in zip(self.model.parameters(), global_params):
                param = global_param
        loss_global, accuracy_local = model.test(net=self.model, testloader=self.testloader, device=self.device)
        
        return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local),"accuracy_person": float(accuracy_person)}
        
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def gen_client_fn(
    trainloaders: List[DataLoader],
    testloaders: List[DataLoader]) -> Callable[[str], FlowerClient]: 

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        device = torch.device("cuda")

        # Load model
        model = model.Net().to(device)

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        testloader = testloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(model, trainloader, testloader, device)
    
    return client_fn