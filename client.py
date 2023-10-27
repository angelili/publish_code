
import models
import torch
from collections import OrderedDict
from typing import  Dict, List, Tuple
import flwr as fl
import numpy as np
# Flower Client
class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
    PyTorch."""
    def __init__(self, trainloader, testloader) -> None:
        super().__init__()

        self.model = models.Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.model.to(self.device)

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
        pfedme: bool = config['pfedme']
        new: bool = config['new']
        lambda_reg: int = config["lambda_reg"]
        local_rounds: int = config["local_rounds"]
        local_epochs: int = config['local_epochs']
        local_iterations: int= config["local_iterations"]
        lr: float= config["learning_rate"]
        mu: float= config["global_learning_rate"]

    

    
        self.set_parameters(parameters)
        if pfedme == True:
            global_params=models.train_pfedme(model=self.model, trainloader=self.trainloader,
            new=new, device=self.device, local_rounds=local_rounds, local_iterations=local_iterations, lambda_reg=lambda_reg,
            lr=lr, mu=mu)

            loss_person, accuracy_person = models.test(model=self.model, testloader=self.testloader, device=self.device)
            with torch.no_grad():     
             for param, global_param in zip(self.model.parameters(), global_params):
                    param = global_param
            loss_local, accuracy_local = models.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local),"accuracy_person": float(accuracy_person)}
        else:
            models.train_fedavg(model=self.model, trainloader=self.trainloader, local_epochs=local_epochs, lr=lr)
            loss_local, accuracy_local = models.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local)}
            
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = models.test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def generate_client_fn(trainloaders, testloaders):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            trainloader=trainloaders[int(cid)], testloader=testloaders[int(cid)]
        )

    return client_fn
