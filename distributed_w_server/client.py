
import models
import torch
from collections import OrderedDict
from typing import  Dict, List, Tuple
import flwr as fl
import numpy as np
import utils
import os

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
      
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]        

    def fit(self, parameters, config):

        pfedme: bool = config['pfedme']

        if pfedme == True:
            new: bool = config['new']
            lambda_reg: int = config["lambda_reg"]
            local_rounds: int = config["local_rounds"]
            local_iterations: int= config["local_iterations"]
            lr: float= config["learning_rate"]
            mu: float= config["global_learning_rate"]

            self.set_parameters(parameters)
            
            global_params=models.train_pfedme(model=self.model, trainloader=self.trainloader,
            new=new, device=self.device, local_rounds=local_rounds, local_iterations=local_iterations, lambda_reg=lambda_reg,
            lr=lr, mu=mu)

            loss_person, accuracy_person = models.test(model=self.model, testloader=self.testloader, device=self.device)
            with torch.no_grad():     
             for param, global_param in zip(self.model.parameters(), global_params):
                    param = global_param
            loss_local, accuracy_local = models.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader), {"accuracy_local": float(accuracy_local),"accuracy_person": float(accuracy_person)}
        else:
            local_epochs: int = config['local_epochs']
            lr: float = config["learning_rate"]


            self.set_parameters(parameters)
            models.train_fedavg(model=self.model, trainloader=self.trainloader, local_epochs=local_epochs, device=self.device, lr=lr)
            loss_local, accuracy_local = models.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader), {"accuracy_local": float(accuracy_local)}
            
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = models.test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start MnistClient."""

    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data

    trainloader, testloader, _ = utils.load_part_of_data()
 
    # Start client
    client = FlowerClient(trainloader, testloader)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()