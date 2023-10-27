import utils
import models

import torch
import os
import flwr as fl

from flwr.common import Metrics


from torchvision.datasets import FashionMNIST
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Dict, Union, List

import matplotlib.pyplot as plt


fedl_no_proxy=True
pFedMe=True
new=True
num_clients=10
num_rounds=100
batch_size=32

def get_evaluate_fn_pfedme(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire FashionMNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
    
    
        
        loss, accuracy = models.test(model, testset, device)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        #loss, accuracy = mnist.test_global(model, testloader, device)

        dict_acc["accuracy_centralized_pfedme"].append(accuracy)
        dict_loss["loss_centralized_pfedme"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def agg_metrics_train_pfedme(dict: Dict) -> Metrics:    
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies_person = [num_examples * m["accuracy_person"] for num_examples, m in metrics]
        accuracies_local = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_personalized_pfedme"].append(sum(accuracies_person)/sum(examples))
        dict["accuracy_local_pfedme"].append(sum(accuracies_local)/sum(examples))


        # Aggregate and return custom metric (weighted average)
        return {"accuracy_personalized": sum(accuracies_person)/sum(examples), "accuracy_local": sum(accuracies_local)/sum(examples)}
    return evaluate

def weighted_average_pfedme(dict: Dict) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict["accuracy_global_pfedme"].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global": sum(accuracies) / sum(examples)}
    return evaluate


def get_evaluate_fn_fedavg(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire FashionMNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
    
    
        
        loss, accuracy = models.test(model, testset, device)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        #loss, accuracy = mnist.test_global(model, testloader, device)

        dict_acc["accuracy_centralized_fedavg"].append(accuracy)
        dict_loss["loss_centralized_fedavg"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def agg_metrics_train_fedavg(dict: Dict) -> Metrics:    
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used

        accuracies_local = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_local_fedavg"].append(sum(accuracies_local)/sum(examples))


        # Aggregate and return custom metric (weighted average)
        return {"accuracy_local_fedavg": sum(accuracies_local)/sum(examples)}
    return evaluate

def weighted_average_fedavg(dict: Dict) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict["accuracy_global_fedavg"].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global_fedavg": sum(accuracies) / sum(examples)}
    return evaluate





def plot_training_history(training_history, path):
    plt.figure()
    # Iterate over each metric in the training history dictionary
    for metric, values in training_history.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric)

    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Training History')
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()
            
def main():

    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # partition dataset and get dataloaders
    _, _, testset = utils.load_part_of_data()

    if pFedMe==True:
        #preparing empty dictionaries for metrics in pfedme, to be filled in each global round
        training_history_acc_dist={"accuracy_global_pfedme": [], "accuracy_local_pfedme": [], "accuracy_personalized_pfedme":[]}
        training_history_acc_cent={'accuracy_centralized_pfedme': []}
        training_history_loss_cent={"loss_centralized_pfedme": []}

    
        def fit_config(server_round: int):
            """Return training configuration dict for each round."""

            config = {'pfedme':True,
                'new': new,
                "lambda_reg":15,
                "local_rounds":120,
                "local_iterations":10,
                "learning_rate": 0.1,
                "global_learning_rate": 0.005 }
            return config



        Strategy = fl.server.strategy.FedAvgM(
                min_fit_clients=3,
                min_evaluate_clients=4,
                min_available_clients=4,
                evaluate_fn=get_evaluate_fn_pfedme(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
                fit_metrics_aggregation_fn=agg_metrics_train_pfedme(training_history_acc_dist),
                evaluate_metrics_aggregation_fn=weighted_average_pfedme(training_history_acc_dist),
                on_fit_config_fn=fit_config,
            
            )


    else:
        
        #preparing empty dictionaries for metrics in fedavg, to be filled in each global round
        training_history_acc_dist={"accuracy_global_fedavg": [], "accuracy_local_fedavg": []}
        training_history_acc_cent={'accuracy_centralized_fedavg': []}
        training_history_loss_cent={"loss_centralized_fedavg": []}

    

        def fit_config(server_round: int):
            """Return training configuration dict for each round."""

            config = {'pfedme': False,
                "local_epochs": 2,
                "learning_rate": 0.1, }
            return config



        Strategy = fl.server.strategy.FedAvgM(
                min_fit_clients=3,
                min_evaluate_clients=4,
                min_available_clients=4,
                evaluate_fn=get_evaluate_fn_fedavg(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
                fit_metrics_aggregation_fn=agg_metrics_train_fedavg(training_history_acc_dist),
                evaluate_metrics_aggregation_fn=weighted_average_fedavg(training_history_acc_dist),
                on_fit_config_fn=fit_config,
            
            )

    


    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=Strategy)
    
    plot_training_history(training_history_acc_dist,'photo_1.png')
    plot_training_history(training_history_acc_cent,'photo_2.png')
    plot_training_history(training_history_loss_cent,'photo_3.png')


if __name__ == "__main__":
    main()

    