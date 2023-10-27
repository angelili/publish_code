
import flwr as fl
import utils 
import torch
import models

from collections import OrderedDict
import flwr as fl
from flwr.common import Metrics


from torchvision.datasets import FashionMNIST
from typing import Callable, Optional, Tuple, Dict, Union, List

_, testset= utils.load_datasets()



#preparing empty dictionaries for metrics in pfedme, to be filled in each global round
training_history_acc_dist={"accuracy_global_pfedme": [], "accuracy_local_pfedme": [], "accuracy_personalized_pfedme":[]}
training_history_acc_cent={'accuracy_centralized_pfedme': []}
training_history_loss_dist={"loss_distributed_pfedme": []}
training_history_loss_cent={"loss_centralized_pfedme": []}

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
        model = model.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
       
       
        
        loss, accuracy = model.test(model, testset, device)
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


def fit_config(server_round: int):
    """Return training configuration dict for each round."""

    config = {'new': False,
        "lambda_reg":15,
        "local_rounds":120,
        "local_iterations":10,
        "learning_rate": 0.1,
        "global_learning_rate": 0.005 }
    return config



Strategy = fl.server.strategy.FedAvgM(
        min_fit_clients=5,
        min_evaluate_clients=6,
        min_available_clients=6,
        evaluate_fn=get_evaluate_fn_pfedme(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
        fit_metrics_aggregation_fn=agg_metrics_train_pfedme(training_history_acc_dist),
        evaluate_metrics_aggregation_fn=weighted_average_pfedme(training_history_acc_dist),
        on_fit_config_fn=fit_config,
       
       )