import utils
import client
import torch
import os
import flwr as fl

from flwr.common import Metrics


from torchvision.datasets import FashionMNIST
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Dict, Union, List

fedl_no_proxy=True
pFedMe=True
num_clients=10
num_rounds=100
batch_size=32

def main():

    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # partition dataset and get dataloaders
    trainloaders, testloaders, testset = utils.load_datasets(
        num_clients=num_clients,
        batch_size=batch_size,
    )
    # prepare function that will be used to spawn each client
    client_fn = client.generate_client_fn(trainloaders, testloaders)


    if pFedMe==True:
        #preparing empty dictionaries for metrics in pfedme, to be filled in each global round
        training_history_acc_dist={"accuracy_global_pfedme": [], "accuracy_local_pfedme": [], "accuracy_personalized_pfedme":[]}
        training_history_acc_cent={'accuracy_centralized_pfedme': []}
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

            config = {'pfedme':True,
                'new': False,
                "lambda_reg":15,
                "local_rounds":120,
                "local_iterations":10,
                "learning_rate": 0.1,
                "global_learning_rate": 0.005 }
            return config



        Strategy = fl.server.strategy.FedAvgM(
                min_fit_clients=9,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_fn=get_evaluate_fn_pfedme(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
                fit_metrics_aggregation_fn=agg_metrics_train_pfedme(training_history_acc_dist),
                evaluate_metrics_aggregation_fn=weighted_average_pfedme(training_history_acc_dist),
                on_fit_config_fn=fit_config,
            
            )

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            client_resources={'num_cpus': 1, 
                "num_gpus": 0.2},
            strategy=Strategy,
            ray_init_args={"address":"auto"} 

            
        )
    else:
        #preparing empty dictionaries for metrics in fedavg, to be filled in each global round
        training_history_acc_dist={"accuracy_global_fedavg": [], "accuracy_local_fedavg": []}
        training_history_acc_cent={'accuracy_centralized_fedavg': []}
        training_history_loss_cent={"loss_centralized_fedavg": []}

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

                dict_acc["accuracy_centralized_fedavg"].append(accuracy)
                dict_loss["loss_centralized_fedavg"].append(loss)
                # return statistics
                return loss, {"accuracy": accuracy}

            return evaluate


        def agg_metrics_train_pfedme(dict: Dict) -> Metrics:    
            def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
                # Multiply accuracy of each client by number of examples used
    
                accuracies_local = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
                examples = [num_examples for num_examples, _ in metrics]

                dict["accuracy_local_fedavg"].append(sum(accuracies_local)/sum(examples))


                # Aggregate and return custom metric (weighted average)
                return {"accuracy_local_fedavg": sum(accuracies_local)/sum(examples)}
            return evaluate

        def weighted_average_pfedme(dict: Dict) -> Metrics:
            # Multiply accuracy of each client by number of examples used
            def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
                accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
                examples = [num_examples for num_examples, _ in metrics]
                dict["accuracy_global_fedavg"].append(sum(accuracies)/sum(examples))
                # Aggregate and return custom metric (weighted average)
                return {"accuracy_global_fedavg": sum(accuracies) / sum(examples)}
            return evaluate


        def fit_config(server_round: int):
            """Return training configuration dict for each round."""

            config = {'pfedme': False,
                "local_epochs":2,
                "local_iterations":10,
                "learning_rate": 0.1, }
            return config



        Strategy = fl.server.strategy.FedAvgM(
                min_fit_clients=9,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_fn=get_evaluate_fn_pfedme(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
                fit_metrics_aggregation_fn=agg_metrics_train_pfedme(training_history_acc_dist),
                evaluate_metrics_aggregation_fn=weighted_average_pfedme(training_history_acc_dist),
                on_fit_config_fn=fit_config,
            
            )

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            client_resources={'num_cpus': 1, 
                "num_gpus": 0.2},
            strategy=Strategy,
            ray_init_args={"address":"auto"} 

            
        )

if __name__ == "__main__":
    main()

    