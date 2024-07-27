"""Main module for running FEMNIST experiments."""


import pathlib
from functools import partial
from typing import Type, Union
import time
import json
import flwr as fl
import hydra
import pandas as pd
import argparse
import torch
from flwr.server.strategy import FedAvg
import numpy as np
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from model import return_model, test, train
import tempfile
from client import create_client
from dataset.dataset import (
    create_federated_dataloaders,
)
from flwr.common import (
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from strategy import FedAvgSameClients
from utils import setup_seed, weighted_average, plot_metric_from_history, mia_attack, set_parameters
from typing import Dict, Callable, Optional, Tuple, List
from collections import OrderedDict
from logging import WARNING, INFO
from model import test

def fit_config(server_round: int, unlearning_round: int = -1) -> Dict[str, Scalar]:
    """Return training configuration dict for each round.
    """
    config = {
        "current_round": server_round,
        "unlearning_round": unlearning_round,
    }
    return config


def get_evaluate_fn(net: torch.nn.Module, 
                    trainloader: DataLoader, 
                    central_testloader: DataLoader,
                    testloader: DataLoader, 
                    args: argparse.Namespace):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(
        server_round: int,
        new_parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # load the old parameters and conduct MIA attack on them
        # save_folder = args.results_dir_path + f'/history_{args.dataset}/'
        # old_parandarrays = np.load(
        #     save_folder + args.affix + f'_round{args.unlearning_round}_old_saved_parameters.npz')
        # old_parameters = [old_parandarrays[key] for key in sorted(old_parandarrays.keys())]
        # old_net = set_parameters(net, old_parameters)
        set_parameters(net, new_parameters)    
        loss, accuracy = test(net, central_testloader,  method=args.method, device=args.device)
        if server_round == args.num_rounds:
            metrics = mia_attack(net, trainloader, testloader, args)
            # save metrics to a file
            with open(args.results_dir_path + '/' + args.dataset + args.affix + '_mia_metrics.json', 'w') as f:
                json.dump(metrics, f)
            return loss, {'accuracy': accuracy, 'mia_metrics': metrics}

        return loss, {'accuracy': accuracy}

    return evaluate


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", version_base=None)
def main(cfg: OmegaConf):
    """Main function for running FEMNIST experiments."""
    # Ensure reproducibility
    setup_seed(cfg.random_seed)
    # Specify PyTorch device
    # pylint: disable=no-member
    with open_dict(cfg):
        cfg.affix = time.strftime("%Y%m%d-%H%M%S")
    log(INFO, "config: %s", cfg)
    # Save the results
    results_dir_path = pathlib.Path(cfg.results_dir_path)
    if not results_dir_path.exists():
        results_dir_path.mkdir(parents=True)

    device = torch.device(cfg.device)
    # Create datasets for federated learning
    trainloaders, valloaders, testloaders, central_testloader = create_federated_dataloaders(
        dataset=cfg.dataset,
        sampling_type=cfg.distribution_type,
        dataset_fraction=cfg.dataset_fraction,
        batch_size=cfg.batch_size,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
        random_seed=cfg.random_seed,
        method = cfg.method,
        min_samples_per_client = cfg.min_samples_per_client
    )

    net = return_model(cfg.dataset, cfg.num_classes).to(device)
    log(INFO, "net: %s", net)
    net_parameters = [val.cpu().numpy()
                        for _, val in net.state_dict().items()]
    # The total number of clients created produced from sampling differs (on different random seeds)
    total_n_clients = len(trainloaders)
    log(INFO, "Total number of clients: %s", total_n_clients)

    client_fnc = partial(
        create_client,
        trainloaders=trainloaders,
        valloaders=valloaders,
        testloaders=testloaders,
        device=device,
        method=cfg.method,
        num_epochs=cfg.epochs_per_round,
        learning_rate=cfg.learning_rate,
        # There exist other variants of the NIST dataset with different # of classes
        dataset=cfg.dataset,
        num_classes=cfg.num_classes,
        num_batches=cfg.batches_per_round,
    )
    flwr_strategy: Union[Type[FedAvg], Type[FedAvgSameClients]]
    if cfg.same_train_test_clients:
        #  Assign reference to a class
        flwr_strategy = FedAvgSameClients
    else:
        flwr_strategy = FedAvg

    strategy = flwr_strategy(
        args=cfg,
        min_available_clients=total_n_clients,
        # min number of clients to sample from for fit and evaluate
        # Keep fraction fit low (not zero for consistency reasons with fraction_evaluate)
        # and determine number of clients by the min_fit_clients
        # (it's max of 1. fraction_fit * available clients 2. min_fit_clients)
        fraction_fit=0.001,
        min_fit_clients=cfg.num_clients_per_round,
        fraction_evaluate=0.001,
        min_evaluate_clients=cfg.num_clients_per_round,
        evaluate_fn=get_evaluate_fn(net, trainloaders, central_testloader, testloaders, cfg), #  Leave empty since it's responsible for the centralized evaluation
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(net_parameters),
    )
    client_resources = None
    if device.type == "cuda":
        client_resources = {"num_gpus": 2.0}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fnc,  # type: ignore
        num_clients=total_n_clients,  # total number of clients in a simulation
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    distributed_history_dict = {}
    for metric, round_value_tuple_list in history.metrics_distributed.items():
        distributed_history_dict["distributed_test_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    for metric, round_value_tuple_list in history.metrics_distributed_fit.items():  # type: ignore
        distributed_history_dict["distributed_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    distributed_history_dict["distributed_test_loss"] = [
        val for _, val in history.losses_distributed
    ]

    results_df = pd.DataFrame.from_dict(distributed_history_dict)
    results_df.to_csv(results_dir_path / f"history_{cfg.dataset}_{cfg.affix}.csv")
    np.save(
        f"{cfg.results_dir_path}/history_{cfg.dataset}_{cfg.affix}",
        history,  # type: ignore
    )

    plot_metric_from_history(
        history,
        cfg.results_dir_path,
        cfg.dataset + cfg.affix,
        metric_type='distributed',
    )
    # save cfg to a yaml file with the same name as cfg.dataset + json.loads(cfg.affix)
    with open(cfg.results_dir_path + '/' + cfg.dataset + cfg.affix + '.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    log(INFO, "save history done")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
