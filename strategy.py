"""FedAvg with the same clients used for both training and evaluation."""
import utils
import copy
import torch
from collections import OrderedDict
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union
import json
import random
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
import argparse
import os
import numpy as np
import ray
from dataset.dataset import (
    cifar10_train_valid_test_partition_selected,
    transform_datasets_into_dataloaders
)
import model


def sample_unlearning_clients(availiable_clients, sample_size, args):
    """Sample clients for unlearning."""
    unlearning_clients = [str(i) for i in args.unlearning_clients]
    sampled_clients = unlearning_clients
    while any(x in unlearning_clients for x in sampled_clients):
        if args.method == 'FATS':
            sampled_clients = random.choices(availiable_clients, k=sample_size)
            sampled_clients = list(set(sampled_clients))
        else:
            sampled_clients = random.sample(availiable_clients, sample_size)

    return sampled_clients


def sample_clients(availiable_clients, sample_size, args):
    """Sample clients."""
    if args.method == 'FATS':
        sampled_clients = random.choices(availiable_clients, k=sample_size)
        sampled_clients = list(set(sampled_clients))
    else:
        sampled_clients = random.sample(availiable_clients, sample_size)

    return sampled_clients


class FedAvgSameClients(FedAvg):
    """FedAvg that samples clients for each round only once (the same clients
    are used for training and testing round n)

    It does not mean that the same client are used in each round. It used just the same clients
    (with different parts of their data) in round i.

    It assumes that there is no different function for evaluation - on_evaluate_config_fn
    (it's ignored).
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[
            int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self._current_round_fit_clients_fits_list: List[Tuple[ClientProxy, FitIns]] = [
        ]
        self.args = args
        # create a dict to save the clients records with 1000 keys
        self.clients_records = {i: None for i in range(1, args.num_rounds+1)}
        self.parameters_records = {
            i: None for i in range(1, args.num_rounds+1)}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(
                server_round, self.args.unlearning_round)

        if server_round == self.args.unlearning_round:
            # detect the first round of unlearning clients based on the history _send_clients files
            # load the history _send_clients files
            self.args.unlearning = True
            save_folder = self.args.results_dir_path + \
                f'/history_{self.args.dataset}'
            # save the current parameters at first
            np.savez(save_folder + self.args.affix +
                     f'_round{server_round}_old_saved_parameters.npz', *parameters_to_ndarrays(parameters))

            # get the first round of existing clients
            earliest_round = server_round - 1
            for round in range(1, server_round):
                log(INFO, "round: %s, clients %s ",
                    round, self.clients_records[round])
                for client in self.clients_records[round]:
                    if client in self.args.unlearning_clients:
                        earliest_round = min(earliest_round, round)

            if self.args.method == 'base':
                earliest_round = 1

            # load the parameters of the earliest round
            # parandarrays = np.load(
            #     save_folder + f'{self.args.affix}_round{earliest_round}_send_parameters.npz')
            # sort the parandarrays by keys and convert to parameters
            # parandarrays = [parandarrays[key] for key in sorted(parandarrays.keys())]
            # parameters = ndarrays_to_parameters(parandarrays)
            parameters_object_id = self.parameters_records[earliest_round]
            parameters = ray.get(parameters_object_id)

            if self.args.unlearning_samples not in [None, 'None']:
                print(self.args.unlearning_samples)
                # for example, if unlearning_samples = 0.5, then just unlearn the first 50% samples of the selected clients
                weights_results = recover_parameters_at_sample_level(parameters_to_ndarrays(
                    parameters), self.args.unlearning_clients, self.args.unlearning_samples, self.args)
                parameters = ndarrays_to_parameters(aggregate(weights_results))

            # release the memory of the parameters of the previous round in ray
            self.parameters_records = {
                i: None for i in range(1, self.args.num_rounds+1)}
            self.clients_records = {
                i: None for i in range(1, self.args.num_rounds+1)}

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.sample(
        #     num_clients=sample_size, mode=json.loads(self.args.mode), min_num_clients=min_num_clients
        # )

        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = sample_size
        client_manager.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(client_manager.clients)

        if sample_size > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                sample_size,
            )
            return []

        if self.args.unlearning:
            sampled_cids = sample_unlearning_clients(
                available_cids, sample_size, self.args)
        else:
            sampled_cids = sample_clients(
                available_cids, sample_size, self.args)

        # Save the sampled clients and the parameters send to them
        # check if the directory exists
        if parameters is not None and server_round < self.args.unlearning_round:
            # save_folder = self.args.results_dir_path + f'/history_{self.args.dataset}/'
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            # parandarrays: List[np.ndarray] = parameters_to_ndarrays(parameters)
            # np.savez(save_folder + self.args.affix + f'_round{server_round}_send_parameters.npz', *parandarrays)
            self.clients_records[server_round] = [
                int(cid) for cid in sampled_cids]
            parameters_object_id = ray.put(parameters)
            self.parameters_records[server_round] = parameters_object_id
            # np.save(save_folder + self.args.affix + f'_round{server_round}_send_clients.npy', sampled_cids)

        # Create FitIns for each client
        clients = [client_manager.clients[cid] for cid in sampled_cids]
        self._current_round_fit_clients_fits_list = [
            (client, fit_ins) for client in clients]
        # Return client/config pairs
        return self._current_round_fit_clients_fits_list

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Keep the fraction_settings for consistency reasons
        if self.fraction_evaluate == 0.0:
            return []
        evaluate_config = []
        for tuple_client_proxy_fit_ins in self._current_round_fit_clients_fits_list:
            eval_ins = EvaluateIns(
                tuple_client_proxy_fit_ins[1].parameters,
                tuple_client_proxy_fit_ins[1].config,
            )
            evaluate_config.append((tuple_client_proxy_fit_ins[0], eval_ins))

        return evaluate_config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate(weights_results))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics)
                            for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                eval_metrics)
            log(
                INFO,
                "server_round %s, metrics_aggregated %s",
                str(server_round),
                str(metrics_aggregated),
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


def recover_parameters_at_sample_level(parameters, unlearning_clients, unlearning_samples, args):
    """
    Recover the parameters at sample level.
    train the parameters without the unlearning_samples in the unlearning_clients sequentially,
    and then average the parameters.
    """
    # build the model and load the parameters
    net_init = model.return_model(args.dataset, args.num_classes)
    utils.set_parameters(net_init, parameters)
    num_clients = int(args.dataset_fraction)
    partitioned_train, _, _ = cifar10_train_valid_test_partition_selected(
        num_clients, args.validation_fraction, args.distribution_type, args.random_seed, args.method, args.dataset, unlearning_clients
    )
    # build the trainset without the unlearning_samples
    partitioned_train_new = []
    for i in range(len(partitioned_train)):
        partitioned_train_new.append(torch.utils.data.Subset(partitioned_train[i], range(
            int(len(partitioned_train[i]) * unlearning_samples), len(partitioned_train[i]))))

    trainloaders = transform_datasets_into_dataloaders(
        partitioned_train_new, batch_size=args.batch_size
    )
    parameters_list = []
    # for each client, train the model without the unlearning_samples
    for i in range(len(unlearning_clients)):
        # train the model on the trainloaders
        net = copy.deepcopy(net_init)
        train_loss, train_acc, val_loss, val_acc = model.train(
            net,
            trainloaders[i],
            [],
            method=args.method,
            epochs=args.epochs_per_round,
            learning_rate=args.learning_rate,
            device=args.device,
            n_batches=args.batches_per_round,
            server_round=0,
            unlearning_round=1,
        )
        parameters_list.append(utils.get_parameters(net))

    weights_results = [
        (para, len(trainloaders[i].dataset))
        for i, para in enumerate(parameters_list)
    ]
    return weights_results
