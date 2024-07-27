"""Client implementation for federated learning."""


from typing import Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
# from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from model import return_model, test, train
from utils import get_parameters, set_parameters

class FlowerClient(fl.client.NumPyClient):
    """Flower client for training with train and validation loss and accuracy
    that enables having training time in epochs or in batches."""

    # pylint: disable=R0902, R0913
    def __init__(
        self,
        cid: str,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        num_batches: int = None,
        method: str = 'base',
    ) -> None:
        """

        Parameters
        ----------
        net: torch.nn.Module
            PyTorch model
        trainloader, valloader, testloader: torch.utils.data.DataLoader
            dataloaders with images and labels
        device: torch.device
            denotes CPU or GPU training
        num_epochs: int
            training time for each client locally
        learning_rate: float
            learning rate used locally for model updates
        num_batches: int
            length of local training in batches (either this or num_epoch is used,
            if num_epoch is not None then num_epochs is used)
        """
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.method = method
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_batches = num_batches
        if self.num_epochs == 0:
            self.num_epochs = len(trainloader)
            # print(f'num_epochs = {self.num_epochs}, len(trainloader) = {len(trainloader)}, len(valloader) = {len(valloader)}')
        

    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.net)

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Fit locally training model."""
        set_parameters(self.net, parameters)
        train_loss, train_acc, val_loss, val_acc = train(
            self.net,
            self.trainloader,
            self.valloader,
            method= self.method,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            device=self.device,
            n_batches=self.num_batches,
            server_round=config["current_round"],
            unlearning_round=config["unlearning_round"],
        )
        return_dict: Dict[str, Scalar]
        if val_loss is None or val_acc is None:
            return_dict = {"train_loss": train_loss, "train_accuracy": train_acc}
        else:
            return_dict = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

        return get_parameters(self.net), len(self.trainloader), return_dict

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate locally training model."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, method=self.method, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


# pylint: disable=too-many-arguments
def create_client(
    cid: str,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    testloaders: List[DataLoader],
    device: torch.device = torch.device("cpu"),
    method: str = 'base',
    num_epochs: int = 0,
    learning_rate: float = 0.001,
    dataset: str = "FeMNIST",
    num_classes: int = 62,
    num_batches: int = None,
) -> FlowerClient:
    """Create client for the flower simulation."""
    net = return_model(dataset, num_classes).to(device)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    testloader = testloaders[int(cid)]
    # print(f'cid = {int(cid)}, len(trainloader) = {len(trainloader)}, len(valloader) = {len(valloader)}, len(testloader) = {len(testloader)}')

    return FlowerClient(
        cid,
        net,
        trainloader,
        valloader,
        testloader,
        device,
        num_epochs,
        learning_rate,
        num_batches,
        method = method,
    )
