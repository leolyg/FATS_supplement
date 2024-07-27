"""Utils used for the FEMNIST project."""


from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.common import Metrics, Scalar
import matplotlib.pyplot as plt
from typing import Optional
import argparse
import copy
from sklearn.ensemble import RandomForestClassifier
from mblearn import AttackModels, ShadowModels
from sklearn.metrics import accuracy_score, precision_score, recall_score
from flwr.common.typing import NDArrays, Scalar


def get_parameters(net: torch.nn.Module) -> NDArrays:
    """Get parameters from a PyTorch network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: NDArrays) -> None:
    """Set parameters to a PyTorch network."""
    print(type(parameters))
    params_dict = zip(net.state_dict().keys(), parameters)

    state_dict = dict({k: torch.Tensor(v) if v.shape != torch.Size(
        []) else torch.Tensor([0]) for k, v in params_dict})
    # ignore argument type because Dict keeps order in the supported python versions
    net.load_state_dict(state_dict, strict=True)  # type: ignore


def mia_attack(
    # old_net,
    new_net,
    trainloaders,
    nonmemloaders,
        args):
    """Perform membership inference attack.
    Args:
        old_parameters: List of old parameters.
        new_parameters: List of new parameters.
        args: Arguments.
    Returns:
        acc, precision, recall
    """
    # move the old and new model to cpu
    new_net.eval()
    new_net.cpu()

    # create the unlearned data loader
    unle_dloaders = [trainloaders[cid] for cid in args.unlearning_clients]

    # create the learned data loader
    le_dloaders = [trainloaders[cid] for cid in range(
        len(trainloaders)) if cid not in args.unlearning_clients]

    rf_attack = RandomForestClassifier(n_estimators=100)
    attacker = AttackModels(
        target_classes=args.num_classes, attack_learner=rf_attack)
    # randomly get 100 samples from the learned data loader to train the attack model
    x, y = [], []
    if args.method == "FATS":
        for cid in range(len(le_dloaders)):
            for images, labels, indexs in le_dloaders[cid]:
                # make images and labels to be numpy array and convert images to 1D
                x.append(images.numpy().reshape(images.shape[0], -1))
                y.append(labels.numpy())
                if len(x) > 100000:
                    break

    else:
        for cid in range(len(le_dloaders)):
            for images, labels in le_dloaders[cid]:
                x.append(images.numpy().reshape(images.shape[0], -1))
                y.append(labels.numpy())
                if len(x) > 100000:
                    break

    x = np.concatenate(x)
    y = np.concatenate(y)
    if args.dataset == "shakespeare":
        y = np.argmax(y, axis=1)

    sh = ShadowModels(x, y, n_models=5, target_classes=args.num_classes,
                      learner=RandomForestClassifier(n_estimators=100))
    attacker.fit(sh.results)

    # get the output of the learned data loader and unlearned data loader from the new model
    x_mem, y_mem = [], []
    if args.method == "FATS":
        for cid in range(len(le_dloaders)):
            for images, labels, indexs in le_dloaders[cid]:
                output = new_net(images)
                output = torch.nn.functional.softmax(output, dim=1)
                x_mem.append(output.detach().numpy())
                y_mem.append(labels.numpy())
                if len(x_mem) > 100000:
                    break
    else:
        for cid in range(len(le_dloaders)):
            for images, labels in le_dloaders[cid]:
                output = new_net(images)
                output = torch.nn.functional.softmax(output, dim=1)
                x_mem.append(output.detach().numpy())
                y_mem.append(labels.numpy())
                if len(x_mem) > 100000:
                    break

    x_mem = np.concatenate(x_mem)
    y_mem = np.concatenate(y_mem)
    if args.dataset == "shakespeare":
        y_mem = np.argmax(y_mem, axis=1)

    x_nonmem, y_nonmem = [], []
    if args.method == "FATS":
        for cid in range(len(unle_dloaders)):
            for images, labels, indexs in unle_dloaders[cid]:
                output = new_net(images)
                output = torch.nn.functional.softmax(output, dim=1)
                x_nonmem.append(output.detach().numpy())
                y_nonmem.append(labels.numpy())

    else:
        for cid in range(len(unle_dloaders)):
            for images, labels in unle_dloaders[cid]:
                output = new_net(images)
                output = torch.nn.functional.softmax(output, dim=1)
                x_nonmem.append(output.detach().numpy())
                y_nonmem.append(labels.numpy())

    x_nonmem = np.concatenate(x_nonmem)
    y_nonmem = np.concatenate(y_nonmem)
    if args.dataset == "shakespeare":
        y_nonmem = np.argmax(y_nonmem, axis=1)

    res_mem = attacker.predict(x_mem, y_mem, batch=True)
    res_nonmem = attacker.predict(x_nonmem, y_nonmem, batch=True)
    ave_acc, ave_precision, ave_recall = [], [], []
    for _ in range(100):
        rand_index = np.random.choice(
            len(res_mem), len(res_nonmem), replace=False)
        y_pred = np.concatenate(
            (np.argmax(res_mem[rand_index], axis=1), np.argmax(res_nonmem, axis=1)))
        y_true = np.concatenate(
            (np.ones_like(y_mem[rand_index]), np.zeros_like(y_nonmem)))
        ave_acc.append(accuracy_score(y_true, y_pred))
        ave_precision.append(precision_score(y_true, y_pred))
        ave_recall.append(recall_score(y_true, y_pred))

    metrics = {
        "acc": np.mean(ave_acc),
        "precision": np.mean(ave_precision),
        "recall": np.mean(ave_recall)
    }

    return metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints
    and drops the other data types of the Metrics.
    """
    n_batches_list = [n_batches for n_batches, _ in metrics]
    n_batches_sum = sum(n_batches_list)
    metrics_lists: Dict[str, List[float]] = {}
    for number_of_batches, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for number_of_batches, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(
                    float(number_of_batches * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / n_batches_sum

    return weighted_metrics


def setup_seed(seed: int):
    """
    Set up seed for numpy and torch.
    Parameters
    ----------
    seed: int
        random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_metric_from_history(
    hist,
    save_plot_path,
    suffix: Optional[str] = "",
    metric_type: str = "distributed",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    expected_maximum : float
        The expected maximum accuracy from the original paper.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_dict = (
        hist.metrics_centralized
        if hist.metrics_centralized
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    fig = plt.figure()
    axis = fig.add_subplot(111)
    plt.plot(np.asarray(rounds), np.asarray(values), label="FedAvg")
    # Set expected graph for data
    # plt.axhline(
    #     y=expected_maximum,
    #     color="r",
    #     linestyle="--",
    #     label=f"Paper's best result @{expected_maximum}",
    # )
    # # Set paper's results
    # plt.axhline(
    #     y=0.99,
    #     color="silver",
    #     label="Paper's baseline @0.9900",
    # )
    # plt.ylim([0, 1])
    plt.title(f"{metric_type.capitalize()} Validation Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Set the apect ratio to 1.0
    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(save_plot_path+f"/{suffix}_{metric_type}_metrics.png")
    plt.close()
