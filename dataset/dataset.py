"""FEMNIST dataset creation module."""


import pathlib
from logging import INFO
from typing import List, Tuple

import numpy as np
import pandas as pd
import json
import torch
import torchvision.transforms as transforms
from dataset.language_utils import word_to_indices, letter_to_vec
from flwr.common.logger import log
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, BatchSampler, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN

from dataset.partition_noniid import (
    get_cifar_10,
    get_svhn,
    get_fashion_mnist,
    get_mnist,
    create_lda_partitions,
    cifar10Transformation,
    svhnTransformation,
    fmnistTransformation,
    mnistTransformation,
)
from dataset.nist_preprocessor import NISTPreprocessor
from dataset.nist_sampler import NistSampler
from dataset.nist_dataset import create_dataset, create_partition_list
from dataset.zip_downloader import ZipDownloader

import os
from collections import defaultdict


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class CELEBA_Client(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    # ./preprocess.sh -s iid --iu 0.02 --sf 0.01 -k 100 -t sample

    def __init__(self, dataclient, train=True, method='base', max_client=-1, transform=None, target_transform=None, root="/home/happy/Documents/datasets/leaf/data/celeba/data"
                 ):
        super().__init__()
        if transform is None:
            self.transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                                 transforms.Resize((128, 128)),
                                                 # transforms.Grayscale(),
                                                 # transforms.Lambda(lambda x: x/255.),
                                                 transforms.ToTensor()])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.method = method

        if max_client != -1:
            train_clients = train_clients[:max_client]

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            self.dic_users = set()
            l = len(train_data_x)
            cur_x = dataclient['x']
            cur_y = dataclient['y']
            for j in range(len(cur_x)):
                self.dic_users.add(j + l)
                # load the image as numpy array based on the path
                img = np.asarray(Image.open(
                    root + '/raw/img_align_celeba/' + cur_x[j]))
                train_data_x.append(img)
                train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            cur_x = dataclient['x']
            cur_y = dataclient['y']
            for j in range(len(cur_x)):
                # load the image as numpy array based on the path
                img = np.asarray(Image.open(
                    root + '/raw/img_align_celeba/' + cur_x[j]))
                test_data_x.append(img)
                test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        # print(f'img shape: {img.shape}, target: {target}')
        # img = np.array([img])
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.method == 'FATS':
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


class CIFAR10_Client(Dataset):
    """
    This dataset is derived from the Leaf repository
    (
    """

    def __init__(self, images, labels, method='base', transform=None, target_transform=None):
        super().__init__()
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.method = method

        self.data = images
        self.label = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        # raise error if the image is not 3 channels
        if len(img.shape) == 2:
            img = np.array(img)
        else:
            img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.method == 'FATS':
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class SHAKESPEARE_Client(Dataset):
    # ./preprocess.sh -s iid --iu 0.02 --sf 0.2 -k 0 -t sample -tf 0.8
    def __init__(self, dataclient, train=True, method='base'):
        super().__init__()
        self.train = train
        self.method = method

        if self.train:
            train_data_x = []
            train_data_y = []
            self.dic_users = set()
            l = len(train_data_x)
            cur_x = dataclient['x']
            cur_y = dataclient['y']
            for j in range(len(cur_x)):
                self.dic_users.add(j + l)
                train_data_x.append(cur_x[j])
                train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            cur_x = dataclient['x']
            cur_y = dataclient['y']
            for j in range(len(cur_x)):
                test_data_x.append(cur_x[j])
                test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.from_numpy(np.array(indices)).type(torch.LongTensor)

        # y = torch.Tensor(np.array(y))
        target = torch.FloatTensor(np.array(target))
        if self.method == 'FATS':
            return indices, target, index
        else:
            return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


class DatasetwithIndex(torch.utils.data.Dataset):
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        data, target = self.source[index]

        # Your transformations here (or set it in source)

        return data, target, index

    def __len__(self):
        return len(self.source)


def load_vision_data(dataname: str = "mnist", max_client=-1, method: str = 'base'):
    if dataname == "cifar10":
        """Load CIFAR-10 (training and test set)."""
        transform = transforms.Compose(
            [
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ]
        )

        trainset = CIFAR10("~/Documents/datasets", train=True,
                           download=True, transform=transform)
        testset = CIFAR10("~/Documents/datasets", train=False,
                          download=True, transform=transform)
        if method == 'FATS':
            trainset = DatasetwithIndex(trainset)
            testset = DatasetwithIndex(testset)
            classes = set(trainset.source.targets)
            datashape = trainset.source.data[0].shape
        else:
            classes = set(trainset.targets)
            datashape = trainset.data[0].shape

    elif dataname == "cifar100":
        """Load CIFAR100 (training and test set)."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            # transforms.Lambda(torch.flatten)
        ])

        trainset = CIFAR100("~/Documents/datasets", train=True,
                            download=True, transform=transform)
        testset = CIFAR100("~/Documents/datasets", train=False,
                           download=True, transform=transform)
        if method == 'FATS':
            trainset = DatasetwithIndex(trainset)
            testset = DatasetwithIndex(testset)
            classes = set(trainset.source.targets)
            datashape = trainset.source.data[0].shape
        else:
            classes = set(trainset.targets)
            datashape = trainset.data[0].shape

    elif dataname == "mnist":
        """Load MNIST (training and test set)."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(torch.flatten)
        ])

        trainset = MNIST("~/Documents/datasets", train=True,
                         download=True, transform=transform)
        testset = MNIST("~/Documents/datasets", train=False,
                        download=True, transform=transform)
        if method == 'FATS':
            trainset = DatasetwithIndex(trainset)
            testset = DatasetwithIndex(testset)
            classes = trainset.source.targets.unique()
            datashape = trainset.source.data[0].shape
        else:
            classes = trainset.targets.unique()
            datashape = trainset.data[0].shape

    elif dataname == "fmnist":
        """Load Fashion MNIST (training and test set)."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Lambda(torch.flatten)
        ])

        trainset = FashionMNIST("~/Documents/datasets", train=True,
                                download=True, transform=transform)
        testset = FashionMNIST("~/Documents/datasets", train=False,
                               download=True, transform=transform)
        if method == 'FATS':
            trainset = DatasetwithIndex(trainset)
            testset = DatasetwithIndex(testset)
            classes = trainset.source.targets.unique()
            datashape = trainset.source.data[0].shape
        else:
            classes = trainset.targets.unique()
            datashape = trainset.data[0].shape

    elif dataname == "SVHN":
        """load SVHN (training and test set)."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            # transforms.Lambda(torch.flatten)
        ])

        trainset = SVHN("~/Documents/datasets", split='train',
                        download=True, transform=transform)
        testset = SVHN("~/Documents/datasets", split='test',
                       download=True, transform=transform)
        if method == 'FATS':
            trainset = DatasetwithIndex(trainset)
            testset = DatasetwithIndex(testset)
            classes = set(trainset.source.labels)
            datashape = trainset.source.data[0].shape
        else:
            classes = set(trainset.labels)
            datashape = trainset.data[0].shape

    num_examples = {"trainset": len(trainset), "testset": len(
        testset),  "classes": len(classes)}
    print(f'dataset: {num_examples}')
    print(f'train sample shape: {datashape}')
    print(f'trainset classes: {classes}')
    return trainset, testset, num_examples

# pylint: disable=too-many-locals


def vision_train_valid_test_partition(
    dataset: str,
    num_clients: int,
    validation_fraction: int = 0.1,
    sampling_type: str = "iid",
    random_seed: int = None,
    method: str = "base",
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).
    trainset, testset, num_clients, train_len, val_len, test_len, random_seed
    Parameters
    ----------
    dataset: str,
        name of the dataset
    num_clients: int,
        number of clients
    validation_fraction: int = 0.1,
        fraction of the training set to use for validation
    sampling_type: str = "iid",
        type of sampling
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    trainset, testset, num_examples = load_vision_data(dataset, method=method)
    train_len = int(num_examples["trainset"] / num_clients)
    val_len = 0 if validation_fraction == 0.0 else int(
        train_len * validation_fraction)
    test_len = int(num_examples["testset"] / num_clients)
    partitioned_train = []
    partitioned_validation = []
    partitioned_test = []
    if sampling_type == 'iid':
        for cid in range(num_clients):
            validation_subset = torch.utils.data.Subset(
                trainset, range(cid * train_len, cid * train_len + val_len)
            )

            train_subset = torch.utils.data.Subset(
                trainset, range(cid * train_len + val_len,
                                (cid + 1) * train_len)
            )

            test_subset = torch.utils.data.Subset(
                testset, range(cid * test_len, (cid + 1) * test_len)
            )

            partitioned_train.append(train_subset)
            partitioned_validation.append(validation_subset)
            partitioned_test.append(test_subset)
    else:
        NIID_SHARD_CONSTANT = 2
        shard_size = int(train_len / NIID_SHARD_CONSTANT)
        # if trainset.source.targets is tensor, then argsort and sort are the same, if list, then argsort this list
        if dataset == 'SVHN':
            img_idxs = trainset.source.labels.argsort().tolist()
        else:
            img_idxs = trainset.source.targets.argsort().tolist() if isinstance(trainset.source.targets,
                                                                                torch.Tensor) else torch.tensor(trainset.source.targets).argsort().tolist()
        tmp = []
        for i in range(num_clients * NIID_SHARD_CONSTANT):
            tmp.append(img_idxs[shard_size * i:shard_size * (i + 1)])

        img_idxs_list = torch.randperm(
            num_clients * NIID_SHARD_CONSTANT, generator=torch.Generator().manual_seed(random_seed)
        )
        for cid in range(num_clients):
            train_idx = tmp[img_idxs_list[NIID_SHARD_CONSTANT * cid]
                            ]+tmp[img_idxs_list[NIID_SHARD_CONSTANT * cid + 1]]
            train_parition = torch.utils.data.Subset(
                trainset, train_idx
            )
            validation_subset = torch.utils.data.Subset(
                train_parition, range(val_len)
            )

            train_subset = torch.utils.data.Subset(
                train_parition, range(val_len, train_len)
            )

            shard_size = int(test_len / 2)
            if dataset == 'SVHN':
                img_idxs = testset.source.labels.argsort().tolist()
            else:
                img_idxs = testset.source.targets.argsort().tolist() if isinstance(testset.source.targets,
                                                                                   torch.Tensor) else torch.tensor(testset.source.targets).argsort().tolist()
            tmp = []
            for i in range(num_clients * 2):
                tmp.append(img_idxs[shard_size * i:shard_size * (i + 1)])

            img_idxs_list = torch.randperm(
                num_clients * 2, generator=torch.Generator().manual_seed(random_seed)
            )
            test_idx = tmp[img_idxs_list[2 * cid]] + \
                tmp[img_idxs_list[2 * cid + 1]]
            test_subset = torch.utils.data.Subset(
                testset, test_idx
            )
            partitioned_train.append(train_subset)
            partitioned_validation.append(validation_subset)
            partitioned_test.append(test_subset)

    return partitioned_train, partitioned_validation, partitioned_test


def celeba_train_valid_test_partition(
    num_clients: int,
    validation_fraction: int = 0.1,
    sampling_type: str = "iid",
    random_seed: int = None,
    method: str = "base",
    root: str = "/home/happy/Documents/datasets/leaf/data/celeba/data"
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).
    trainset, testset, num_clients, train_len, val_len, test_len, random_seed
    Parameters
    ----------
    dataset: str,
        name of the dataset
    num_clients: int,
        number of clients
    validation_fraction: int = 0.1,
        fraction of the training set to use for validation
    sampling_type: str = "iid",
        type of sampling
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_clients, _, train_data_temp, test_data_temp = read_data(root+"/train",
                                                                  root+"/test")
    partitioned_train = []
    partitioned_validation = []
    partitioned_test = []
    for cid in range(num_clients):
        train_partition = CELEBA_Client(
            train_data_temp[train_clients[cid]], train=True, method=method, root=root)
        train_len = len(train_partition)
        val_len = int(train_len * validation_fraction)
        train_len = train_len - val_len
        validation_subset = torch.utils.data.Subset(
            train_partition, range(val_len)
        )
        train_subset = torch.utils.data.Subset(
            train_partition, range(val_len, train_len)
        )
        test_subset = CELEBA_Client(
            test_data_temp[train_clients[cid]], train=False, method=method, root=root)
        partitioned_train.append(train_subset)
        partitioned_validation.append(validation_subset)
        partitioned_test.append(test_subset)

    return partitioned_train, partitioned_validation, partitioned_test


def shakespeare_train_valid_test_partition(
    num_clients: int,
    validation_fraction: int = 0.1,
    sampling_type: str = "iid",
    random_seed: int = None,
    method: str = "base",
    root: str = "./data/shakespeare/data"
    # root: str = "/home/happy/Documents/datasets/leaf/data/shakespeare/data"
    # root: str = "/ibex/user/wangc0g/pytorch-gpu-data-science-project/data/shakespeare/data"
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).
    trainset, testset, num_clients, train_len, val_len, test_len, random_seed
    Parameters
    ----------
    dataset: str,
        name of the dataset
    num_clients: int,
        number of clients
    validation_fraction: int = 0.1,
        fraction of the training set to use for validation
    sampling_type: str = "iid",
        type of sampling
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_clients, _, train_data_temp, test_data_temp = read_data(root+"/train",
                                                                  root+"/test")
    partitioned_train = []
    partitioned_validation = []
    partitioned_test = []
    for cid in range(num_clients):
        train_partition = SHAKESPEARE_Client(
            train_data_temp[train_clients[cid]], train=True, method=method)
        train_len = len(train_partition)
        val_len = int(train_len * validation_fraction)
        train_len = train_len - val_len
        validation_subset = torch.utils.data.Subset(
            train_partition, range(val_len)
        )
        train_subset = torch.utils.data.Subset(
            train_partition, range(val_len, train_len)
        )
        test_subset = SHAKESPEARE_Client(
            test_data_temp[train_clients[cid]], train=False, method=method)
        partitioned_train.append(train_subset)
        partitioned_validation.append(validation_subset)
        partitioned_test.append(test_subset)

    return partitioned_train, partitioned_validation, partitioned_test


def cifar10_train_valid_test_partition(
    num_clients: int,
    validation_fraction: int = 0.1,
    sampling_type: str = "niid",
    random_seed: int = None,
    method: str = "base",
    dataname: str = "cifar10"
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).
    trainset, testset, num_clients, train_len, val_len, test_len, random_seed
    Parameters
    ----------
    dataset: str,
        name of the dataset
    num_clients: int,
        number of clients
    validation_fraction: int = 0.1,
        fraction of the training set to use for validation
    sampling_type: str = "iid",
        type of sampling
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    alpha = 0.5
    if dataname == "cifar10":
        train_path, testset = get_cifar_10()
    elif dataname == "SVHN":
        train_path, testset = get_svhn()
        alpha = 10
    elif dataname == "fmnist":
        train_path, testset = get_fashion_mnist()
    elif dataname == "mnist":
        train_path, testset = get_mnist()

    # testset with datasplit
    if method == "FATS":
        testset = DatasetwithIndex(testset)

    train_data, labels = torch.load(train_path)
    idx = np.array(range(len(labels)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=num_clients, concentration=alpha, accept_imbalanced=True
    )
    partitioned_train = []
    partitioned_validation = []
    partitioned_test = []
    test_len = len(testset) // num_clients
    for cid in range(num_clients):
        images_partition = train_data[partitions[cid][0]]
        labels_partition = partitions[cid][1]
        if dataname == "cifar10":
            # create a dataset with the partitioned data
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=cifar10Transformation())
        elif dataname == "SVHN":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=svhnTransformation())
        elif dataname == "fmnist":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=fmnistTransformation())
        elif dataname == "mnist":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=mnistTransformation())

        train_len = len(train_partition)
        val_len = int(train_len * validation_fraction)
        train_len = train_len - val_len
        validation_subset = torch.utils.data.Subset(
            train_partition, range(val_len)
        )
        train_subset = torch.utils.data.Subset(
            train_partition, range(val_len, train_len)
        )
        test_subset = torch.utils.data.Subset(
            testset, range(cid*test_len, (cid+1)*test_len))
        partitioned_train.append(train_subset)
        partitioned_validation.append(validation_subset)
        partitioned_test.append(test_subset)

    return partitioned_train, partitioned_validation, partitioned_test


def cifar10_train_valid_test_partition_selected(
    num_clients: int,
    validation_fraction: int = 0.1,
    sampling_type: str = "niid",
    random_seed: int = None,
    method: str = "base",
    dataname: str = "cifar10",
    selected_clients: List[int] = []
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).
    trainset, testset, num_clients, train_len, val_len, test_len, random_seed
    Parameters
    ----------
    dataset: str,
        name of the dataset
    num_clients: int,
        number of clients
    validation_fraction: int = 0.1,
        fraction of the training set to use for validation
    sampling_type: str = "iid",
        type of sampling
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    alpha = 0.5
    if dataname == "cifar10":
        train_path, testset = get_cifar_10()
    elif dataname == "SVHN":
        train_path, testset = get_svhn()
        alpha = 10
    elif dataname == "fmnist":
        train_path, testset = get_fashion_mnist()
    elif dataname == "mnist":
        train_path, testset = get_mnist()

    # testset with datasplit
    if method == "FATS":
        testset = DatasetwithIndex(testset)

    train_data, labels = torch.load(train_path)
    idx = np.array(range(len(labels)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=num_clients, concentration=alpha, accept_imbalanced=True
    )
    partitioned_train = []
    partitioned_validation = []
    partitioned_test = []
    test_len = len(testset) // num_clients
    for cid in selected_clients:
        images_partition = train_data[partitions[cid][0]]
        labels_partition = partitions[cid][1]
        if dataname == "cifar10":
            # create a dataset with the partitioned data
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=cifar10Transformation())
        elif dataname == "SVHN":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=svhnTransformation())
        elif dataname == "fmnist":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=fmnistTransformation())
        elif dataname == "mnist":
            train_partition = CIFAR10_Client(
                images_partition, labels_partition, method=method, transform=mnistTransformation())

        train_len = len(train_partition)
        val_len = int(train_len * validation_fraction)
        train_len = train_len - val_len
        validation_subset = torch.utils.data.Subset(
            train_partition, range(val_len)
        )
        train_subset = torch.utils.data.Subset(
            train_partition, range(val_len, train_len)
        )
        test_subset = torch.utils.data.Subset(
            testset, range(cid*test_len, (cid+1)*test_len))
        partitioned_train.append(train_subset)
        partitioned_validation.append(validation_subset)
        partitioned_test.append(test_subset)

    return partitioned_train, partitioned_validation, partitioned_test


def partition_dataset(
    dataset: Dataset, division_list: List[List[int]]
) -> List[Dataset]:
    """
    Partition dataset for niid settings - by writer id (each partition has only single writer data).
    Parameters
    ----------
    dataset: Dataset
        dataset of all images
    division_list: List[List[int]]
        list of lists of indices to identify unique writers

    Returns
    -------
    subsets: List[Dataset]
        subsets of datasets divided by writer id
    """
    subsets = []
    for sequence in division_list:
        subsets.append(Subset(dataset, sequence))
    return subsets


# pylint: disable=too-many-locals
def train_valid_test_partition(
    partitioned_dataset: List[Dataset],
    train_split: float = 0.9,
    validation_split: float = 0.0,
    test_split: float = 0.1,
    random_seed: int = None,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).

    Parameters
    ----------
    partitioned_dataset: List[Dataset]
        partitioned datasets
    train_split: float
        part of the data used for training
    validation_split: float
        part of the data used for validation
    test_split: float
        part of the data used for testing
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_subsets = []
    validation_subsets = []
    test_subsets = []

    for subset in partitioned_dataset:
        subset_len = len(subset)
        train_len = int(train_split * subset_len)
        # Do this checkup for full dataset use
        # Consider the case sample size == 5 and
        # train_split = 0.5 test_split = 0.5
        # if such check as below is not performed
        # one sample will be missing
        if validation_split == 0.0:
            test_len = subset_len - train_len
            val_len = 0
        else:
            test_len = int(test_split * subset_len)
            val_len = subset_len - train_len - test_len
        train_dataset, validation_dataset, test_dataset = random_split(
            subset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_subsets.append(train_dataset)
        validation_subsets.append(validation_dataset)
        test_subsets.append(test_dataset)
    return train_subsets, validation_subsets, test_subsets


def transform_datasets_into_dataloaders(
    datasets: List[Dataset], batch_size, **dataloader_kwargs
) -> List[DataLoader]:
    """
    Transform datasets into dataloaders.
    Parameters
    ----------
    datasets: List[Dataset]
        list of datasets
    dataloader_kwargs
        arguments to DataLoader

    Returns
    -------
    dataloader: List[DataLoader]
        list of dataloaders
    """
    dataloaders = []
    for dataset in datasets:
        PreSampler = BatchSampler(RandomSampler(
            range(len(dataset)), replacement=False), batch_size=batch_size, drop_last=False)
        dataloaders.append(DataLoader(
            dataset, batch_sampler=PreSampler, **dataloader_kwargs))
    return dataloaders


# pylint: disable=too-many-arguments
def create_federated_dataloaders(
    dataset: str = "FeMNIST",
    sampling_type: str = 'niid',
    dataset_fraction: float = 0.05,
    batch_size: int = 10,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    method: str = "base",
    min_samples_per_client: int = 0,
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """Create the federated dataloaders by following all the preprocessing
    steps and division.

    Parameters
    ----------
    sampling_type: str
        "niid" or "iid"
    dataset_fraction: float
        fraction of the total data that will be used for sampling
    batch_size: int
        batch size
    train_fraction, validation_fraction, test_fraction: float
        fraction of each local dataset used for training, validation, testing
    random_seed: int
        random seed for data shuffling

    Returns
    -------
    """
    if dataset == "FeMNIST":
        if train_fraction + validation_fraction + test_fraction != 1.0:
            raise ValueError(
                "The fraction of train, validation and test should add up to 1.0."
            )
        # Download and unzip the data
        log(INFO, "NIST data downloading started")
        nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
        nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
        nist_by_class_downloader = ZipDownloader(
            "by_class", "data/raw_FeMNIST", nist_by_class_url, "data/raw_FeMNIST_byclass.zip")
        nist_by_writer_downloader = ZipDownloader(
            "by_write", "data/raw_FeMNIST", nist_by_writer_url, "data/raw_FeMNIST_bywrite.zip"
        )
        nist_by_class_downloader.download()
        nist_by_writer_downloader.download()
        log(INFO, "NIST data downloading done")

        # Preprocess the data
        log(INFO, "Preprocessing of the NIST data started")
        nist_data_path = pathlib.Path("data")
        nist_preprocessor = NISTPreprocessor(nist_data_path)
        nist_preprocessor.preprocess()
        log(INFO, "Preprocessing of the NIST data done")

        # Create information for sampling
        log(INFO, "Creation of the sampling information started")
        df_info_path = pathlib.Path(
            "data/processed_FeMNIST/processed_images_to_labels.csv")
        df_info = pd.read_csv(df_info_path, index_col=0)
        sampler = NistSampler(df_info)
        sampled_data_info = sampler.sample(
            sampling_type, dataset_fraction, min_samples_per_client=min_samples_per_client, random_seed=random_seed
        )
        sampled_data_info_path = pathlib.Path(
            f"data/processed_FeMNIST/{sampling_type}_sampled_images_to_labels.csv"
        )
        sampled_data_info.to_csv(sampled_data_info_path)
        log(INFO, "Creation of the sampling information done")

        # Create a list of DataLoaders
        log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets started")
        sampled_data_info = pd.read_csv(sampled_data_info_path)
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(sampled_data_info["character"])
        full_dataset = create_dataset(sampled_data_info, labels, method=method)
        division_list = create_partition_list(sampled_data_info)
        partitioned_dataset = partition_dataset(full_dataset, division_list)
        partitioned_train, partitioned_validation, partitioned_test = train_valid_test_partition(
            partitioned_dataset,
            random_seed=random_seed,
            train_split=train_fraction,
            validation_split=validation_fraction,
            test_split=test_fraction,)

    elif dataset == "shakespeare":
        # create the full dataset and partition it into a list with pre-defined clients number
        num_clients = int(dataset_fraction)
        partitioned_train, partitioned_validation, partitioned_test = shakespeare_train_valid_test_partition(
            num_clients, validation_fraction, sampling_type, random_seed, method
        )

    elif dataset in ["cifar10", "SVHN", "fmnist", "mnist"]:
        # create the full dataset and partition it into a list with pre-defined clients number
        num_clients = int(dataset_fraction)
        partitioned_train, partitioned_validation, partitioned_test = cifar10_train_valid_test_partition(
            num_clients, validation_fraction, sampling_type, random_seed, method, dataset
        )

    elif dataset == "celeba":
        # create the full dataset and partition it into a list with pre-defined clients number
        num_clients = int(dataset_fraction)
        partitioned_train, partitioned_validation, partitioned_test = celeba_train_valid_test_partition(
            num_clients, validation_fraction, sampling_type, random_seed, method
        )

    trainloaders = transform_datasets_into_dataloaders(
        partitioned_train, batch_size=batch_size
    )
    valloaders = transform_datasets_into_dataloaders(
        partitioned_validation, batch_size=batch_size
    )
    testloaders = transform_datasets_into_dataloaders(
        partitioned_test, batch_size=batch_size
    )
    # combine the subset in partitioned_test to create a central test set
    central_test = torch.utils.data.ConcatDataset(partitioned_test)
    central_testloaders = DataLoader(central_test, batch_size=batch_size)
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets done")
    return trainloaders, valloaders, testloaders, central_testloaders
