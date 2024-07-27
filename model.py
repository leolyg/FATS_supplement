"""Implementation of the model used for the FEMNIST experiments."""


from logging import INFO
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch import Tensor
from torch.utils.data import DataLoader
import torch_optimizer as torch_optim
import torch.nn.functional as F


def return_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Return the model used for the FEMNIST experiments."""
    if dataset_name == "FeMNIST" or dataset_name == "mnist" or dataset_name == "fmnist":
        return NISTNet(num_classes=num_classes)
    elif dataset_name == 'SVHN' or dataset_name == 'celeba':
        return ResNet18(num_classes=num_classes, GRAYSCALE=False, dataname=dataset_name)
    elif dataset_name == 'shakespeare':
        return LSTMShakespeare(num_classes=num_classes)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        return VGG16(num_classes=num_classes)
        # return ResNet18(num_classes = num_classes, GRAYSCALE = False, dataname = dataset_name)


class LSTMShakespeare(nn.Module):
    def __init__(self, num_classes: int = 80, seq_len: int = 80, embedding_len: int = 8, n_hidden: int = 256):
        super(LSTMShakespeare, self).__init__()
        self.n_hidden = n_hidden

        self.embeds = nn.Embedding(seq_len, embedding_len)
        self.multi_lstm = nn.LSTM(
            input_size=embedding_len, hidden_size=n_hidden, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(n_hidden, num_classes)

    def forward(self, x, out_activation=False):
        x = x.to(torch.int64)
        x_ = self.embeds(x)
        h0 = torch.rand(2, x_.size(0), self.n_hidden).to(x.device)
        c0 = torch.rand(2, x_.size(0), self.n_hidden).to(x.device)
        activation, (h_n, c_n) = self.multi_lstm(x_, (h0, c0))

        fc_ = activation[:, -1, :]

        output = self.fc(fc_)
        if out_activation:
            return output, activation
        else:
            return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale, dataname):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dataname = dataname
        if self.dataname == 'celeba':
            self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
            self.fc = nn.Linear(2048 * block.expansion, num_classes)
        else:
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, _ = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        if self.dataname == 'celeba':
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits


def ResNet18(layers=[2, 2, 2, 2], num_classes=10, GRAYSCALE=True, dataname='mnist'):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=layers,
                   num_classes=num_classes,
                   grayscale=GRAYSCALE,
                   dataname=dataname)
    return model


class NISTNet(nn.Module):
    """Implementation of the model used in the LEAF paper for training on
    FEMNIST data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward step in training."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# build a vgg16 model for cifar10


class VGG16(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG16, self).__init__()
        self.features = self._make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
def train(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    method: str,
    epochs: Optional[int],
    learning_rate: float,
    device: torch.device,
    n_batches: Optional[int] = None,
    server_round: Optional[int] = None,
    unlearning_round: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Train a given model with CrossEntropy and SGD (or some version of it
    like batch-SGD).

    n_batches is an alternative way of specifying the training length
    (instead of epochs)
    """
    criterion = torch.nn.CrossEntropyLoss()
    if method == "infocom22" and server_round >= unlearning_round:
        optimizer = torch_optim.Adahessian(
            net.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    net.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    # pylint: disable=no-else-return
    if method == "FATS":
        if epochs or n_batches:
            epochs = epochs if epochs else n_batches
            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0
                for images, labels, indexs in trainloader:
                    correct, epoch_loss, total = train_step(
                        correct,
                        criterion,
                        device,
                        epoch_loss,
                        images,
                        labels,
                        net,
                        optimizer,
                        total,
                    )
                    break

                epoch_loss = epoch_loss / total
                epoch_acc = correct / total

                if verbose:
                    log(
                        INFO,
                        "Epoch %s: train loss %s, accuracy %s",
                        str(epoch + 1),
                        str(epoch_loss),
                        str(epoch_acc),
                    )
            # Train loss reported is typically the last epoch loss
            train_loss, train_acc = epoch_loss, epoch_acc
            if len(valloader):
                val_loss, val_acc = test(
                    net, valloader, method=method, device=device)
            else:
                val_loss, val_acc = None, None
            # print(f'train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}')
            return train_loss, train_acc, val_loss, val_acc
    else:
        if epochs:
            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0
                for images, labels in trainloader:
                    correct, epoch_loss, total = train_step(
                        correct,
                        criterion,
                        device,
                        epoch_loss,
                        images,
                        labels,
                        net,
                        optimizer,
                        total,
                    )
                epoch_loss = epoch_loss / total
                epoch_acc = correct / total

                if verbose:
                    log(
                        INFO,
                        "Epoch %s: train loss %s, accuracy %s",
                        str(epoch + 1),
                        str(epoch_loss),
                        str(epoch_acc),
                    )
            # Train loss reported is typically the last epoch loss
            train_loss, train_acc = epoch_loss, epoch_acc
            if len(valloader):
                val_loss, val_acc = test(net, valloader, device)
            else:
                val_loss, val_acc = None, None
            return train_loss, train_acc, val_loss, val_acc
        elif n_batches:
            # Training time given in number of batches not epochs
            correct, total, train_loss = 0, 0, 0.0
            for batch_idx, (images, labels) in enumerate(trainloader):
                if batch_idx == n_batches:
                    break
                correct, train_loss, total = train_step(
                    correct,
                    criterion,
                    device,
                    train_loss,
                    images,
                    labels,
                    net,
                    optimizer,
                    total,
                )
            train_acc = correct / total
            train_loss = train_loss / total
            if verbose:
                log(
                    INFO,
                    "Batch len based training: train loss %s, accuracy %s",
                    str(train_loss),
                    str(train_acc),
                )
            if len(valloader):
                val_loss, val_acc = test(net, valloader, device)
            else:
                val_loss, val_acc = None, None
            return train_loss, train_acc, val_loss, val_acc
        else:
            raise ValueError(
                "either n_epochs or n_batches should be specified ")


def train_step(
    correct: int,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch_loss: float,
    images: Tensor,
    labels: Tensor,
    net: nn.Module,
    optimizer: Union[torch.optim.SGD, torch_optim.Adahessian],
    total: int,
) -> Tuple[int, float, int]:
    """Single train step.

    Returns
    -------
    correct, epoch_loss, total: Tuple[int, float, int]
        number of correctly predicted samples, sum of loss, total number of
        samples
    """
    # device = torch.device("mps:0")
    net.to(device)
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    output = net(images)
    loss = criterion(output, labels)
    # if the optimizer is adahessian, we need create graph
    if isinstance(optimizer, torch_optim.Adahessian):
        loss.backward(create_graph=True)
        optimizer.step()
    else:
        loss.backward()
        optimizer.step()

    net.to("cpu")
    epoch_loss += loss.item()
    total += labels.size(0)
    _, predicted = torch.max(output.data, 1)
    if labels.shape > predicted.shape:
        _, labels = labels.max(dim=1)
    correct += predicted.eq(labels).sum().item()
    return correct, float(epoch_loss), total


def test(
    net: nn.Module, dataloader: DataLoader, method: str = 'base', device: torch.device = torch.device("cpu")
) -> Tuple[float, float]:
    """Test - calculate metrics on the given dataloader."""
    criterion = torch.nn.CrossEntropyLoss()
    if len(dataloader) == 0:
        raise ValueError("Dataloader can't be 0, exiting...")
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        if method == 'FATS':
            for images, labels, indexs in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss += criterion(output, labels).item()
                output = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                if labels.shape > predicted.shape:
                    _, labels = labels.max(dim=1)

                correct += predicted.eq(labels).sum().item()
        else:
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss += criterion(output, labels).item()
                output = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                if labels.shape > predicted.shape:
                    _, labels = labels.max(dim=1)

                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        loss /= total
    return float(loss), accuracy
