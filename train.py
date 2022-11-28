from sklearn.metrics import accuracy_score
import torch
import yaml
from accumulator import Accumulator
from net import LeNet
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


device = 'cuda:0'
lenet = LeNet().to(device)

loss = nn.NLLLoss()

from torch.optim import Adam

optimizer = Adam(lenet.parameters(), lr=0.001)

train_dataset = CIFAR10('./datasets/cifar10', train=True, transform=transforms.ToTensor(), download=False)
val_dataset = CIFAR10('./datasets/cifar10', train=False, transform=transforms.ToTensor(), download=False)

with open('params.yaml', 'r') as fp:
    config = yaml.safe_load(fp)

train_dataloader = DataLoader(
    train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=config['dataset']['batch_size'],
    shuffle=False
)

log_softmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)


device = 'cuda:0'
num_epochs = config['hyperparams']['num_epochs']

for epoch_num in range(num_epochs):
    accumulator = Accumulator()

    lenet.train()
    for index, batch in enumerate(train_dataloader):
        X, y = batch

        batch_size = X.shape[0]
        X = X.to(device)
        y = y.to(device)

        logits = lenet(X, verbose=False)

        optimizer.zero_grad()
        loss_value = loss(log_softmax(logits), y)
        loss_value.backward()

        accumulator.append(loss_value.item(), batch_size)

        optimizer.step()

    print(epoch_num, 'train', accumulator.get())

    lenet.eval()
    val_accumulator = Accumulator()
    accuracy_accumulator = Accumulator()
    for index, batch in enumerate(val_dataloader):
        X, y = batch
        batch_size = X.shape[0]
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = lenet(X, verbose=False)
            loss_value = loss(log_softmax(logits), y)
            val_accumulator.append(loss_value.item(), batch_size)
            probabilities = softmax(logits)
            class_labels = torch.argmax(probabilities, dim=1)
            accuracy = accuracy_score(
                y_true=y.cpu().numpy(),
                y_pred=class_labels.cpu().numpy()
            )
            accuracy_accumulator.append(accuracy, batch_size)

    print(epoch_num, 'val', val_accumulator.get())
    print(epoch_num, 'accuracy', accuracy_accumulator.get())
