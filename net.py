import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.dense1 = nn.Linear(400, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

    def forward(self, x, verbose=True):
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        pool1_out = self.pool1(relu1_out)
        if verbose:
            print(pool1_out.shape)

        conv2_out = self.conv2(pool1_out)
        relu2_out = self.relu2(conv2_out)
        pool2_out = self.pool2(relu2_out)
        if verbose:
            print(pool2_out.shape)
        flatten = pool2_out.view((-1, 400))

        dense1_out = self.dense1(flatten)
        relu3_out = self.relu3(dense1_out)

        dense2_out = self.dense2(relu3_out)
        relu4_out = self.relu4(dense2_out)

        logits = self.dense3(relu4_out)

        return logits
        
