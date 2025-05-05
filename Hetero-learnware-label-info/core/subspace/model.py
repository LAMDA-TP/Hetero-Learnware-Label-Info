""" Ref: https://github.com/clabrugere/pytorch-scarf/blob/master/scarf/model.py
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(
    nn.Module
): 
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        if input_dim != output_dim:
            self.downsample = nn.Sequential(
                nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class TableResNet(nn.Module):
    """
    A module for creating a TableResNet neural network that can be configured for
    either classification or regression tasks.

    This implementation of TableResNet allows users to specify task types,
    determining how the network handles the output layer based on the chosen task.
    The network consists of two main residual blocks followed by a task-specific
    output processing layer.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Dimension of the hidden layer outputs.
    output_dim : int
        Dimension of the output from the second residual block. For classification,
        this should be set to the number of classes. For regression, this typically
        matches the dimension of the transformed feature space before the final linear
        transformation to the target variable size.
    dropout_rate : float, optional
        Dropout rate for regularization. Default is 0.5.
    task_type : str, optional
        Type of the machine learning task: 'classification' or 'regression'.
        Default is 'classification'.

    Attributes
    ----------
    block1 : ResidualBlock
        First residual block, transforming from input_dim to hidden_dim.
    block2 : ResidualBlock
        Second residual block, transforming from hidden_dim to output_dim.
    softmax : nn.Softmax
        Softmax activation layer, used if the task type is 'classification'.
    linear : nn.Linear
        Linear transformation layer, used if the task type is 'regression' to map
        the output to a single target variable.

    Examples
    --------
    To create a model for a 3-class classification problem:
        model = TableResNet(input_dim=100, hidden_dim=50, output_dim=3, task_type='classification')

    To create a model for regression with an input feature dimension of 100:
        model = TableResNet(input_dim=100, hidden_dim=50, output_dim=10, task_type='regression')
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.5,
        task_type="classification",
    ):
        super(TableResNet, self).__init__()
        self.block1 = ResidualBlock(input_dim, hidden_dim, dropout_rate)
        self.block2 = ResidualBlock(hidden_dim, output_dim, dropout_rate)
        self.task_type = task_type
        if task_type == "classification":
            self.softmax = nn.Softmax(dim=1)
        elif task_type == "regression":
            self.linear = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.task_type == "classification":
            x = self.softmax(x)
        elif self.task_type == "regression":
            x = self.linear(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoding_dim,
        dropout_rate=0.5,
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock(input_dim, hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, encoding_dim, dropout_rate),
        )

        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(encoding_dim, hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, input_dim, dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    @torch.inference_mode()  # close the autograd engine
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_features(self, x_emb: torch.Tensor) -> torch.Tensor:
        return self.decoder(x_emb)