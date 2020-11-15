"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()

        module_list = []
        layers = [n_inputs] + n_hidden + [n_classes]
        
        # Iterate through each layers in- and output nodes
        print(list(zip(layers, layers[1:])))
        for i, (n_in, n_out) in enumerate(zip(layers, layers[1:])):
            lin = nn.Linear(n_in, n_out)
            module_list.append(lin)
            # use ELU only between input and hidden layers
            if i < len(layers) - 2:
                # module_list.append(ReLUModule())
                module_list.append(nn.ELU())
            else:
                # use SoftMax as last module
                module_list.append(nn.LogSoftmax())

        self.network = nn.Sequential(*module_list)
        self.network.apply(self.init_lin_weights)

    def init_lin_weights(self, model):
        if type(model) == nn.Linear:
            torch.nn.init.normal_(model.weight, 0, 0.0001)
            model.bias.data.fill_(0)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        # print(self.modules)
        out = self.network(x)
        
        return out
