"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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
        
        self.modules = []
        layers = [n_inputs] + n_hidden + [n_classes]
        
        # Iterate through each layers in- and output nodes
        for n_in, n_out in zip(layers, layers[1:]):
            self.modules.append(LinearModule(n_in, n_out))
            self.modules.append(ELUModule())
            
        self.modules.append(SoftMaxModule())
    
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
        
        for module in self.modules:
            x = module.forward(x)
        
        return x
    
    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """
        
        for module in self.modules[::-1]:
            dout = module.backward(dout)
        
        return dout

    def update_params(self, learn_rate):
        """
        Performs an update step in all layers that have weights or biasses.

        Args:
          learn_rate: The learning rate used for the update step
        """
        # print('\n@@@@@@@@@@@@@@\n')
        # print("in update")
        for module in self.modules:
            # Apply gradient step only if module as params AND gradients
            if hasattr(module, 'params') and hasattr(module, 'grads'):
                # Update step for weights
                if 'weight' in module.params.keys() and 'weight' in module.grads.keys():
                    module.params['weight'] -= learn_rate * module.grads['weight']

                # Update step for biasses
                if 'bias' in module.params.keys() and 'bias' in module.grads.keys():
                    module.params['bias'] -= learn_rate * module.grads['bias']

        # print('\n@@@@@@@@@@@@@@\n')