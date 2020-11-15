"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        super(ConvNet, self).__init__()

        act_fn = nn.ReLU()

        self.net_modules = (
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1), # Conv0
            SimplePreActBlock(nn.ReLU(), 64),   # PreAct1

            DownsampleBlock(64, 128),           # conv1, maxpool1
            SimplePreActBlock(nn.ReLU(), 128),  # PreAct2_a
            SimplePreActBlock(nn.ReLU(), 128),  # PreAct2_b

            DownsampleBlock(128, 256),          # conv2, maxpool2,
            SimplePreActBlock(nn.ReLU(), 256),  # PreAct3_a
            SimplePreActBlock(nn.ReLU(), 256),  # PreAct3_b

            DownsampleBlock(256, 512),          # conv3, maxpool3
            SimplePreActBlock(nn.ReLU(), 512),  # PreAct4_a
            SimplePreActBlock(nn.ReLU(), 512),  # PreAct4_b

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # maxpool4
            SimplePreActBlock(nn.ReLU(), 512),  # PreAct5_a PreAct5_b
            SimplePreActBlock(nn.ReLU(), 512),  # PreAct5_a PreAct5_b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # maxpool5

            nn.BatchNorm2d(512),
            nn.ReLU(),

            View([512]), # Reshape output of conv to batch_size x 512
            nn.Linear(in_features=512, out_features=n_classes), # linear
        )

        self.net = nn.Sequential(*self.net_modules)
    
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
        
        # out = self.net(x)
        # out = x
        # for mod in self.net_modules:
        #     print("\n-----\n")
        #     print("Module:", mod)
        #     print("in:",out.shape)
        #     if type(mod) == nn.Linear:
        #         print("weight:", mod.weight.shape)
        #     out = mod(out)

        #     print("out:", out.shape)
        
        out = self.net(x)

        return out

class DownsampleBlock(nn.Module):

    def __init__(self, c_in, c_out):
        super(DownsampleBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class SimplePreActBlock(nn.Module):

    def __init__(self, act_fn, c_in, double=False):
        super(SimplePreActBlock, self).__init__()

        
        # if not downsample:
        #     c_out = c_in

        # self.downsample = nn.Sequential(
        #     nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ) if downsample else None

        modules = (
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
        )

        if double:
            modules = (
                nn.BatchNorm2d(c_in),
                nn.ReLU(),
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c_in),
                nn.ReLU(),
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
            )

        self.net = nn.Sequential(*modules)

    def forward(self, x):

        # First downsample x,y dims and upscale channel dim
        # if self.downsample != None:
        #     x = self.downsample(x)

        # Then apply PreAct module
        z = self.net(x)
        
        

        return z + x


class View(nn.Module):
    """
    Reshapes the the input tensor

    Args:
        shape: tuple of shape of tensor not including batch size
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)