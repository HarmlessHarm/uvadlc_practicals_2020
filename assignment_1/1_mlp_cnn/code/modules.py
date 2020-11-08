"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
from gradient_check import eval_numerical_gradient_array


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
        Also, initialize gradients with zeros.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample    
        """

        self.params = dict()
        self.grads = dict()
        self.params['weight'] = np.random.normal(0, 0.0001, size=(in_features, out_features))
        self.params['bias'] = np.zeros(out_features).reshape(1,-1)

        # print('\nIN INIT\n')
        # print(self.params['bias'].shape)
        # print('\nEND INIT\n')
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x
        out = self.x @ self.params['weight'] + self.params['bias']
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        dx = dout @ self.params['weight'].T
        
        self.grads['weight'] = self.x.T @ dout
        batch_size = dout.shape[0]
        self.grads['bias'] = np.ones((1, batch_size)) @ dout

        # print('\nIN BACKWARD\n')
        # print(self.params['bias'].shape)
        # print(self.grads['bias'].shape)
        # print('\nEND BACKWARD\n')

        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def softmax(self, x):
        """
        Function to calculate softmax
        Args:
          x: input of the softmax
        Returns:
          out: output of the softmax
        """
        # Subtracting max of X from X to reduce the size of the exponents.
        x_max = np.max(x, axis=1, keepdims=True)
        out = np.exp(x-x_max) / (np.sum(np.exp(x- x_max), axis=1, keepdims=True))
        return out

    def softmax_grad(self, sigmas):
        """
        Calculates the gradient of the softmax for one input sample
        Args:
          sigmax: row vector with softmax values for one input (shape: 1 x N)
        Returns:
          grad: gradient of softmax (shape N x N)
        """
        grad = np.diag(sigmas) - sigmas.reshape(-1, 1) * sigmas
        return grad

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        self.x = x
        out = self.softmax(x)
        
        return out
    

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """

        m, n = self.x.shape
        softmax_mat = self.softmax(self.x)
        
        # dx = np.zeros((m, n))
        # for i, sample in enumerate(softmax_mat):
        #     # soft = self.softmax(sample)
        #     soft_grad = self.softmax_grad(sample)
        #     dx[i] = dout[i] @ soft_grad
        

        # Example with einsum notation
        # Reference: https://themaverickmeerkat.com/2019-10-23-Softmax/
        
        t1 = np.einsum('ij,ik->ijk', softmax_mat, softmax_mat)
        t2 = np.einsum('ij,jk->ijk', softmax_mat, np.eye(n, n))
        dSm = t2 - t1
        dx = np.einsum('ijk,ik->ij', dSm, dout)

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """

        out = np.mean(-np.sum(y * np.log(x), axis=1))
        # out = -np.log(x[np.arange(x.shape[0]), y.argmax(1)]).mean()
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """
        # dx = - y / x
        dx = -(y / x) / y.shape[0]
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x
        # print('IN ELU')
        # print(x.shape)
        # print("max",np.max(x))
        # print("min",np.min(x))
        out = np.where( x < 0, np.exp(x) - 1, x)
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout * np.where( self.x < 0, np.exp(self.x), 1)

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x
        # print('IN ELU')
        # print(x.shape)
        out = np.where( x < 0, 0, x)
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout * np.where( self.x < 0, 0, 1)

        return dx