"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import tqdm

from statistics import statistics
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
MODEL_DIR_DEFAULT = './models/cnn_model.pkl'

FLAGS = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    pred_val = torch.argmax(predictions, axis=1)
    targ_val = torch.argmax(targets, axis=1)
    right = len(torch.where(targ_val == pred_val)[0])
    accuracy = right / len(pred_val)
    
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    # x_test, y_test = cifar10['test'].images, cifar10['test'].labels
    # x_test, y_test = torch.tensor(x_test).to(device), torch.tensor(y_test).to(device)

    test_batches = cifar10['test'].labels.shape[0] // FLAGS.batch_size
    # x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Initialize conv net
    n_channels = cifar10['train'].images.shape[1]
    n_classes = cifar10['train'].labels.shape[1]

    conv_net = ConvNet(n_channels, n_classes).to(device)
    print(conv_net)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(conv_net.parameters())
    
    stats = Statistics()

    for e in range(FLAGS.max_steps):
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)

        y_hat = conv_net(x)

        train_loss = criterion(y_hat, torch.argmax(y, axis=1))
        train_acc = accuracy(y_hat, y)

        stats.add_train_loss(train_loss)
        stats.add_train_accuracy(train_acc)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (e+1) % FLAGS.eval_freq == 0:
            y_test_hat = torch.zeros(cifar10['test'].labels.shape)
            print(f"Running Test at epcoh: {e}")
            # for i, (x_test_b, y_test_b) in enumerate(zip(x_test, y_test)):
            for test_i in tqdm.tqdm(range(test_batches)):

                x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
                x_test, y_test = torch.tensor(x_test).to(device), torch.tensor(y_test).to(device)

                with torch.no_grad():
                    out = conv_net(x_test)

                start = test_i * FLAGS.batch_size
                end = start + FLAGS.batch_size
                y_test_hat[start: end] = out

                # all_y_hat += y_test_hat

            test_loss = nn.CrossEntropyLoss(y_test_hat, torch.Tensor(cifar10['test'].labels))
            test_acc = accuracy(y_test_hat, torch.Tensor(cifar10['test'].labels))

            stats.add_test_loss(test_loss)
            stats.add_test_accuracy(test_acc)

            print(f"Train train_loss at {str(e).zfill(4)}: {round(train_loss.item(), 4)}, Test accuracy is: {round(test_acc, 4)} \n")
            # print(f"Train loss at {str(e).zfill(4)}: {round(loss.item(), 4)}")

    store_model(conv_net)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def store_model(model):
    torch.save(model.state_dict(), FLAGS.model_dir)


def load_model(model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(FLAGS.model_dir))
    model.eval()
    return model

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data'),
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR_DEFAULT,
                        help='Directory for storing model')

    FLAGS, unparsed = parser.parse_known_args()
    
    main()
