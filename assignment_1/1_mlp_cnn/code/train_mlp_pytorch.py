"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle

from Statistics import Statistics
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OPTIMIZER = "sgd"


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
MODEL_DIR_DEFAULT = './models/'
MODEL_NAME_DEFAULT = 'mlp_pytorch'


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


def train(n_input, dnn_hidden_units, n_classes, cifar10):
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    
    
    mlp = MLP(n_input, dnn_hidden_units, n_classes).to(device)
    print(mlp)

    criterion = torch.nn.NLLLoss()
    if FLAGS.optim == 'adam':
        optimizer = torch.optim.Adam(mlp.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = torch.optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)

    stats = Statistics(FLAGS.eval_freq)

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels
    x_test, y_test = torch.tensor(x_test).to(device), torch.tensor(y_test).to(device)

    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    for e in range(FLAGS.max_steps):
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)

        x_flat = x.reshape(x.shape[0],-1)
        y_hat = mlp(x_flat)

        train_loss = criterion(y_hat, torch.argmax(y, axis=1))
        train_acc = accuracy(y_hat, y)

        stats.add_train_loss(train_loss.item())
        stats.add_train_accuracy(train_acc)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (e+1) % FLAGS.eval_freq == 0 or e == 0:
            print(f"Running Test at epoch: {e}")
            
            y_test_hat = mlp(x_test_flat)
            with torch.no_grad():
                test_loss = criterion(y_test_hat, torch.argmax(y_test, axis=1))
            test_acc = accuracy(y_test_hat, y_test)

            stats.add_test_loss(test_loss.item())
            stats.add_test_accuracy(test_acc)

            print_list = [
                f"Train loss : {round(train_loss.item(), 4)}",
                f"Train acc  : {round(train_acc, 4)}",
                f"Test  loss : {round(test_loss.item(), 4)}",
                f"Test  acc  : {round(test_acc, 4)}"
            ]
            print("\n".join(print_list))
            print()

    store_model(mlp, stats)
    stats.plot_statistics(title="PyTorch Multi Layer Perceptron", hline=0.48)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def make_model_name():
    return FLAGS.model_dir + FLAGS.model_name

def store_model(model, stats):
    # with open(FLAGS.model_dir, 'w') as f:
    torch.save(model.state_dict(), make_model_name() + '.model')
    
    with open(make_model_name() + '.stats', 'wb') as sf:
        pickle.dump(stats, sf)


    print(f"Model stored in {FLAGS.model_dir}")


def load_model(model_class, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(make_model_name() + '.model'))
    model.eval()

    with open(make_model_name() + '.stats', 'rb') as sf:
        stats = pickle.load(sf)
    
    return model, stats

def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    
    # Load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # Initialize model
    n_input = np.prod(cifar10['train'].images.shape[1:])
    n_classes = cifar10['train'].labels.shape[1]
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    if (FLAGS.load_model):
        print("LOAD MODEL")
        mlp, stats = load_model(MLP, n_input, dnn_hidden_units, n_classes)
        print("LOADED MODEL")
        print(mlp)
        stats.plot_statistics(title="PyTorch Multi Layer Perceptron", hline=0.48)
    
    else:
        print("TRAIN MODEL")
        # Run the training operation
        train(n_input, dnn_hidden_units, n_classes, cifar10)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--optim', type=str, default=OPTIMIZER,
                        help='Optimizer to use [(default)sgd, adam]'),
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR_DEFAULT,
                        help='Directory for storing model'),
    parser.add_argument('--model_name', type=str, default=MODEL_NAME_DEFAULT,
                        help='File name for storing model'),
    parser.add_argument('--load_model', type=str2bool, nargs='?', const=True ,default=False,
                        help='Wether or not to load a pre-saved model'),
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
