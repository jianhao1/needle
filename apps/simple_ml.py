"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from tqdm import tqdm
import gc

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
# device = ndl.cpu()
device = ndl.cuda()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(label_filename, 'rb') as lbpath:
      magic, num_items = struct.unpack('>II', lbpath.read(8))
      assert magic == 2049
      y = np.frombuffer(lbpath.read(num_items), dtype=np.uint8)
      assert len(y) == num_items
    
    with gzip.open(image_filename, 'rb') as imgpath:
      magic, num_images, rows, cols = struct.unpack('>IIII', imgpath.read(16))
      assert magic == 2051
      images = np.frombuffer(imgpath.read(num_images * rows * cols), dtype=np.uint8)
      assert len(images) == num_images * rows * cols
      X = images.reshape(num_images, rows * cols).astype(np.float32) / 255.0
    
    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(len(y)), y])
    return ( ndl.log(ndl.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,)) ).sum() / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    m = X.shape[0]
    k = W2.shape[1]
    for start in range(0, m, batch):
      end = min(start + batch, m)
      size = end - start
      X_batch = ndl.Tensor(X[start:end])
      y_batch = np.zeros((size, k))
      y_batch[np.arange(size), y[start:end]] = 1
      y_batch = ndl.Tensor(y_batch)

      loss = softmax_loss(ndl.relu(X_batch @ W1) @ W2, y_batch)
      loss.backward()

      W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
      W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    tot_correct = 0
    tot_loss = 0

    if opt:
      model.train()
    else:
      model.eval()
    
    n = len(dataloader.dataset)
    with tqdm(total=n) as progress_bar:
      for X, y in dataloader:
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        logits = model(X)
        loss = loss_fn(logits, y)
        tot_loss += loss.numpy()[0] * y.shape[0]
        tot_correct += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
        if opt:
          loss.backward()
          opt.step()
        progress_bar.update(y.shape[0])
    
    return tot_correct / n, tot_loss / n


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
      avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
      print(f'epoch {i} | avg_acc: {avg_acc}, avg_loss: {avg_loss}')
    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    return epoch_general_cifar10(dataloader, model, loss_fn())


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    tot_correct = 0
    tot_loss = 0

    if opt:
      model.train()
    else:
      model.eval()
    
    nbatch, batch_size = data.shape
    hidden = None
    n = 0
    for i in tqdm(range(0, nbatch - 1, seq_len)):
      gc.collect() # 也许是因为tensor放在显存而不是内存，触发不了gc？会爆显存，手动gc就可以了。为什么之前没问题？
      X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
      y_pred, hidden = model(X, hidden)

      # detach
      if isinstance(hidden, tuple):
        h, c = hidden
        hidden = (h.detach(), c.detach())
      else:
        hidden = hidden.detach()
      
      loss = loss_fn(y_pred, y)
      if opt:
        loss.backward()
        opt.step()
      
      tot_loss += loss.numpy()[0] * y.shape[0]
      tot_correct += np.sum(y_pred.numpy().argmax(axis=1) == y.numpy())

      n += y.shape[0]

    return tot_correct / n, tot_loss / n


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
      avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), opt, clip, device, dtype)
      print(f'epoch {i} | avg_acc: {avg_acc}, avg_loss: {avg_loss}')
    return avg_acc, avg_loss    

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    return epoch_general_ptb(data, model, seq_len, loss_fn(), None, None, device, dtype)

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
