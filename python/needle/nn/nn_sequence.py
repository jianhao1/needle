"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x)) ** (-1)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        bound = 1 / hidden_size ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias:
          self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
          self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias = bias
        if nonlinearity == 'tanh':
          self.nonlinearity = ops.tanh
        elif nonlinearity == 'relu':
          self.nonlinearity = ops.relu
        else:
          raise ValueError('nonlinearity type error.')
        self.hidden_size = hidden_size

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs, _ = X.shape
        if h is None:
          h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
          out += self.bias_ih.reshape((1, self.bias_ih.shape[0])).broadcast_to((X.shape[0], self.bias_ih.shape[0]))
          out += self.bias_hh.reshape((1, self.bias_hh.shape[0])).broadcast_to((X.shape[0], self.bias_hh.shape[0]))
        return self.nonlinearity(out)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        assert num_layers >= 1
        self.rnn_cells = []
        self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device=device, dtype=dtype))
        for i in range(1, num_layers):
          self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device=device, dtype=dtype))

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        in_list = list(ops.split(X, 0))
        if h0:
          h0_list = list(ops.split(h0, 0))
        else:
          h0_list = [None] * len(self.rnn_cells)
        h_n = []
        for l in range(len(self.rnn_cells)):
          prev_h = h0_list[l]
          for t in range(len(in_list)):
            prev_h = self.rnn_cells[l](in_list[t], prev_h)
            in_list[t] = prev_h
          h_n.append(prev_h)
        return ops.stack(in_list, 0), ops.stack(h_n, 0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        bound = 1 / hidden_size ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias:
          self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
          self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias = bias


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        batch, hidden = X.shape[0], self.W_ih.shape[1]//4
        if h:
          h0, c0 = h
        else:
          h0, c0 = init.zeros(batch, hidden, device=X.device, dtype=X.dtype), \
                init.zeros(batch, hidden, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
          out += self.bias_ih.reshape((1, hidden*4)).broadcast_to((batch, hidden*4))
          out += self.bias_hh.reshape((1, hidden*4)).broadcast_to((batch, hidden*4))
        i, f, g, o = ops.split(out.reshape((batch, 4, hidden)), 1)
        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = ops.tanh(g)
        o = Sigmoid()(o)
        c_ = f * c0 + i * g
        h_ = o * ops.tanh(c_)
        return h_, c_


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        assert num_layers >= 1
        self.lstm_cells = []
        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device=device, dtype=dtype))
        for i in range(1, num_layers):
          self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device=device, dtype=dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        num_layers = len(self.lstm_cells)
        in_list = list(ops.split(X, 0))
        if h:
          h_list = list(zip(list(ops.split(h[0], 0)), list(ops.split(h[1], 0))))
        else:
          h_list = [None] * num_layers
        h_n = []
        c_n = []
        for l in range(num_layers):
          prev_h = h_list[l]
          for t in range(len(in_list)):
            prev_h = self.lstm_cells[l](in_list[t], prev_h)
            in_list[t] = prev_h[0]
          h_n.append(prev_h[0])
          c_n.append(prev_h[1])
        return ops.stack(in_list, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        out = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        out = out.reshape((seq_len * bs, self.num_embeddings)) @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))