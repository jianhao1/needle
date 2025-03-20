import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main)
    return nn.Sequential(res, nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    # block_list = [ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)]
    # return nn.Sequential(nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU(), *block_list, nn.Linear(hidden_dim, num_classes))
    # 上面的方式会有问题，搞不懂？？？
    return nn.Sequential(nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU(), *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)], nn.Linear(hidden_dim, num_classes))


def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    tot_error = 0
    tot_loss = 0

    if opt:
      model.train()
    else:
      model.eval()
    
    batches = 0
    for X, y in dataloader:
      logits = model(X)
      loss = nn.SoftmaxLoss()(logits, y)
      tot_loss += loss.numpy()
      tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
      if opt:
        loss.backward()
        opt.step()
      batches += 1
    
    n = len(dataloader.dataset)
    return tot_error/n, tot_loss/batches


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    model = MLPResNet(28*28, hidden_dim, num_classes=10)
    train_set = ndl.data.MNISTDataset(f'{data_dir}/train-images-idx3-ubyte.gz', f'{data_dir}/train-labels-idx1-ubyte.gz')
    test_set = ndl.data.MNISTDataset(f'{data_dir}/t10k-images-idx3-ubyte.gz', f'{data_dir}/t10k-labels-idx1-ubyte.gz')
    train_loader = ndl.data.DataLoader(train_set, batch_size, True)
    test_loader = ndl.data.DataLoader(test_set, batch_size)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
      train_error, train_loss = epoch(train_loader, model, opt)
    test_error, test_loss = epoch(test_loader, model)
    return train_error, train_loss, test_error, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
