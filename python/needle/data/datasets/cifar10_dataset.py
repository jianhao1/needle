import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms)
        if train:
          files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
          files = ['test_batch']
        X = []
        Y = []
        for f in files:
          with open(os.path.join(base_folder, f), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            X.append(data_dict[b'data'])
            Y.append(data_dict[b'labels'])
        self.X = np.concatenate(X).reshape((-1, 3, 32, 32)) / 255.0
        self.Y = np.concatenate(Y)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        image = self.apply_transforms(self.X[index])
        label = self.Y[index]
        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.X.shape[0]
