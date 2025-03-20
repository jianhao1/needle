from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)

        with gzip.open(label_filename, 'rb') as lbpath:
          magic, num_items = struct.unpack('>II', lbpath.read(8))
          assert magic == 2049
          self.y = np.frombuffer(lbpath.read(num_items), dtype=np.uint8)
          assert len(self.y) == num_items

        with gzip.open(image_filename, 'rb') as imgpath:
          magic, num_images, rows, cols = struct.unpack('>IIII', imgpath.read(16))
          assert magic == 2051
          images = np.frombuffer(imgpath.read(num_images * rows * cols), dtype=np.uint8)
          assert len(images) == num_images * rows * cols
          self.X = images.reshape(num_images, rows, cols, 1).astype(np.float32) / 255.0

    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.X[index]), self.y[index]

    def __len__(self) -> int:
        return self.X.shape[0]