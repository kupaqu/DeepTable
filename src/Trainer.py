import torch

from typing import List
from sklearn.base import ClassifierMixin

from OpenMLDataset import OpenMLDataset
from gan import Discriminator, Generator

class Trainer:
    def __init__(self, clfs: List[ClassifierMixin], data_dir: str, batch_size: int = 16):
        train_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=False)
        test_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=True)
        self._train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self._test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    def _train_discriminator(self):
        ...

    def _train_generator(self):
        ...

    def _test_discriminator(self):
        ...
    
    def _test_generator(self):
        ...

    def _train_gan(self):
        ...
    