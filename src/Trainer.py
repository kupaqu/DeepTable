import torch

from torch import nn
from typing import List
from sklearn.base import ClassifierMixin

from OpenMLDataset import OpenMLDataset
from GAN import GAN
from batch_utils import get_batch_lambda, get_batch_metafeatures

class Trainer:
    def __init__(self, clfs: List[ClassifierMixin], data_dir: str, batch_size: int = 16, lr: int = 0.0003):
        self.clfs = clfs
        self.batch_size = batch_size

        train_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=False)
        test_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=True)
        self._train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self._test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        n_clfs = len(clfs)
        n_metas = self._get_n_metas_from_dataset(train_dataset)
        self.gan = GAN(len(clfs), n_clfs, n_metas)

        self.d_opt = torch.optim.Adam(self.gan.d.parameters(), lr=lr)
        self.g_opt = torch.optim.Adam(self.gan.g.parameters(), lr=lr)

        self._device = 'cpu'
    
    def _get_n_metas_from_dataset(self, dataset: torch.utils.data.Dataset):
        _, _, _, meta = dataset[0]
        n_metas = meta.shape[0]

        return n_metas
    
    def to(self, device: str):
        self._device = device
        self.gan.to(device)

    def _train_discriminator(self, X: torch.Tensor, y: torch.Tensor, \
                             lambda_: torch.Tensor, meta: torch.Tensor):
        self.d_opt.zero_grad()

        # train on real
        pred_lambda, pred_label = self.gan.d_forward(X, meta)
        true_label = torch.ones(self.batch_size, 1, device=self._device)

        real_lambda_loss = nn.functional.l1_loss(pred_lambda, lambda_)
        real_label_loss = nn.functional.binary_cross_entropy(pred_label, true_label)
        real_loss = real_lambda_loss + real_label_loss

        # train on generated
        fake_X = self.gan.g_forward(meta)
        fake_lambda = get_batch_lambda(clfs=self.clfs, X=fake_X).to(self._device)
        
        pred_lambda, pred_label = self.gan.d_forward(fake_X)
        fake_label = torch.zeros(self.batch_size, 1, device=self._device)

        fake_lambda_loss = nn.functional.l1_loss(pred_lambda, fake_lambda)
        fake_label_loss = nn.functional.binary_cross_entropy(pred_label, fake_label)
        fake_loss = fake_lambda_loss + fake_label_loss

        # update weights
        loss = real_loss + fake_loss
        loss.backward()
        self.d_opt.step()

        return loss.item()

    def _train_generator(self, X: torch.Tensor, y: torch.Tensor, \
                         lambda_: torch.Tensor, meta: torch.Tensor):
        self.g_opt.zero_grad()

        fake_X = self.gan.g_forward(meta)
        pred_lambda_, pred_label = self.gan.d_forward(fake_X)
        target_label = torch.ones(self.batch_size, 1, device=self._device)
        
        # TODO: experiment with lambda loss in generator

        loss = nn.functional.binary_cross_entropy(pred_label, target_label)
        loss.backward()
        self.g_opt.step()

        return loss.item()

    def _train_gan(self, X: torch.Tensor, y: torch.Tensor, \
                         lambda_: torch.Tensor, meta: torch.Tensor):
        X = X.to(self._device)
        y = y.to(self._device)
        lambda_ = lambda_.to(self._device)
        meta = meta.to(self._device)

        d_loss = self._train_discriminator(X, y, lambda_, meta)
        g_loss = self._train_generator(X, y, lambda_, meta)

        return d_loss, g_loss
    
    def _test_discriminator(self):
        ...
    
    def _test_generator(self):
        ...

    def _test_gan(self):
        ...