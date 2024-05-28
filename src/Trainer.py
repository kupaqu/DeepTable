import torch
import os

from torch.nn.functional import l1_loss, mse_loss, binary_cross_entropy
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from typing import List, Dict
from sklearn.base import ClassifierMixin

from OpenMLDataset import OpenMLDataset
from GAN import GAN
from batch_utils import get_batch_lambda, get_batch_metafeatures
from utils import join_dicts, sum_dicts

class Trainer:
    def __init__(self, clfs: List[ClassifierMixin], data_dir: str, batch_size: int = 16, lr: int = 0.0003, run_dir=None):
        self.clfs = clfs

        train_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=False)
        test_dataset = OpenMLDataset(clfs=clfs, data_dir=data_dir, test=True)

        train_size = int(0.67 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        self._train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self._test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        self._val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        self.n_clfs = len(clfs)
        n_metas = self._get_n_metas_from_dataset(train_dataset)
        self.gan = GAN(self.n_clfs, n_metas)

        self.d_opt = torch.optim.Adam(self.gan.d.parameters(), lr=lr)
        self.g_opt = torch.optim.Adam(self.gan.g.parameters(), lr=lr)

        self._device = 'cpu'

        if run_dir is None:
            self.run_dir = os.path.join(os.getcwd(), 'DeepTable-runs')
            os.makedirs(self.run_dir, exist_ok=True)
        else:
            self.run_dir = run_dir

        self.history = {
            'n_epoch': 0,
            'Train': [],
            'Val': []
        }
    
    def _get_n_metas_from_dataset(self, dataset: torch.utils.data.Dataset) -> int:
        _, _, _, meta = dataset[0]
        n_metas = meta.shape[0]

        return n_metas
    
    def to(self, device: str):
        self._device = device
        self.gan.to(device)

    def _train_discriminator(self, X: torch.Tensor, y: torch.Tensor, \
                             lambda_: torch.Tensor, meta: torch.Tensor) -> float:
        self.d_opt.zero_grad()

        # getting batch_size of the current batch
        batch_size = X.shape[0]

        # train on real
        pred_lambda, pred_label = self.gan.d_forward(X, meta)
        true_label = torch.ones(batch_size, 1, device=self._device)

        real_lambda_loss = l1_loss(pred_lambda, lambda_)
        real_label_loss = binary_cross_entropy(pred_label, true_label)
        real_loss = real_lambda_loss + real_label_loss

        # train on generated
        fake_X = self.gan.g_forward(meta)
        fake_lambda = get_batch_lambda(clfs=self.clfs, X=fake_X).to(self._device)
        
        pred_lambda, pred_label = self.gan.d_forward(fake_X)
        fake_label = torch.zeros(batch_size, 1, device=self._device)

        fake_lambda_loss = l1_loss(pred_lambda, fake_lambda)
        fake_label_loss = binary_cross_entropy(pred_label, fake_label)
        fake_loss = fake_lambda_loss + fake_label_loss

        # update weights
        loss = real_loss + fake_loss
        loss.backward()
        self.d_opt.step()

        return loss.item()

    def _train_generator(self, X: torch.Tensor, y: torch.Tensor, \
                         lambda_: torch.Tensor, meta: torch.Tensor) -> float:
        self.g_opt.zero_grad()

        # getting batch_size of the current batch
        batch_size = X.shape[0]

        fake_X = self.gan.g_forward(meta)
        pred_lambda_, pred_label = self.gan.d_forward(fake_X)
        target_label = torch.ones(batch_size, 1, device=self._device)
        
        # TODO: experiment with lambda loss in generator
        # fake_meta = get_batch_metafeatures(fake_X, y).to(self._device)
        # meta_loss = l1_loss(fake_meta, meta)

        loss = binary_cross_entropy(pred_label, target_label)
        loss.backward()
        self.g_opt.step()

        return loss.item()

    def _train_gan(self, X: torch.Tensor, y: torch.Tensor, \
                         lambda_: torch.Tensor, meta: torch.Tensor) -> Dict[str, float]:

        d_loss = self._train_discriminator(X, y, lambda_, meta)
        g_loss = self._train_generator(X, y, lambda_, meta)
        losses = {'Discriminator BCE loss': d_loss, 'Generator BCE loss': g_loss}

        return losses
    
    def _evaluate_discriminator(self, X: torch.Tensor, lambda_: torch.Tensor, meta: torch.Tensor):
        metrics = {}

        with torch.no_grad():
            pred_lambda, pred_label = self.gan.d_forward(X, meta)
            metrics['Lambda classifier MAE'] = l1_loss(pred_lambda, lambda_).item()
            metrics['Lambda classifier MSE'] = mse_loss(pred_lambda, lambda_).item()

            pred_ids, true_ids = pred_lambda.argmax(dim=1, keepdim=False), lambda_.argmax(dim=1, keepdim=False)
            metrics['Lambda classifier accuracy'] = multiclass_accuracy(pred_ids, true_ids).item()
            metrics['Lambda classifier F1 macro'] = multiclass_f1_score(pred_ids, true_ids, \
                                                                        num_classes=self.n_clfs, average='macro').item()
            
        return metrics

    def _evaluate_generator(self, meta: torch.Tensor) -> Dict[str, float]:
        metrics = {}

        with torch.no_grad():
            fake_X = self.gan.g_forward(meta)
            fake_meta = get_batch_metafeatures(fake_X).to(self._device)
            metrics['Generated metafeatures MAE'] = l1_loss(fake_meta, meta).item()
            metrics['Generated metafeatures MSE'] = mse_loss(fake_meta, meta).item()

        return metrics

    def _evaluate_gan(self, X: torch.Tensor, y: torch.Tensor, \
                      lambda_: torch.Tensor, meta: torch.Tensor) -> Dict[str, float]:
        
        d_metrics = self._evaluate_discriminator(X, lambda_, meta)
        g_metrics = self._evaluate_generator(meta)
        metrics = join_dicts([d_metrics, g_metrics])

        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        
        running_metrics = None

        for X, y, lambda_, meta in self._train_dataloader:
            X = X.to(self._device)
            y = y.to(self._device)
            lambda_ = lambda_.to(self._device)
            meta = meta.to(self._device)

            losses = self._train_gan(X, y, lambda_, meta)
            metrics = self._evaluate_gan(X, y, lambda_, meta)
            metrics = join_dicts([losses, metrics])

            if running_metrics:
                running_metrics = sum_dicts([running_metrics, metrics])
            else:
                running_metrics = metrics

        epoch_metrics = {k: v/len(self._train_dataloader) for k, v in running_metrics.items()}

        return epoch_metrics
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        running_metrics = None

        for X, y, lambda_, meta in dataloader:
            X = X.to(self._device)
            y = y.to(self._device)
            lambda_ = lambda_.to(self._device)
            meta = meta.to(self._device)

            metrics = self._evaluate_gan(X, y, lambda_, meta)

            if running_metrics:
                running_metrics = sum_dicts([running_metrics, metrics])
            else:
                running_metrics = metrics

        evaluated_metrics = {k: v/len(dataloader) for k, v in running_metrics.items()}

        return evaluated_metrics
    
    def train(self, epochs: int, test_epoch: int, track_metric: str, mode: str = 'min'):
        
        best_val_metric = None
        last_val_metric = None

        initial_epoch = self.history['n_epoch']

        for i in range(initial_epoch, epochs):
            print(f'Epoch {i+1}:')
            # training
            train_metrics = self.train_epoch()
            self.history['Train'].append(train_metrics)
            self._verbose('Train', train_metrics)

            # validating
            val_metrics = self.evaluate(self._val_dataloader)
            self.history['Val'].append(val_metrics)
            self._verbose('Val', val_metrics)

            if best_val_metric is None:
                best_val_metric = val_metrics[track_metric]
            last_val_metric = val_metrics[track_metric]

            self.history['n_epoch'] = i

            if mode == 'min' and last_val_metric < best_val_metric \
                or mode == 'max' and last_val_metric > best_val_metric:
                best_val_metric = last_val_metric
                self.save_checkpoint('best.pt')
            self.save_checkpoint('last.pt')

            # testing
            if test_epoch > 0 and i % test_epoch == 0:
                test_metrics = self.evaluate(self._test_dataloader)
                self._verbose('Test', test_metrics)
                self._plot_history()

    def _verbose(self, label: str, metrics: Dict[str, float]):
        # TODO: logging to file
        print(f'\t{label} metrics:')
        for k, v in metrics.item():
            print(f'\t\t{k}: {v}')
        print()

    def _plot_history(self):
        ...

    def save_checkpoint(self, name: str):
        torch.save({'model_state_dict': self.gan.state_dict(),
                    'optimizer_state_dict': {'d_opt': self.d_opt.state_dict(),
                                             'g_opt': self.g_opt.state_dict()},
                    'history': self.history},
                    os.path.join(self.run_dir, f'{name}.pth'))

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.gan.load_state_dict(checkpoint['model_state_dict'])
        self.d_opt.load_state_dict(checkpoint['optimizer_state_dict']['d_opt'])
        self.g_opt.load_state_dict(checkpoint['optimizer_state_dict']['g_opt'])
        self.history = checkpoint['history']
    
    def get_model(self):
        return self.gan