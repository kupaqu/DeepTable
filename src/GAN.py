import torch

from Discriminator import Discriminator
from Generator import Generator
from batch_utils import get_batch_metafeatures

class GAN:
    def __init__(self, n_clfs: int, n_metas: int):
        self._device = 'cpu'
        self.d = Discriminator(n_clfs, n_metas)
        self.g = Generator(n_metas)

    def to(self, device: str):
        self._device = device
        self.d.to(device)
        self.g.to(device)

    def d_forward(self, x: torch.Tensor, meta: torch.Tensor = None):
        if meta is None: # if meta is None, it means that table is generated, so assume zeros comes before ones
            batch_size = x.shape[0]
            n_rows = x.shape[2]
            y = torch.cat([torch.zeros((batch_size, n_rows//2)), torch.ones((batch_size, n_rows//2))], dim=1)
            meta = get_batch_metafeatures(x, y).to(self._device)

        return self.d(x, meta)
    
    def g_forward(self, meta: torch.Tensor):
        batch_size = meta.shape[0]
        z = torch.randn(batch_size, 32, 1, 1, device=self._device)

        return self.g(z, meta).view_as(-1, 16, 128)