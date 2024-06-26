import torch

from .Discriminator import Discriminator
from .Generator import Generator
from .batch_utils import get_batch_metafeatures

from typing import Tuple

class GAN:
    """Custom implementation of Conditional GAN based on DeepTable.
    """
    def __init__(self, n_clfs: int, n_metas: int):
        self._device = 'cpu'
        self.d = Discriminator(n_clfs, n_metas)
        self.g = Generator(n_metas)

    def to(self, device: str):
        """Performs GAN device conversion.

        Args:
            device (str): Device ('cpu' or 'cuda')
        """
        self._device = device
        self.d.to(device)
        self.g.to(device)

    def d_forward(self, x: torch.Tensor, meta: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for discriminator forward method with preprocessing.

        Args:
            x (torch.Tensor): Input table
            meta (torch.Tensor, optional): Metafeatures of the table x

        Returns:
            torch.Tensor, torch.Tensor: Lambda vector, label (real or fake)
        """
        if meta is None: # if meta is None, it means that table is generated
            meta = get_batch_metafeatures(x).to(self._device)

        return self.d(x, meta)
    
    def g_forward(self, meta: torch.Tensor) -> torch.Tensor:
        """Wrapper for generator forward method with preprocessing.

        Args:
            meta (torch.Tensor): Metafeatures of the table that we want to generate

        Returns:
            torch.Tensor: Generated table
        """
        batch_size = meta.shape[0]
        meta = meta.view(batch_size, -1, 1, 1)
        z = torch.randn(batch_size, 32, 1, 1, device=self._device)

        return self.g(z, meta).view(-1, 16, 128)