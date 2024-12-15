import torch

from torch import nn

class Generator(nn.Module):
    def __init__(self, n_metas: int):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(32+n_metas, 64, kernel_size=(4, 15)), # не добавляем паддинг и страйд т. к. делит на классы
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 128, kernel_size=(4, 15)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=(4, 15)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=(4, 15)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=(4, 8))
        )

    def forward(self, z: torch.Tensor, meta: torch.Tensor, n_classes: torch.Tensor):
        device = z.get_device()

        to_concat = []
        for i in range(n_classes):
            pos = torch.ones_like(z, device=device) * torch.tensor(i, device=device) + z
            input_ = torch.cat((pos, meta), 1)

            output = self.upconv(input_)
            # print(output.shape)
            to_concat.append(output)

        concat = torch.cat(to_concat, 3)
        # print(concat.shape)
        return concat