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

            nn.ConvTranspose2d(32, 1, kernel_size=(4, 7))
        )

    def forward(self, z: torch.Tensor, meta: torch.Tensor, n_classes: torch.Tensor):
        device = z.get_device()

        to_concat = []
        input_ = torch.cat((z, meta), 1)
        for i in range(n_classes):
            pos = torch.ones((input_.shape[0], 1, 1, 1), device=device) * torch.tensor(i, device=device)
            input_with_pos = torch.cat((input_, pos), 1)
            
            output = self.upconv(input_with_pos)
            # print(output.shape)
            to_concat.append(output)

        concat = torch.cat(to_concat, 3)
        print(concat.shape)
        return concat