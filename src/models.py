import math
import torch
from torch import nn


def init_orthogonal(module: nn.Module):
    """
    Initializes model weights orthogonally.
    This has been shown to greatly improve training efficiency.
    """
    with torch.no_grad():
        for param in module.parameters():
            if len(param.size()) >= 2:
                torch.nn.init.orthogonal_(param.data)


class QNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        def conv_act(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                # nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.backbone = nn.Sequential(
            conv_act(obs_channels, 16, 5),
            conv_act(16, 32, 3),
            nn.Conv2d(32, 64, 3, padding="same"),
        )
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.advantage_head = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, action_count),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_batch).amax(dim=(-2, -1))  # max pooling
        fc_features = self.fc(backbone_features)
        value = self.value_head(fc_features)
        advantages = self.advantage_head(fc_features)
        return value - advantages.mean(dim=1, keepdim=True) + advantages


class QNetV2(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        def conv_block_pool(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.MaxPool2d(2, ceil_mode=True),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_channels, 16, 5, padding="same"),
            nn.SiLU(),
            *conv_block_pool(16, 32),
            *conv_block_pool(32, 64),
            *conv_block_pool(64, 128),
            nn.Conv2d(128, 128, 3, groups=128 // 16, padding="same"),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(256, action_count),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            for linear_layer in [self.value_head[-1], self.advantage_head[-1]]:
                linear_layer.weight.fill_(0)
                linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_batch)
        value = self.value_head(backbone_features)
        advantages = self.advantage_head(backbone_features)
        return value - advantages.mean(dim=1, keepdim=True) + advantages


class NetV2(nn.Module):
    def __init__(self, obs_shape: torch.Size, output_dim: int) -> None:
        super().__init__()

        def conv_block_pool(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.MaxPool2d(2, ceil_mode=True),
                nn.SiLU(),
            )

        obs_channels, _, _ = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(obs_channels, 16, 5, padding="same"),
            nn.SiLU(),
            *conv_block_pool(16, 32),
            *conv_block_pool(32, 64),
            *conv_block_pool(64, 128),
            nn.Conv2d(128, 128, 3, groups=128 // 16, padding="same"),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
        )

        init_orthogonal(self)

        with torch.no_grad():
            # Have the final layer output zeros at the start of training.
            linear_layer = self.network[-1]
            linear_layer.weight.fill_(0)
            linear_layer.bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.network(input_batch)


class DebugMLPQNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(math.prod(obs_shape), 128),
            nn.SiLU(),
            nn.Linear(128, action_count),
        )
        with torch.no_grad():
            # Have both heads output zeros at the start of training.
            self.mlp[-1].weight.fill_(0)
            self.mlp[-1].bias.fill_(0)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.mlp(input_batch.view(input_batch.shape[0], -1))
