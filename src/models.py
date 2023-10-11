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

        def conv_act(
            in_channels: int, out_channels: int, kernel_size: int
        ) -> nn.Sequential:
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
