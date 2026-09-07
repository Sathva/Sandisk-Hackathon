"""
Model C: Multi-Resolution 1D CNN + Tabular MLP Architecture.

Fuses raw 2,000-element block sequences (Branch 1) with tabular features (Branch 2:
500 parametric + 19 spatial features = 519 features for Model C0).
"""

import torch
import torch.nn as nn


class BlockSequenceCNN(nn.Module):
    """
    Branch 1: 1D Convolutional Neural Network for raw 2,000-element sub-die block readings.
    Extracts local and multi-scale temporal/spatial block patterns.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        # Layer 1: k=11, padding='same'
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=11, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # 2000 -> 500

        # Layer 2: k=11, padding='same'
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding="same")
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # 500 -> 125

        # Layer 3: k=7, padding='same'
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()  # 125

        # Multi-scale pooling: both average (overall level) and max (localized peak/defect)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 2000)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))  # (B, 32, 500)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))  # (B, 64, 125)
        x = self.relu3(self.bn3(self.conv3(x)))              # (B, 128, 125)

        avg_feat = self.avg_pool(x).flatten(1)  # (B, 128)
        max_feat = self.max_pool(x).flatten(1)  # (B, 128)
        block_emb = torch.cat([avg_feat, max_feat], dim=1)  # (B, 256)
        return block_emb


class TabularMLP(nn.Module):
    """
    Branch 2: Multi-Layer Perceptron for normalized tabular features.
    Default for C0: 519 features (500 parametric + 19 spatial).
    """

    def __init__(self, in_features: int = 519, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiResolutionCNN(nn.Module):
    """
    Dual-branch hybrid architecture fusing raw 1D block sequences and tabular features.
    """

    def __init__(
        self,
        num_tabular_features: int = 519,
        block_channels: int = 1,
        block_embed_dim: int = 256,
        tab_hidden_dim: int = 256,
        tab_out_dim: int = 128,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_tabular = num_tabular_features > 0

        # Branch 1: Raw Block CNN
        self.block_cnn = BlockSequenceCNN(in_channels=block_channels, embed_dim=block_embed_dim)

        # Branch 2: Tabular MLP
        if self.use_tabular:
            self.tabular_mlp = TabularMLP(
                in_features=num_tabular_features,
                hidden_dim=tab_hidden_dim,
                out_dim=tab_out_dim,
                dropout=dropout,
            )
            fusion_in_dim = block_embed_dim + tab_out_dim  # 256 + 128 = 384
        else:
            self.tabular_mlp = None
            fusion_in_dim = block_embed_dim  # 256

        # Fusion & Classification Head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, x_blocks: torch.Tensor, x_tabular: torch.Tensor = None) -> torch.Tensor:
        # x_blocks: (B, 1, 2000)
        block_emb = self.block_cnn(x_blocks)  # (B, 256)

        if self.use_tabular and x_tabular is not None:
            tab_emb = self.tabular_mlp(x_tabular)  # (B, 128)
            fused = torch.cat([block_emb, tab_emb], dim=1)  # (B, 384)
        else:
            fused = block_emb

        logits = self.fusion_head(fused).squeeze(-1)  # (B,)
        return logits


def get_model_summary(model: nn.Module, device: str = "cpu") -> str:
    """Returns human-readable architecture summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "=" * 70,
        f"Model: {model.__class__.__name__}",
        "=" * 70,
        str(model),
        "=" * 70,
        f"Total Parameters:     {total_params:,}",
        f"Trainable Parameters: {trainable_params:,}",
        "=" * 70,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test forward pass with dummy data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing MultiResolutionCNN on device: {device}")

    model = MultiResolutionCNN(num_tabular_features=519).to(device)
    print(get_model_summary(model))

    dummy_blocks = torch.randn(4, 1, 2000, device=device)
    dummy_tab = torch.randn(4, 519, device=device)

    out = model(dummy_blocks, dummy_tab)
    print(f"\nForward pass successful! Output logits shape: {out.shape}, values: {out.detach().cpu().numpy()}")
