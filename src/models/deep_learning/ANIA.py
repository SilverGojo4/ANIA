# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
"""
ANIA model for MIC prediction in the AMP-MIC project.

This module defines the ANIA model, which processes FCGR features using an Inception-based architecture,
followed by Transformer Encoder and Dense layers to predict Minimum Inhibitory Concentration (MIC) values.
It includes explainability features such as learnable branch weights and Grad-CAM visualization.
"""
# ============================== Third-Party Library Imports ==============================
import torch
from torch import nn


# ============================== Custom Function ==============================
class BasicConv2d(nn.Module):
    """
    A basic Conv2D block with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BasicConv2d block.
        """
        return self.relu(self.bn(self.conv(x)))


class FCGRInceptionModule(nn.Module):
    """
    Inception Module for FCGR features in the AMP-MIC project, redesigned to align with Inception v3 principles.

    This module processes FCGR features (6 channels, 16x16) using a multi-branch structure,
    incorporating 1x1, decomposed 3x3 (1x3 and 3x1), and decomposed 5x5 (two 3x3) convolutions,
    along with a pooling branch (now using MaxPool).
    """

    def __init__(
        self,
        in_channels: int = 11,  # Fixed for CGR features
        out_channels: int = 256,  # Total output channels
        branch1x1_channels: int = 64,  # Branch 1 output channels
        branch3x3_channels: int = 96,  # Branch 2 output channels (decomposed 3x3)
        branch3x3_reduction: int = 64,  # Reduction channels for 3x3 branch
        branch5x5_channels: int = 64,  # Branch 3 output channels (decomposed 5x5)
        branch5x5_reduction: int = 48,  # Reduction channels for 5x5 branch
        branch_pool_channels: int = 32,  # Branch 4 output channels
    ):
        """
        Initialize the FCGR Inception Module.

        Parameters
        ----------
        in_channels : int
            Number of input channels (default is 6 for multi-channel FCGR features).
        out_channels : int
            Total number of output channels (default is 256).
        branch1x1_channels : int
            Output channels for Branch 1 (default is 64).
        branch3x3_channels : int
            Output channels for Branch 2 (default is 96).
        branch3x3_reduction : int
            Reduction channels for Branch 2 (default is 64).
        branch5x5_channels : int
            Output channels for Branch 3 (default is 64).
        branch5x5_reduction : int
            Reduction channels for Branch 3 (default is 48).
        branch_pool_channels : int
            Output channels for Branch 4 (default is 32).
        """
        super().__init__()

        # Validate total output channels
        expected_out_channels = (
            branch1x1_channels
            + branch3x3_channels
            + branch5x5_channels
            + branch_pool_channels
        )
        if out_channels != expected_out_channels:
            raise ValueError(
                f"out_channels ({out_channels}) must equal sum of branch channels ({expected_out_channels})"
            )

        # Branch 1: 1x1 Conv
        self.branch1x1 = BasicConv2d(
            in_channels,
            branch1x1_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        # Branch 2: 1x1 Conv -> 1x3 Conv -> 3x1 Conv (decomposed 3x3)
        self.branch3x3_1 = BasicConv2d(
            in_channels,
            branch3x3_reduction,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.branch3x3_2 = BasicConv2d(
            branch3x3_reduction,
            branch3x3_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),  # Keep 16x16
        )
        self.branch3x3_3 = BasicConv2d(
            branch3x3_channels,
            branch3x3_channels,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),  # Keep 16x16
        )

        # Branch 3: 1x1 Conv -> 3x3 Conv -> 3x3 Conv (decomposed 5x5)
        self.branch5x5_1 = BasicConv2d(
            in_channels,
            branch5x5_reduction,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.branch5x5_2 = BasicConv2d(
            branch5x5_reduction,
            branch5x5_reduction,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.branch5x5_3 = BasicConv2d(
            branch5x5_reduction,
            branch5x5_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Branch 4: MaxPool -> 1x1 Conv (changed from AvgPool)
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=1, padding=1
            ),  # Changed to MaxPool, keep 16x16
            BasicConv2d(
                in_channels,
                branch_pool_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
        )

        # Learnable weights for each branch
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)

        # Register hooks to save features and gradients
        self.features = {}
        self.gradients = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCGR Inception Module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 11, 16, 16).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, 16, 16).
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
        branch_pool = self.branch_pool(x)

        # Always save features (even in eval mode)
        self.features["branch1x1"] = branch1x1
        self.features["branch3x3"] = branch3x3
        self.features["branch5x5"] = branch5x5
        self.features["branch_pool"] = branch_pool

        # Apply branch weights
        weights = torch.softmax(self.branch_weights, dim=0)
        branch1x1 = branch1x1 * weights[0]
        branch3x3 = branch3x3 * weights[1]
        branch5x5 = branch5x5 * weights[2]
        branch_pool = branch_pool * weights[3]

        out = torch.cat(
            [branch1x1, branch3x3, branch5x5, branch_pool], dim=1
        )  # (batch_size, 256, 16, 16)

        # Save output for Grad-CAM
        if out.requires_grad:
            out.register_hook(self._save_gradients)
            self.features["output"] = out

        return out

    def _save_gradients(self, grad: torch.Tensor) -> None:
        """
        Hook to save gradients of the output.

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the output.
        """
        self.gradients["output"] = grad

    def get_branch_weights(self) -> torch.Tensor:
        """
        Get the learned branch weights.

        Returns
        -------
        torch.Tensor
            The learned weights for each branch.
        """
        return self.branch_weights

    def get_features(self) -> dict:
        """
        Get the features of each branch.

        Returns
        -------
        dict
            Dictionary containing the features of each branch.
        """
        return self.features

    def get_gradients(self) -> dict:
        """
        Get the gradients of the output.

        Returns
        -------
        dict
            Dictionary containing the gradients of the output.
        """
        return self.gradients


class ANIA(nn.Module):
    """
    ANIA (FCGR-specific part of An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration)
    for MIC prediction in the AMP-MIC project.

    This model processes FCGR features through two Inception Modules with residual connections,
    followed by Flatten, Multi-Head Attention, and Dense layers to predict MIC values.
    """

    def __init__(
        self,
        in_channels: int = 11,
        # Inception 1
        inception1_out_channels: int = 192,
        inception1_branch1x1_channels: int = 48,
        inception1_branch3x3_channels: int = 64,
        inception1_branch3x3_reduction: int = 48,
        inception1_branch5x5_channels: int = 48,
        inception1_branch5x5_reduction: int = 32,
        inception1_branch_pool_channels: int = 32,
        # Inception 2
        inception2_out_channels: int = 256,
        inception2_branch1x1_channels: int = 64,
        inception2_branch3x3_channels: int = 96,
        inception2_branch3x3_reduction: int = 64,
        inception2_branch5x5_channels: int = 64,
        inception2_branch5x5_reduction: int = 48,
        inception2_branch_pool_channels: int = 32,
        # Transformer encoder settings
        num_heads: int = 8,
        d_model: int = 512,
        num_encoder_layers: int = 2,
        dense_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the ANIA model.

        Parameters
        ----------
        in_channels : int
            Number of input channels for FCGR features (default is 11 for multi-channel FCGR features).
        inception1_out_channels : int
            Number of output channels from the first Inception Module (default is 256).
        inception2_out_channels : int
            Number of output channels from the second Inception Module (default is 384).
        inception1_branch1x1_channels : int
            Output channels for Branch 1 of the first Inception Module (default is 64).
        inception1_branch3x3_channels : int
            Output channels for Branch 2 of the first Inception Module (default is 96).
        inception1_branch3x3_reduction : int
            Reduction channels for Branch 2 of the first Inception Module (default is 64).
        inception1_branch5x5_channels : int
            Output channels for Branch 3 of the first Inception Module (default is 64).
        inception1_branch5x5_reduction : int
            Reduction channels for Branch 3 of the first Inception Module (default is 48).
        inception1_branch_pool_channels : int
            Output channels for Branch 4 of the first Inception Module (default is 32).
        inception2_branch1x1_channels : int
            Output channels for Branch 1 of the second Inception Module (default is 96).
        inception2_branch3x3_channels : int
            Output channels for Branch 2 of the second Inception Module (default is 128).
        inception2_branch3x3_reduction : int
            Reduction channels for Branch 2 of the second Inception Module (default is 96).
        inception2_branch5x5_channels : int
            Output channels for Branch 3 of the second Inception Module (default is 96).
        inception2_branch5x5_reduction : int
            Reduction channels for Branch 3 of the second Inception Module (default is 64).
        inception2_branch_pool_channels : int
            Output channels for Branch 4 of the second Inception Module (default is 64).
        num_heads : int
            Number of attention heads in Multi-Head Attention (default is 8).
        d_model : int
            Dimension of the model for Multi-Head Attention (default is 512).
        dense_hidden_dim : int
            Hidden dimension of the dense layer (default is 256).
        dropout_rate : float
            Dropout rate for regularization (default is 0.3).
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        # First Inception Module
        self.inception1 = FCGRInceptionModule(
            in_channels=in_channels,
            out_channels=inception1_out_channels,
            branch1x1_channels=inception1_branch1x1_channels,
            branch3x3_channels=inception1_branch3x3_channels,
            branch3x3_reduction=inception1_branch3x3_reduction,
            branch5x5_channels=inception1_branch5x5_channels,
            branch5x5_reduction=inception1_branch5x5_reduction,
            branch_pool_channels=inception1_branch_pool_channels,
        )

        # Second Inception Module
        self.inception2 = FCGRInceptionModule(
            in_channels=inception1_out_channels,
            out_channels=inception2_out_channels,
            branch1x1_channels=inception2_branch1x1_channels,
            branch3x3_channels=inception2_branch3x3_channels,
            branch3x3_reduction=inception2_branch3x3_reduction,
            branch5x5_channels=inception2_branch5x5_channels,
            branch5x5_reduction=inception2_branch5x5_reduction,
            branch_pool_channels=inception2_branch_pool_channels,
        )

        # Residual projection
        self.residual_projection = nn.Conv2d(
            inception1_out_channels,
            inception2_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Spatial pooling
        self.spatial_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear projection to Transformer input dim
        self.projection = nn.Sequential(
            nn.Linear(inception2_out_channels, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Dense regression head
        self.dense = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, dense_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden_dim, 1),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ANIA model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted MIC value (batch_size, 1) and attention weights.
        """
        # Step 1: Inception Modules with Residual
        x = self.inception1(x)  # (B, C1, H, W)
        residual = x
        x = self.inception2(x)  # (B, C2, H, W)
        residual = self.residual_projection(residual)  # match channel
        x = x + residual  # residual add

        # Step 2: Pooling
        x = self.spatial_pool(x)  # (B, C2, 8, 8)

        # Step 3: Reshape & Project
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 64, C)
        x = self.projection(x)  # (B, 64, d_model)

        # Step 4: Transformer Encoder
        x = self.transformer_encoder(x)  # (B, 64, d_model)

        # Step 5: Global Pooling + Dense Regression
        x = x.mean(dim=1)  # (B, d_model)
        out = self.dense(x)  # (B, 1)

        return out

    def _initialize_weights(self) -> None:
        """
        Initialize weights of the model using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def compute_gradcam(self, block: str, branch: str) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a specified block and branch.

        Parameters
        ----------
        block : str
            Inception block name ('inception1' or 'inception2').
        branch : str
            Branch name in the inception module ('branch1x1', 'branch3x3', 'branch5x5', 'branch_pool', or 'output').

        Returns
        -------
        torch.Tensor
            Grad-CAM heatmap of shape (height, width), normalized to [0, 1].
        """
        # Select Inception module
        module = self.inception1 if block == "inception1" else self.inception2

        # Extract features and gradients
        features = module.get_features()[branch]  # (B, C, H, W)
        gradients = module.get_gradients()["output"]  # (B, C, H, W)

        # Check shape match
        if features.shape != gradients.shape:
            raise ValueError(
                f"Features shape {features.shape} does not match gradients shape {gradients.shape}"
            )

        # Compute Grad-CAM weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * features).sum(dim=1)  # (B, H, W)
        cam = torch.relu(cam)  # ReLU activation
        cam = cam / (cam.max() + 1e-8)  # Normalize to [0, 1]

        return cam[0]  # Return heatmap of first sample
