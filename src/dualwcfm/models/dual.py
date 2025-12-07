import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDualPotential(nn.Module):
    """
    Neural network for learning dual potentials f: image -> R
    Used in KL-regularized optimal transport dual formulation.

    This network takes an image as input and outputs a scalar value f(x)
    representing the dual potential at that point.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=32,
        hidden_dims=[64, 128, 256, 512],
        dropout_rate=0.1,
        use_batchnorm=True,
        activation="leaky_relu",
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            img_size: Input image size (assumed square)
            hidden_dims: List of hidden dimensions for convolutional layers
            dropout_rate: Dropout rate for regularization
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size

        # Store activation choice
        self.activation = activation

        # Build convolutional feature extractor
        conv_layers = []
        current_size = img_size
        current_channels = in_channels

        for i, hidden_dim in enumerate(hidden_dims):
            conv_layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=3, stride=2, padding=1))

            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(hidden_dim))

            if activation == "leaky_relu":
                conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "elu":
                conv_layers.append(nn.ELU(inplace=True))
            else:
                conv_layers.append(nn.ReLU(inplace=True))

            conv_layers.append(nn.Dropout2d(dropout_rate))

            current_channels = hidden_dim
            current_size = current_size // 2

        self.feature_extractor = nn.Sequential(*conv_layers)

        # Calculate flattened size
        self.flattened_size = current_channels * current_size * current_size

        # Final regression head to single scalar
        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.LayerNorm(256) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == "leaky_relu" else (nn.ELU() if activation == "elu" else nn.ReLU()),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == "leaky_relu" else (nn.ELU() if activation == "elu" else nn.ReLU()),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2) if activation == "leaky_relu" else (nn.ELU() if activation == "elu" else nn.ReLU()),
            nn.Dropout(dropout_rate),
            # Final linear layer to single scalar output
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu" if self.activation == "leaky_relu" else "relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass: image -> scalar dual potential

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)

        Returns:
            f: Scalar dual potential values of shape (batch_size, 1)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Flatten
        features = features.view(features.size(0), -1)

        # Regress to scalar
        f = self.regressor(features)

        return f

    def forward_with_features(self, x):
        """
        Forward pass that also returns intermediate features
        Useful for debugging or feature analysis
        """
        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        f = self.regressor(flattened)
        return f, features


class ResNetDualPotential(nn.Module):
    """ResNet-based architecture for higher resolution images"""

    def __init__(self, in_channels=3, base_dim=64, num_blocks=4):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        current_dim = base_dim

        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(current_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_dim * 2, current_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(current_dim * 2),
            )
            self.blocks.append(block)
            current_dim *= 2

        # Global average pooling and final linear layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(current_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self.blocks:
            identity = x
            if x.shape[1] != block[-2].weight.shape[0]:  # type: ignore
                # Projection shortcut if dimensions change
                identity = nn.functional.avg_pool2d(identity, 2)
                identity = torch.cat([identity, identity], dim=1)
            x = block(x) + identity
            x = self.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class EfficientDualPotential(nn.Module):
    """Optimized architecture for 32x32 images"""

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()

        # Feature extractor with residual connections
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Downsample blocks with skip connections
        self.block1 = self._make_block(base_channels, base_channels * 2)
        self.block2 = self._make_block(base_channels * 2, base_channels * 4)
        self.block3 = self._make_block(base_channels * 4, base_channels * 8)

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, 64),
            nn.LayerNorm(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(inplace=True),
            nn.Linear(32, 1),
        )

        self.elu = nn.ELU(inplace=True)

        # Initialize
        self._initialize_weights()

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # Residual blocks
        identity = x
        x = self.block1(x)
        if identity.size(1) != x.size(1):
            # Projection shortcut
            identity = nn.functional.avg_pool2d(identity, 2)
            identity = torch.cat([identity] * (x.size(1) // identity.size(1)), dim=1)
        x = x + identity
        x = self.elu(x)

        identity = x
        x = self.block2(x)
        if identity.size(1) != x.size(1):
            identity = nn.functional.avg_pool2d(identity, 2)
            identity = torch.cat([identity] * (x.size(1) // identity.size(1)), dim=1)
        x = x + identity
        x = self.elu(x)

        identity = x
        x = self.block3(x)
        if identity.size(1) != x.size(1):
            identity = nn.functional.avg_pool2d(identity, 2)
            identity = torch.cat([identity] * (x.size(1) // identity.size(1)), dim=1)
        x = x + identity
        x = self.elu(x)

        # Pool and regress
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
