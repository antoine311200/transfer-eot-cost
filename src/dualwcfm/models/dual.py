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
