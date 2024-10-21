
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim=64):
        # Initialize the base feature extractor (from Stable Baselines)
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim=cnn_output_dim)

        # Get the dimensions of the image and raw observation space
        image_shape = observation_space.spaces['image'].shape
        raw_obs_shape = observation_space.spaces['raw_observation'].shape[0]

        # CNN to process the image input
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(209024, cnn_output_dim),  # Adjust the linear layer input size based on image shape
            nn.ReLU(),
        )

        # MLP to process the raw observation input
        self.mlp = nn.Sequential(
            nn.Linear(raw_obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Output feature size after combining CNN and MLP features
        self._features_dim = cnn_output_dim + 64

    def forward(self, observations):
        # Process image input through CNN
        image_obs = observations['image'].float() / 255.0  # Normalize image to [0, 1]
        cnn_features = self.cnn(image_obs)

        # Process raw observation through MLP
        raw_obs = observations['raw_observation'].float()
        mlp_features = self.mlp(raw_obs)

        # Concatenate the CNN and MLP features
        return th.cat([cnn_features, mlp_features], dim=1)
