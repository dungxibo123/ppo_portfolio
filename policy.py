import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_assets, window_size = observation_space.shape

        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=features_dim,
            batch_first=True
        )

        self._features_dim = features_dim

    def forward(self, observations):
        # observations: (batch, assets, window)
        x = observations.permute(0, 2, 1)  # -> (batch, window, assets)
        _, (h, _) = self.lstm(x)
        return h[-1]



class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, n_heads=4):
        super().__init__(observation_space, features_dim)

        n_assets, window = observation_space.shape

        # Temporal encoding
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=features_dim,
            batch_first=True
        )

        # Cross-asset attention
        self.attn = nn.MultiheadAttention(
            embed_dim=features_dim,
            num_heads=n_heads,
            batch_first=True
        )

        self.fc = nn.Linear(features_dim, features_dim)
        self._features_dim = features_dim

    def forward(self, obs):
        # (batch, assets, window) → (batch, window, assets)
        x = obs.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)   # (batch, window, dim)

        # Use last timestep
        h = lstm_out[:, -1, :]       # (batch, dim)

        # Expand for attention (fake sequence of assets)
        h = h.unsqueeze(1)           # (batch, 1, dim)

        attn_out, _ = self.attn(h, h, h)

        out = self.fc(attn_out.squeeze(1))
        return out
