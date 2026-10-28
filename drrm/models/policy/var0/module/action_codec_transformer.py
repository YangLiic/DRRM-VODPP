from collections import OrderedDict

import torch
import torch.nn as nn
from .blocks import Attention, RmsNorm, get_multimodal_cond_pos_embed

class ActionTransformerEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        output_dim: int,
        num_heads: int,
        horizon: int,
        depth: int = 4
    ):
        super().__init__()
        # Token embeddings
        # self.cls_embed = nn.Embedding(1, enc_hidden_dim)
        self.horizon = horizon
        self.action_input_proj = nn.Linear(action_dim, hidden_size)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            Attention(
                dim=hidden_size, num_heads=num_heads, 
                qkv_bias=True, qk_norm=True, 
                norm_layer=RmsNorm
            ) for _ in range(depth)
        ])

        self.norm = RmsNorm(hidden_size, eps=1e-6)

        # Output projection to latent parameters (mu and log_var for each dimension)
        self.latent_output_proj = nn.Linear(hidden_size, output_dim)

        # Fixed positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, horizon, hidden_size)
        )
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize pos_embed by sin-cos embedding
        pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('action', self.horizon),
            ])
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, actions) -> torch.Tensor:
        # Prepare tokens
        tokens = self.action_input_proj(actions)

        # Add positional embeddings
        x = tokens + self.pos_embed

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.latent_output_proj(x)

        return x

class ActionTransformerDecoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        input_dim: int,
        num_heads: int,
        horizon: int,
        depth: int = 4,
        **kwargs
    ):
        super().__init__()
        self.horizon = horizon
        self.latent_input_proj = nn.Linear(input_dim, hidden_size)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            Attention(
                dim=hidden_size, num_heads=num_heads, 
                qkv_bias=True, qk_norm=True, 
                norm_layer=RmsNorm
            ) for _ in range(depth)
        ])

        self.norm = RmsNorm(hidden_size, eps=1e-6)

        # Output projection back to action space
        self.action_output_proj = nn.Linear(hidden_size, action_dim)

        # Fixed positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, horizon, hidden_size)
        )
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize pos_embed by sin-cos embedding
        pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('action', self.horizon),
            ]),
            embed_modality=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, latents) -> torch.Tensor:
        # Prepare tokens
        tokens = self.latent_input_proj(latents)

        # Add positional embeddings
        x = tokens + self.pos_embed

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.action_output_proj(x)

        return x

