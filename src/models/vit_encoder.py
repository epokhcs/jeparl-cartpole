"""Vision Transformer Encoder with spatio-temporal positional encoding."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SpatioTemporalPositionalEncoding(nn.Module):
    """Learnable positional encoding for spatial (i, j) and temporal (t) positions."""

    def __init__(
        self,
        d_model: int,
        max_spatial_h: int = 25,
        max_spatial_w: int = 38,
        max_temporal: int = 3,
        dropout: float = 0.1
    ):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension (embedding size)
            max_spatial_h: Maximum height patches
            max_spatial_w: Maximum width patches
            max_temporal: Maximum temporal frames
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Learnable embeddings for each dimension
        self.spatial_h_embed = nn.Embedding(max_spatial_h, d_model)
        self.spatial_w_embed = nn.Embedding(max_spatial_w, d_model)
        self.temporal_embed = nn.Embedding(max_temporal, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute positional encodings.

        Args:
            positions: Tensor of shape (B, N, 3) containing (i, j, t) indices

        Returns:
            Positional encodings of shape (B, N, d_model)
        """
        # Extract position indices
        pos_i = positions[..., 0].long()  # (B, N)
        pos_j = positions[..., 1].long()  # (B, N)
        pos_t = positions[..., 2].long()  # (B, N)

        # Convert temporal from {-2, -1, 0} to {0, 1, 2}
        pos_t = pos_t + 2

        # Get embeddings
        h_embed = self.spatial_h_embed(pos_i)  # (B, N, d_model)
        w_embed = self.spatial_w_embed(pos_j)  # (B, N, d_model)
        t_embed = self.temporal_embed(pos_t)   # (B, N, d_model)

        # Combine embeddings (sum)
        pos_encoding = h_embed + w_embed + t_embed

        return self.dropout(pos_encoding)


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, N, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (B, N, d_model)
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with spatio-temporal positional encoding.

    Processes patches from multiple frames and outputs a global [CLS] token embedding.
    """

    def __init__(
        self,
        patch_dim: int = 768,  # 3 * 16 * 16
        d_model: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        max_spatial_h: int = 25,
        max_spatial_w: int = 38,
        max_temporal: int = 3,
        dropout: float = 0.1
    ):
        """Initialize Vision Transformer encoder.

        Args:
            patch_dim: Dimension of input patches (C * patch_size^2)
            d_model: Model embedding dimension (output dimension)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feedforward dimension (default: 4 * d_model)
            max_spatial_h: Maximum height patches
            max_spatial_w: Maximum width patches
            max_temporal: Maximum temporal frames
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Patch embedding: project patches to d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = SpatioTemporalPositionalEncoding(
            d_model=d_model,
            max_spatial_h=max_spatial_h,
            max_spatial_w=max_spatial_w,
            max_temporal=max_temporal,
            dropout=dropout
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Layer norm for final output
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # Initialize cls token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor,
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """Forward pass through ViT encoder.

        Args:
            patches: Patch tensor of shape (B, N, patch_dim)
            positions: Position tensor of shape (B, N, 3) containing (i, j, t)
            return_all_tokens: If True, return all tokens; otherwise only [CLS]

        Returns:
            If return_all_tokens=False: [CLS] embeddings of shape (B, d_model)
            If return_all_tokens=True: All token embeddings of shape (B, N+1, d_model)
        """
        B, N, _ = patches.shape

        # Embed patches
        x = self.patch_embed(patches)  # (B, N, d_model)

        # Add positional encoding
        pos_enc = self.pos_encoding(positions)  # (B, N, d_model)
        x = x + pos_enc

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_model)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        if return_all_tokens:
            return x
        else:
            # Return only [CLS] token
            return x[:, 0]  # (B, d_model)

    def get_attention_maps(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """Get attention maps from a specific layer (for visualization).

        Args:
            patches: Patch tensor of shape (B, N, patch_dim)
            positions: Position tensor of shape (B, N, 3)
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Attention weights of shape (B, num_heads, N+1, N+1)
        """
        B, N, _ = patches.shape

        # Embed patches
        x = self.patch_embed(patches)
        pos_enc = self.pos_encoding(positions)
        x = x + pos_enc

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Forward through layers up to target layer
        target_layer = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx

        for idx, block in enumerate(self.transformer_blocks):
            if idx < target_layer:
                x = block(x)
            elif idx == target_layer:
                # Get attention weights from this layer
                attn_out, attn_weights = block.attention(x, x, x, average_attn_weights=False)
                return attn_weights  # (B, num_heads, N+1, N+1)

        return None
