import jax
from flax import nnx as nn
from jax import numpy as jnp

from model.transformer import STTransformer
from model.vq import VectorQuantizer


class VTEncoder(nn.Module):
    """video tokenizer encoder - pixels to discrete tokens via VQ"""
    def __init__(
        self,
        *,
        d_model: int,
        # Patch size
        patch_size: int,
        num_patches: int,
        max_frames: int,
        # ST Transformer
        st_blocks: int,
        num_heads: int,
        # key / query size
        kq_size: int,
        d_ff: int,
        dropout: float,
        # Vector Quantizer
        num_codes: int,
        latent_dim: int,
        rngs: nn.Rngs,
    ):
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size  # rgb

        self.patch_proj = nn.Linear(self.patch_dim, d_model, rngs=rngs)
        self.down_proj = nn.Linear(d_model, latent_dim, rngs=rngs)
        self.vq = VectorQuantizer(num_codes, latent_dim, rngs=rngs)

        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, 1, num_patches, d_model)) * 0.02
        )
        self.temporal_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, max_frames, 1, d_model)) * 0.02
        )

        self.st_transformer = STTransformer(
            st_blocks, d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs
        )

    def __call__(
        self, x: jax.Array, *, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        num_h, num_w = H // self.patch_size, W // self.patch_size
        num_patches = num_h * num_w

        # patchify
        x = x.reshape(B, T, C, num_h, self.patch_size, num_w, self.patch_size)
        # (B, T, C, num_h, p, num_w, p) -> (B, T, num_h, num_w, C, p, p)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6)
        # flatten to (B, T, num_patches, patch_dim)
        x = x.reshape(B, T, num_patches, self.patch_dim)

        # (B, T, P, d_model)
        x = self.patch_proj(x)
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]
        x = self.st_transformer(x, deterministic=deterministic)
        # (B, T, P, latent_dim)
        x = self.down_proj(x)
        x, loss, indices = self.vq(x)

        return x, loss, indices


class VTDecoder(nn.Module):
    """video tokenizer decoder - tokens back to pixels"""

    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,
        patch_size: int,
        num_patches: int,
        max_frames: int,
        st_blocks: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        rngs: nn.Rngs,
    ):
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size

        self.up_proj = nn.Linear(latent_dim, d_model, rngs=rngs)
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, 1, num_patches, d_model)) * 0.02
        )
        self.temporal_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, max_frames, 1, d_model)) * 0.02
        )
        self.st_transformer = STTransformer(
            st_blocks, d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs
        )
        self.patch_proj = nn.Linear(d_model, self.patch_dim, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, img_height: int, img_width: int, deterministic: bool = False
    ) -> jax.Array:
        # x: (B, T, num_patches, latent_dim)
        B, T, num_patches, _ = x.shape

        # (B, T, P, d_model)
        x = self.up_proj(x)
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]
        x = self.st_transformer(x, deterministic=deterministic)
        # (B, T, P, patch_dim)
        x = self.patch_proj(x)

        # unpatchify
        num_h = img_height // self.patch_size
        num_w = img_width // self.patch_size
        # (B, T, P, C*p*p) -> (B, T, num_h, num_w, C, p, p)
        x = x.reshape(B, T, num_h, num_w, 3, self.patch_size, self.patch_size)
        # (B, T, num_h, num_w, C, p, p) -> (B, T, C, num_h, p, num_w, p)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6)
        # (B, T, C, H, W)
        x = x.reshape(B, T, 3, img_height, img_width)

        return x
