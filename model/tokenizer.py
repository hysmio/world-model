import jax
from flax import nnx as nn
from jax import numpy as jnp

from model.transformer import STTransformer
from model.vq import VectorQuantizer


# Vision Tokenizer Encoder, turns pixels into tokens
# Currently just uses very basic patching, Genie paper mentioned
# patching explicitly, but also said based on ViT which uses Conv2d
class VTEncoder(nn.Module):
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
        # Patch dimension is the number of channels * the pixels in the patch
        # assume rgb
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_proj = nn.Linear(self.patch_dim, d_model, rngs=rngs)

        # project from encoder d_model to latent_dim for VQ
        self.down_proj = nn.Linear(d_model, latent_dim, rngs=rngs)

        # codebook only handles quantization, no d_model projections
        self.vq = VectorQuantizer(num_codes, latent_dim, rngs=rngs)

        # learned positional embeddings
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, 1, num_patches, d_model)) * 0.02
        )
        self.temporal_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, max_frames, 1, d_model)) * 0.02
        )

        self.st_transformer = STTransformer(
            st_blocks, d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs
        )

    # turns x (B, T, C, H, W) into (B, T, num_patches, latent_dim)
    def __call__(
        self, x: jax.Array, *, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # if patch_size = 4, H = 8, W = 8, then it should be 4 patches, 2 along each axis
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        num_h = H // self.patch_size
        num_w = W // self.patch_size

        # reshape to expose patch grid
        # (B, T, C, num_h, patch_size, num_w, patch_size)
        x = x.reshape(B, T, C, num_h, self.patch_size, num_w, self.patch_size)

        # permute so patches are a 2d array containing the C * patch_size * patch_size values
        # (B, T, num_h, num_w, C, patch_size, patch_size)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6)

        # flatten the spatial grid and patch dimensions
        x = x.reshape(B, T, num_patches, self.patch_dim)

        # project the patches from (B, T, num_patches, patch_dim) to (B, T, num_patches, d_model)
        x = self.patch_proj(x)

        # add spatial & temporal embeddings
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]

        # add transformer blocks
        x = self.st_transformer(x, deterministic=deterministic)

        # project down to latent dimension before VQ
        x = self.down_proj(x)

        # convert into discrete token codebook
        x, loss, indices = self.vq(x)

        return x, loss, indices


class VTDecoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,
        patch_size: int,
        num_patches: int,
        max_frames: int,
        # ST Transformer
        st_blocks: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        rngs: nn.Rngs,
    ):
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size

        # project from latent_dim back to d_model
        self.up_proj = nn.Linear(latent_dim, d_model, rngs=rngs)

        # learned positional embeddings for decoder
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, 1, num_patches, d_model)) * 0.02
        )
        self.temporal_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, max_frames, 1, d_model)) * 0.02
        )

        # use the same transformer architecture for decoding
        self.st_transformer = STTransformer(
            st_blocks, d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs
        )

        # project from d_model back to patch pixels
        self.patch_proj = nn.Linear(d_model, self.patch_dim, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, img_height: int, img_width: int, deterministic: bool = False
    ) -> jax.Array:
        # x shape: (B, T, num_patches, latent_dim)
        B, T, num_patches, _ = x.shape

        # project up to d_model
        x = self.up_proj(x)

        # add positional embeddings
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]

        # run through transformer
        x = self.st_transformer(x, deterministic=deterministic)

        # project to patch pixels
        x = self.patch_proj(x)  # (B, T, num_patches, patch_dim)

        # reshape back to image
        num_h = img_height // self.patch_size
        num_w = img_width // self.patch_size

        # (B, T, num_patches, C * p * p) -> (B, T, num_h, num_w, C, p, p)
        x = x.reshape(B, T, num_h, num_w, 3, self.patch_size, self.patch_size)

        # (B, T, num_h, num_w, C, p, p) -> (B, T, C, num_h, p, num_w, p)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6)

        # (B, T, C, num_h, p, num_w, p) -> (B, T, C, H, W)
        x = x.reshape(B, T, 3, img_height, img_width)

        return x
