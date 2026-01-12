import jax
from flax import nnx as nn
from jax import numpy as jnp


class PatchEmbedding(nn.Module):
    """Conv2d-based patch embedding (ViT style).

    Alternative to the linear projection in VTEncoder.
    Uses convolution with kernel_size=stride=patch_size to extract patches.
    """

    def __init__(
        self,
        *,
        img_size: int = 96,
        patch_size: int = 16,
        d_model: int = 512,
        rngs: nn.Rngs,
    ):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Conv with kernel_size=stride=patch_size extracts non-overlapping patches
        # Input: (B, H, W, C), Output: (B, H//p, W//p, d_model)
        self.conv = nn.Conv(
            in_features=3,
            out_features=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            rngs=rngs,
        )

        # learned spatial embedding
        self.spatial_embedding = nn.Param(
            jax.random.normal(rngs.params(), (1, self.num_patches, d_model)) * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (B, H, W, C) - JAX/Flax uses channels-last by default
        B = x.shape[0]

        # apply conv to get patches
        x = self.conv(x)  # (B, num_h, num_w, d_model)

        # flatten spatial dimensions
        x = x.reshape(B, -1, x.shape[-1])  # (B, num_patches, d_model)

        # add spatial embedding
        x = x + self.spatial_embedding.value

        return x
