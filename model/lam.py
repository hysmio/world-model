import jax
from flax import nnx as nn
from jax import numpy as jnp

from model.transformer import STTransformer
from model.vq import VectorQuantizer

# infers latent actions from frames t & t1
class LatentActionModel(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,  # input dimension from video tokenizer
        action_dim: int,  # dimension of latent actions
        num_actions: int,  # number of discrete actions in codebook
        num_patches: int,
        max_frames: int,
        # Transformer
        num_layers: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        rngs: nn.Rngs,
    ):
        self.d_model = d_model
        self.action_dim = action_dim

        # project from video latent dim to LAM d_model
        self.input_proj = nn.Linear(latent_dim, d_model, rngs=rngs)

        # learned positional embeddings
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, 1, num_patches, d_model)) * 0.02
        )
        self.temporal_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, max_frames, 1, d_model)) * 0.02
        )

        # transformer to process frames and infer actions
        self.transformer = STTransformer(
            num_layers, d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs
        )

        # project down to action dimension
        self.action_proj = nn.Linear(d_model, action_dim, rngs=rngs)

        # reuse generic VQ for discrete actions
        self.action_vq = VectorQuantizer(num_actions, action_dim, rngs=rngs)

    def __call__(
        self, z: jax.Array, *, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Infer latent actions from tokenized video frames.

        Args:
            z: Tokenized video frames (B, T, num_patches, latent_dim)
            deterministic: Whether to use deterministic mode (no dropout)

        Returns:
            actions: Quantized latent actions (B, T-1, action_dim)
            loss: VQ commitment loss
            indices: Action indices (B, T-1)
        """
        B, T, P, _ = z.shape

        # project to d_model
        x = self.input_proj(z)  # (B, T, P, d_model)

        # add positional embeddings
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]

        # run through transformer
        x = self.transformer(x, deterministic=deterministic)  # (B, T, P, d_model)

        # project to action dimension
        x = self.action_proj(x)  # (B, T, P, action_dim)

        # pool across patches (mean pooling)
        x = x.mean(axis=2)  # (B, T, action_dim)

        # compute actions as difference between consecutive frames
        # action_t represents the action that takes frame t to frame t+1
        actions = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, action_dim)

        # quantize actions
        actions_q, loss, indices = self.action_vq(actions)

        return actions_q, loss, indices

    def encode_action(
        self, z_t: jax.Array, z_t1: jax.Array, *, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """Encode action between two specific frames.

        Args:
            z_t: Current frame tokens (B, num_patches, latent_dim)
            z_t1: Next frame tokens (B, num_patches, latent_dim)

        Returns:
            action: Quantized action (B, action_dim)
            index: Action index (B,)
        """
        # stack frames: (B, 2, P, latent_dim)
        z = jnp.stack([z_t, z_t1], axis=1)

        actions_q, _, indices = self(z, deterministic=deterministic)

        # only one transition, so squeeze
        return actions_q[:, 0, :], indices[:, 0]

    def decode_action(self, indices: jax.Array) -> jax.Array:
        """Decode action indices to action embeddings (for inference).

        Args:
            indices: Action indices (B,) or (B, T-1)

        Returns:
            actions: Action embeddings with same shape + action_dim
        """
        return self.action_vq.lookup(indices)
