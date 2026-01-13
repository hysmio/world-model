import jax
from flax import nnx as nn
from jax import numpy as jnp

from model.transformer import STTransformer
from model.vq import VectorQuantizer


class LatentActionModel(nn.Module):
    """infers latent actions between consecutive frames"""
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
        """z: (B, T, P, latent_dim) -> actions (B, T-1, action_dim), vq_loss, indices"""
        B, T, P, _ = z.shape

        x = self.input_proj(z)
        x = x + self.spatial_pos.value + self.temporal_pos.value[:, :T, :, :]
        x = self.transformer(x, deterministic=deterministic)
        x = self.action_proj(x)
        x = x.mean(axis=2)  # pool patches

        # action = difference between consecutive frame embeddings
        actions = x[:, 1:, :] - x[:, :-1, :]
        actions_q, loss, indices = self.action_vq(actions)

        return actions_q, loss, indices

    def encode_action(
        self, z_t: jax.Array, z_t1: jax.Array, *, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """encode action between two frames -> (action, index)"""
        z = jnp.stack([z_t, z_t1], axis=1)
        actions_q, _, indices = self(z, deterministic=deterministic)
        return actions_q[:, 0, :], indices[:, 0]

    def decode_action(self, indices: jax.Array) -> jax.Array:
        """indices -> action embeddings"""
        return self.action_vq.lookup(indices)
