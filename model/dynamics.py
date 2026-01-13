import jax
from flax import nnx as nn
from jax import numpy as jnp

from model.transformer import SpatialLayer, PositionwiseFeedForward


class SpatialBlock(nn.Module):
    """spatial attention + ffn block for dynamics model"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        *,
        rngs: nn.Rngs,
    ):
        self.attn = SpatialLayer(d_model, num_heads, kq_size, dropout, rngs=rngs)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout, rngs=rngs)
        self.norm = nn.LayerNorm(d_model, rngs=rngs)
        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        # x: (B, P, d_model) - reshape to (B, 1, P, d_model) for SpatialLayer
        x = x[:, None, :, :]
        x = self.attn(x, deterministic=deterministic)
        x = x[:, 0, :, :]  # back to (B, P, d_model)

        ff_out = self.ff(x, deterministic=deterministic)
        x = self.norm(x + self.dropout(ff_out, deterministic=deterministic))
        return x



class DynamicsModel(nn.Module):
    """predicts next frame tokens given current frame + action"""

    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,
        action_dim: int,
        num_codes: int,  # VQ codebook size
        num_patches: int,
        num_layers: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        rngs: nn.Rngs,
    ):
        self.d_model = d_model
        self.num_codes = num_codes

        self.token_proj = nn.Linear(latent_dim, d_model, rngs=rngs)
        self.action_proj = nn.Linear(action_dim, d_model, rngs=rngs)
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, num_patches, d_model)) * 0.02
        )
        self.blocks = nn.List([
            SpatialBlock(d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model, rngs=rngs)
        self.output_proj = nn.Linear(d_model, num_codes, rngs=rngs)

    def __call__(
        self,
        z_t: jax.Array,
        action: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """z_t: (B, P, latent_dim), action: (B, action_dim) -> logits (B, P, num_codes)"""
        B, P, _ = z_t.shape

        x = self.token_proj(z_t)  # (B, P, d_model)
        action_emb = self.action_proj(action)[:, None, :]  # (B, 1, d_model)
        x = x + action_emb + self.spatial_pos.value

        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        x = self.output_norm(x)
        return self.output_proj(x)

    def predict_tokens(
        self,
        z_t: jax.Array,
        action: jax.Array,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sample: bool = False,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        """predict next frame token indices"""
        logits = self(z_t, action, deterministic=deterministic)

        if sample:
            assert rng is not None
            logits = logits / temperature
            return jax.random.categorical(rng, logits, axis=-1)
        return jnp.argmax(logits, axis=-1)

    def loss(
        self,
        z_t: jax.Array,
        action: jax.Array,
        target_indices: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """cross-entropy loss for next frame prediction"""
        logits = self(z_t, action, deterministic=deterministic)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_one_hot = jax.nn.one_hot(target_indices, self.num_codes)
        loss = -jnp.sum(target_one_hot * log_probs, axis=-1)
        return jnp.mean(loss)
