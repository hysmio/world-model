import jax
from flax import nnx as nn
from jax import numpy as jnp


class SpatialBlock(nn.Module):
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
        self.num_heads = num_heads
        self.kq_size = kq_size
        self.scale = kq_size**-0.5

        self.q_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.k_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.v_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.out_proj = nn.Linear(num_heads * kq_size, d_model, rngs=rngs)

        self.ff_1 = nn.Linear(d_model, d_ff, rngs=rngs)
        self.ff_2 = nn.Linear(d_ff, d_model, rngs=rngs)

        self.norm1 = nn.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nn.LayerNorm(d_model, rngs=rngs)
        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        # x shape: (B, P, d_model)
        B, P, _ = x.shape

        # self-attention
        q = self.q_proj(x).reshape(B, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=deterministic)

        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, P, self.num_heads * self.kq_size)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out, deterministic=deterministic))

        # feedforward
        ff = self.ff_1(x)
        ff = jax.nn.gelu(ff)
        ff = self.dropout(ff, deterministic=deterministic)
        ff = self.ff_2(ff)

        x = self.norm2(x + self.dropout(ff, deterministic=deterministic))

        return x


# Dynamics Model - predicts next frame tokens given current frame + action
class DynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        latent_dim: int,
        action_dim: int,
        num_codes: int,  # video tokenizer codebook size
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

        # project video tokens to d_model
        self.token_proj = nn.Linear(latent_dim, d_model, rngs=rngs)

        # project action to d_model for conditioning
        self.action_proj = nn.Linear(action_dim, d_model, rngs=rngs)

        # learned spatial positional embeddings
        self.spatial_pos = nn.Param(
            jax.random.normal(rngs.params(), (1, num_patches, d_model)) * 0.02
        )

        # spatial-only transformer blocks
        self.blocks = nn.List([
            SpatialBlock(d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs)
            for _ in range(num_layers)
        ])

        # output head: logits over video codebook
        self.output_norm = nn.LayerNorm(d_model, rngs=rngs)
        self.output_proj = nn.Linear(d_model, num_codes, rngs=rngs)

    def __call__(
        self,
        z_t: jax.Array,
        action: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        # z_t: (B, num_patches, latent_dim)
        # action: (B, action_dim)
        # returns: logits (B, num_patches, num_codes)
        B, P, _ = z_t.shape

        x = self.token_proj(z_t)  # (B, P, d_model)

        # broadcast action to all patches
        action_emb = self.action_proj(action)[:, None, :]  # (B, 1, d_model)
        x = x + action_emb

        x = x + self.spatial_pos.value

        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        x = self.output_norm(x)
        logits = self.output_proj(x)

        return logits

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
        # returns: token indices (B, num_patches)
        logits = self(z_t, action, deterministic=deterministic)

        if sample:
            assert rng is not None, "rng required for sampling"
            logits = logits / temperature
            indices = jax.random.categorical(rng, logits, axis=-1)
        else:
            indices = jnp.argmax(logits, axis=-1)

        return indices

    def loss(
        self,
        z_t: jax.Array,
        action: jax.Array,
        target_indices: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        # cross-entropy loss for next frame prediction
        logits = self(z_t, action, deterministic=deterministic)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_one_hot = jax.nn.one_hot(target_indices, self.num_codes)
        loss = -jnp.sum(target_one_hot * log_probs, axis=-1)

        return jnp.mean(loss)
