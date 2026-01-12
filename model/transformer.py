import jax
from flax import nnx as nn
from jax import numpy as jnp


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, *, rngs: nn.Rngs):
        self.w_1 = nn.Linear(d_model, d_ff, rngs=rngs)
        self.w_2 = nn.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        x = self.w_1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.w_2(x)
        return x


class SpatialLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, kq_size: int, dropout: float, *, rngs: nn.Rngs
    ):
        self.num_heads = num_heads
        self.kq_size = kq_size
        self.scale = kq_size**-0.5

        # q, k, v projections with kq_size per head
        self.q_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.k_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.v_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.out_proj = nn.Linear(num_heads * kq_size, d_model, rngs=rngs)

        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)
        self.layer_norm = nn.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        B, T, P, E = x.shape

        # reshape for spatial attention, patches attend to each other within each frame
        # (B, T, P, E) -> (B*T, P, E)
        x_reshaped = x.reshape(B * T, P, E)

        # project to q, k, v
        q = self.q_proj(x_reshaped)  # (B*T, P, num_heads * kq_size)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # reshape to separate heads: (B*T, P, H, kq_size) -> (B*T, H, P, kq_size)
        q = q.reshape(B * T, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        k = k.reshape(B * T, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        v = v.reshape(B * T, P, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)

        # scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B*T, H, P, P)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=deterministic)

        # apply attention to values
        out = attn @ v  # (B*T, H, P, kq_size)

        # concat heads and project back to d_model
        # (B*T, H, P, kq_size) -> (B*T, P, H * kq_size)
        out = out.transpose(0, 2, 1, 3).reshape(B * T, P, self.num_heads * self.kq_size)
        out = self.out_proj(out)  # (B*T, P, d_model)

        # reshape back to (B, T, P, E)
        out = out.reshape(B, T, P, E)

        # residual + norm
        x = x + self.dropout(out, deterministic=deterministic)
        x = self.layer_norm(x)

        return x


class TemporalLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, kq_size: int, dropout: float, *, rngs: nn.Rngs
    ):
        self.num_heads = num_heads
        self.kq_size = kq_size
        self.scale = kq_size**-0.5

        self.q_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.k_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.v_proj = nn.Linear(d_model, num_heads * kq_size, rngs=rngs)
        self.out_proj = nn.Linear(num_heads * kq_size, d_model, rngs=rngs)

        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)
        self.layer_norm = nn.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        B, T, P, E = x.shape

        # reshape for temporal attention, frames attend to each other for each patch
        # (B, T, P, E) -> (B*P, T, E)
        x_reshaped = x.transpose(0, 2, 1, 3).reshape(B * P, T, E)

        # project to q, k, v
        q = self.q_proj(x_reshaped)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # reshape to separate heads: (B*P, T, H, kq_size) -> (B*P, H, T, kq_size)
        q = q.reshape(B * P, T, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        k = k.reshape(B * P, T, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)
        v = v.reshape(B * P, T, self.num_heads, self.kq_size).transpose(0, 2, 1, 3)

        # attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B*P, H, T, T)

        # causal mask, frame t can only attend to frames 0..t
        causal_mask = jnp.triu(jnp.full((T, T), float("-inf")), k=1)
        attn = attn + causal_mask

        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=deterministic)

        # apply attention to values
        out = attn @ v  # (B*P, H, T, kq_size)

        # concat heads and project back
        # (B*P, H, T, kq_size) -> (B*P, T, H * kq_size)
        out = out.transpose(0, 2, 1, 3).reshape(B * P, T, self.num_heads * self.kq_size)
        out = self.out_proj(out)  # (B*P, T, d_model)

        # reshape back to (B, T, P, E)
        out = out.reshape(B, P, T, E).transpose(0, 2, 1, 3)

        # residual + norm
        x = x + self.dropout(out, deterministic=deterministic)
        x = self.layer_norm(x)

        return x


class SpatioTemporalBlock(nn.Module):
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
        self.spatial_layer = SpatialLayer(d_model, num_heads, kq_size, dropout, rngs=rngs)
        self.temporal_layer = TemporalLayer(d_model, num_heads, kq_size, dropout, rngs=rngs)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, rngs=rngs)
        self.layer_norm = nn.LayerNorm(d_model, rngs=rngs)
        self.dropout = nn.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        x = self.spatial_layer(x, deterministic=deterministic)
        x = self.temporal_layer(x, deterministic=deterministic)
        ff_output = self.feed_forward(x, deterministic=deterministic)

        x = x + self.dropout(ff_output, deterministic=deterministic)
        x = self.layer_norm(x)

        return x


class STTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        kq_size: int,
        d_ff: int,
        dropout: float,
        *,
        rngs: nn.Rngs,
    ):
        self.blocks = nn.List([
            SpatioTemporalBlock(d_model, num_heads, kq_size, d_ff, dropout, rngs=rngs)
            for _ in range(num_layers)
        ])

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        return x
