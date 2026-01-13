import jax
from flax import nnx as nn
from jax import numpy as jnp


class VectorQuantizer(nn.Module):
    """quantizes last dim to nearest codebook entry, works with any shape"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        *,
        rngs: nn.Rngs,
    ):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        init_scale = 1 / num_embeddings
        self.codebook = nn.Param(
            jax.random.uniform(
                rngs.params(),
                (num_embeddings, embedding_dim),
                minval=-init_scale,
                maxval=init_scale,
            )
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """x: (..., embed_dim) -> z_q, loss, indices"""
        original_shape = x.shape
        assert x.shape[-1] == self.embedding_dim

        # flatten to (N, embed_dim)
        x_flat = x.reshape(-1, self.embedding_dim)
        codebook = self.codebook.value  # (K, embed_dim)

        # squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        x_sq = (x_flat**2).sum(axis=-1, keepdims=True)  # (N, 1)
        e_sq = (codebook**2).sum(axis=-1)  # (K,)
        distances = x_sq + e_sq - 2 * (x_flat @ codebook.T)  # (N, K)

        # nearest codebook entry
        indices_flat = jnp.argmin(distances, axis=-1)  # (N,)
        z_q_flat = codebook[indices_flat]  # (N, embed_dim)

        z_q = z_q_flat.reshape(original_shape)
        indices = indices_flat.reshape(original_shape[:-1])

        # commitment loss
        codebook_loss = jnp.mean((jax.lax.stop_gradient(x) - z_q) ** 2)
        commitment_loss = jnp.mean((x - jax.lax.stop_gradient(z_q)) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator
        z_q = x + jax.lax.stop_gradient(z_q - x)

        return z_q, loss, indices

    def lookup(self, indices: jax.Array) -> jax.Array:
        return self.codebook.value[indices]
