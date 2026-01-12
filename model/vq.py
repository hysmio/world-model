import jax
from flax import nnx as nn
from jax import numpy as jnp


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        latent_dim: int,
        commitment_cost: float = 0.25,
        *,
        rngs: nn.Rngs,
    ):
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # codebook: K vectors of size latent_dim, shared between encoder and decoder
        init_scale = 1 / num_embeddings
        self.codebook = nn.Param(
            jax.random.uniform(
                rngs.params(),
                (num_embeddings, latent_dim),
                minval=-init_scale,
                maxval=init_scale,
            )
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # x shape: (B, T, num_patches, latent_dim)
        B, T, num_patches, latent_dim = x.shape
        assert (
            latent_dim == self.latent_dim
        ), f"Input dim {latent_dim} != codebook dim {self.latent_dim}"

        # flatten to (B * T * num_patches, latent_dim)
        z_e_flat = x.reshape(B * T * num_patches, self.latent_dim)  # (M, latent_dim)

        # compute squared distances using expanded formula
        codebook = self.codebook.value  # (K, latent_dim)

        # sum each of the M vectors (with axis=-1) -> results in shape (M, 1)
        z_e_sq = (z_e_flat**2).sum(axis=-1, keepdims=True)

        # sum K codebook vectors -> result shape: (K,)
        e_sq = (codebook**2).sum(axis=-1)

        # compute the dot products z_e·e for all pairs -> result shape: (M, K)
        ze = z_e_flat @ codebook.T  # (M, K)

        # distances are the elementwise addition - 2 * ze
        distances = z_e_sq + e_sq - 2 * ze  # (M, K)

        # get the index with the smallest distance (closest)
        encoding_indices = jnp.argmin(distances, axis=-1)  # (M,)

        # gather the closest codebook vectors
        z_q = codebook[encoding_indices]  # (M, latent_dim)
        z_q = z_q.reshape(B, T, num_patches, self.latent_dim)

        # compute the commitment loss
        codebook_loss = jnp.mean((jax.lax.stop_gradient(x) - z_q) ** 2)
        commitment_loss = jnp.mean((x - jax.lax.stop_gradient(z_q)) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator trick for gradient propagation
        z_q = x + jax.lax.stop_gradient(z_q - x)

        return z_q, loss, encoding_indices
