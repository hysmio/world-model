import jax
from flax import nnx as nn
from jax import numpy as jnp


class VectorQuantizer(nn.Module):
    """Vector Quantizer that works with any input shape.

    Quantizes the last dimension of the input tensor to the nearest
    codebook entry. Works with any number of leading dimensions.
    """

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

        # codebook: K vectors of size embedding_dim
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
        """Quantize input tensor.

        Args:
            x: Input tensor of shape (..., embedding_dim)

        Returns:
            z_q: Quantized tensor, same shape as input
            loss: Commitment loss (scalar)
            indices: Codebook indices, shape (...)
        """
        # save original shape and flatten all but last dim
        original_shape = x.shape
        embedding_dim = x.shape[-1]
        assert (
            embedding_dim == self.embedding_dim
        ), f"Input dim {embedding_dim} != codebook dim {self.embedding_dim}"

        # flatten to (N, embedding_dim) where N = product of all leading dims
        x_flat = x.reshape(-1, self.embedding_dim)
        N = x_flat.shape[0]

        # compute squared distances using expanded formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        codebook = self.codebook.value  # (K, embedding_dim)

        x_sq = (x_flat**2).sum(axis=-1, keepdims=True)  # (N, 1)
        e_sq = (codebook**2).sum(axis=-1)  # (K,)
        xe = x_flat @ codebook.T  # (N, K)

        distances = x_sq + e_sq - 2 * xe  # (N, K)

        # get nearest codebook entry
        indices_flat = jnp.argmin(distances, axis=-1)  # (N,)

        # gather quantized vectors
        z_q_flat = codebook[indices_flat]  # (N, embedding_dim)

        # reshape back to original shape
        z_q = z_q_flat.reshape(original_shape)
        indices = indices_flat.reshape(original_shape[:-1])

        # compute commitment loss
        codebook_loss = jnp.mean((jax.lax.stop_gradient(x) - z_q) ** 2)
        commitment_loss = jnp.mean((x - jax.lax.stop_gradient(z_q)) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator
        z_q = x + jax.lax.stop_gradient(z_q - x)

        return z_q, loss, indices

    def lookup(self, indices: jax.Array) -> jax.Array:
        """Look up embeddings from codebook indices.

        Args:
            indices: Indices of shape (...)

        Returns:
            embeddings: Shape (..., embedding_dim)
        """
        return self.codebook.value[indices]
