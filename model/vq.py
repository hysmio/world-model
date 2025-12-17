import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, latent_dim: int, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.latent_dim = latent_dim # ty: ignore[unresolved-attribute]
        self.num_embeddings = num_embeddings # ty: ignore[unresolved-attribute]
        self.commitment_cost = commitment_cost # ty: ignore[unresolved-attribute]

        # codebook: K vectors of size latent_dim, shared between encoder and decoder
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x shape: (B, T, num_patches, latent_dim)
        B, T, num_patches, latent_dim = x.shape
        assert latent_dim == self.latent_dim, f"Input dim {latent_dim} != codebook dim {self.latent_dim}"

        # flatten to B * T * num_patches, latent_dim
        z_e_flat = x.view(B * T * num_patches, self.latent_dim)  # (M, latent_dim)

        # compute squared distances using expanded formula
        codebook = self.codebook.weight  # (K, latent_dim)

        # sum each of the M vectors (with dim=-1) -> results in shape (M, 1)
        z_e_sq = (z_e_flat**2).sum(dim=-1, keepdim=True)

        # sum K codebook vectors -> result shape, (K)
        e_sq = (codebook**2).sum(dim=-1)

        # compute the dot products z_e·e for all pairs -> result shape: (M, K)
        #   because the shape of z_e_flat is (M, latent_dim) & codebook is (K, latent_dim)
        #   we can transpose this so it's a matrix multiplication eg. (M, latent_dim) @ (latent_dim, K)
        ze = z_e_flat @ codebook.T # (M, K)

        # distances are the elementwise addition - 2 * ze
        distances = z_e_sq + e_sq - 2 * ze # (M, K)

        # get the index with the smallest distance (closest)
        encoding_indices = torch.argmin(distances, dim=-1) # (M)

        # gather the closest codebook vectors
        z_q = self.codebook(encoding_indices)  # (M, latent_dim)
        z_q = z_q.view(B, T, num_patches, self.latent_dim)  # (B, T, num_patches, latent_dim)

        # compute the commitment loss
        loss = F.mse_loss(z_q, x.detach()) + self.commitment_cost * F.mse_loss(
            z_q.detach(), x
        )

        # straight-through estimator trick for gradient propagation
        z_q = x + (z_q - x).detach()

        return z_q, loss, encoding_indices
