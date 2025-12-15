from torch import Tensor, nn, torch
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, d_model: int, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.d_model = d_model
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # The codebook: K vectors of size d_model
        self.codebook = nn.Embedding(num_embeddings, d_model)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x shape: (B, T, num_patches, d_model)
        B, T, num_patches, d_model = x.shape

        # Flatten to B * T * num_patches, d_model. Shape (M, d_model)
        x_flat = x.view(B * T * num_patches, d_model)

        # Compute squared distances using expanded formula
        codebook = self.codebook.weight  # (K, d_model)

        # Sum each of the N vectors (with dim=-1) -> results in shape (N, 1)
        # equal to
        x_sq = (x_flat**2).sum(dim=-1, keepdim=True)

        # Sum K codebook vectors -> result shape, (K,1)
        e_sq = (codebook**2).sum(dim=-1)

        # Compute the dot products x·e for all pairs -> result shape: (N, K)
        #   because the shape of x_flat is (M, d_model) & codebook is (K, d_model)
        #   we can transpose this so it's a matrix multiplication eg. (M, d_model) @ (d_model, K)
        #   this gives us a matrix of shape ()
        xe = x_flat @ codebook.T

        # Distances are the elementwise addition - 2 * xe
        distances = x_sq + e_sq - 2 * xe

        # Get the index with the smallest distance (closest)
        # shape (N, num_encodings) eg.
        encoding_indices = torch.argmin(distances, dim=-1)

        # Gather the closest codebook vectors
        z_q = self.codebook(encoding_indices)
        z_q = z_q.view(B, T, num_patches, d_model)

        # Compute the commitment loss
        loss = F.mse_loss(z_q, x.detach()) + self.commitment_cost * F.mse_loss(
            z_q.detach(), x
        )

        # Straight-through estimator trick for gradient propagation
        z_q = x + (z_q - x).detach()

        return z_q, loss, encoding_indices
