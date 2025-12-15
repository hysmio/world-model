import torch
from torch import Tensor, nn


# Vision Tokenizer Encoder, turns pixels into tokens
# Currently just uses very basic patching, Gemma paper mentioned
# patching explicitly, but also said based on ViT which uses Conv2d
class VTEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        d_model: int,
        max_frames: int,
        num_patches: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        # Patch dimension is the number of channels * the pixels in the patch
        self.patch_dim = in_channels * patch_size * patch_size
        self.embedding = nn.Linear(self.patch_dim, d_model)

        # learned positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_patches, d_model))
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model))

    # turns x (B, T, C, H, W) into (B, T, num_patches, d_model)
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # if patch_size = 4, H = 8, W = 8, then it should be 4 patches, 2 along each axis
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # reshape to expose patch grid
        x = x.view(
            B,
            T,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        # shape (2, 4, 3, 2, 4, 2, 4)

        # permute this tensor so that the patches are effectively a 2d array, containing the C, patch_size, patch_size, eg. 3 * 8 * 8 = 192 values of each
        # patches pixels, this aligns these values in memory so they'll be 192 values in a row, corresponding to the patch, the next 192, will be the next patch
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        # shape (B, T, num_h, num_w, C, p, p)

        # flatten the spatial grid and patch_dimensions
        x = x.reshape(B, T, num_patches, self.patch_dim)

        return self.embedding(x) + self.spatial_pos + self.temporal_pos
