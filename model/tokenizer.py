import torch
from torch import Tensor, nn

from model.transformer import STTransformer
from model.vq import VectorQuantizer


# Vision Tokenizer Encoder, turns pixels into tokens
# Currently just uses very basic patching, Gemma paper mentioned
# patching explicitly, but also said based on ViT which uses Conv2d
class VTEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,

        # Patch size
        patch_size: int,
        num_patches: int,
        max_frames: int,

        # ST Transformer
        st_blocks: int,
        num_heads: int,
        # key / query size
        kq_size: int,
        d_ff: int,
        dropout: float,

        # Vector Quantizer
        num_codes: int,
        latent_dim: int
    ):
        super().__init__()
        self.patch_size = patch_size # ty: ignore[unresolved-attribute]
        # Patch dimension is the number of channels * the pixels in the patch
        # assume rgb
        self.patch_dim = 3 * patch_size * patch_size # ty: ignore[unresolved-attribute]

        self.patch_proj = nn.Linear(self.patch_dim, d_model)

        # project from encoder d_model to latent_dim for VQ
        self.down_proj = nn.Linear(d_model, latent_dim)

        # codebook only handles quantization, no d_model projections
        self.vq = VectorQuantizer(num_codes, latent_dim)

        # learned positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, num_patches, d_model))
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model))

        self.st_transformer = STTransformer(st_blocks, d_model, num_heads, kq_size, d_ff, dropout)

    # turns x (B, T, C, H, W) into (B, T, num_patches, d_model)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

        # project the patches from (B, T, num_patches, patch_dim) to (B, T, num_patches, d_model)
        x = self.patch_proj(x)

        # add spatial & temporal embeddings
        x = x + self.spatial_pos + self.temporal_pos

        # add transformer blocks
        x = self.st_transformer(x)

        # project down to latent dimension before VQ
        x = self.down_proj(x)

        # convert into discrete token codebook
        x, loss, indices = self.vq(x)

        return x, loss, indices
