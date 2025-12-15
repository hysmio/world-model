from torch import Tensor, nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor) -> Tensor:
        attn_output, _ = self.self_attn(src, src, src)
        src = self.layer_norm1(src + attn_output)
        ff_output = self.feed_forward(src)
        src = self.layer_norm2(src + ff_output)
        return src


class Encoder(nn.Module):
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor) -> Tensor:
        for layer in self.layers:
            src = layer(src)
        return src


class STTransformer(nn.Module):
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, num_patches, d_model)
        B, T, num_patches, d_model = x.shape

        # Flatten spatiotemporal: (B, T * num_patches, d_model)
        x = x.view(B, T * num_patches, d_model)

        x = self.encoder(x)

        # Reshape back: (B, T, num_patches, d_model)
        return x.view(B, T, num_patches, d_model)
