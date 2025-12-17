import torch
from torch import Tensor, nn
from torch.nn import functional as F

from einops import rearrange


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class SpatialLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kq_size: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads # ty: ignore[unresolved-attribute]
        self.kq_size = kq_size # ty: ignore[unresolved-attribute]
        self.scale = kq_size ** -0.5

        # q, k, v projections with kq_size per head
        self.q_proj = nn.Linear(d_model, num_heads * kq_size)
        self.k_proj = nn.Linear(d_model, num_heads * kq_size)
        self.v_proj = nn.Linear(d_model, num_heads * kq_size)
        self.out_proj = nn.Linear(num_heads * kq_size, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, E = x.shape

        # reshape for spatial attention, patches attend to each other within each frame
        x_reshaped = rearrange(x, 'b t p e -> (b t) p e')  # (B*T, P, E)

        # project to q, k, v
        q = self.q_proj(x_reshaped)  # (B*T, P, num_heads * kq_size)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # reshape to separate heads: (B*T, H, P, kq_size)
        q = rearrange(q, 'b p (h d) -> b h p d', h=self.num_heads)
        k = rearrange(k, 'b p (h d) -> b h p d', h=self.num_heads)
        v = rearrange(v, 'b p (h d) -> b h p d', h=self.num_heads)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, H, P, P)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # apply attention to values
        out = attn @ v  # (B*T, H, P, kq_size)

        # concat heads and project back to d_model
        out = rearrange(out, 'b h p d -> b p (h d)')  # (B*T, P, num_heads * kq_size)
        out = self.out_proj(out)  # (B*T, P, d_model)

        # reshape back
        out = rearrange(out, '(b t) p e -> b t p e', b=B, t=T)

        # residual + norm
        x = x + self.dropout(out)
        x = self.layer_norm(x)

        return x


class TemporalLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kq_size: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads # ty: ignore[unresolved-attribute]
        self.kq_size = kq_size # ty: ignore[unresolved-attribute]
        self.scale = kq_size ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads * kq_size)
        self.k_proj = nn.Linear(d_model, num_heads * kq_size)
        self.v_proj = nn.Linear(d_model, num_heads * kq_size)
        self.out_proj = nn.Linear(num_heads * kq_size, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, E = x.shape

        # reshape for temporal attention, frames attend to each other for each patch
        x_reshaped = rearrange(x, 'b t p e -> (b p) t e')  # (B*P, T, E)

        # project to q, k, v
        q = self.q_proj(x_reshaped)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # reshape to separate heads: (B*P, H, T, kq_size)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*P, H, T, T)

        # causal mask, frame t can only attend to frames 0..t
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1
        )
        attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # apply attention to values
        out = attn @ v  # (B*P, H, T, kq_size)

        # concat heads and project back
        out = rearrange(out, 'b h t d -> b t (h d)')  # (B*P, T, num_heads * kq_size)
        out = self.out_proj(out)  # (B*P, T, d_model)

        # reshape back
        out = rearrange(out, '(b p) t e -> b t p e', b=B, p=P)

        # residual + norm
        x = x + self.dropout(out)
        x = self.layer_norm(x)

        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, kq_size: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.spatial_layer = SpatialLayer(d_model, num_heads, kq_size, dropout)
        self.temporal_layer = TemporalLayer(d_model, num_heads, kq_size, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_layer(x)
        x = self.temporal_layer(x)
        ff_output = self.feed_forward(x)

        x = x + self.dropout(ff_output)
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
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads, kq_size, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)

        return x
