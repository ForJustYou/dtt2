import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# ReVIN
# -----------------------------
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    x: (B, L, C)
    normalize over L dimension per (B, C).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self._cached_mean = None
        self._cached_std = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            # mean/std over time dimension L
            mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + self.eps)

            self._cached_mean = mean
            self._cached_std = std

            x_norm = (x - mean) / std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        elif mode == "denorm":
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError("RevIN: denorm called before norm.")
            x_den = x
            if self.affine:
                x_den = (x_den - self.beta) / (self.gamma + self.eps)
            x_den = x_den * self._cached_std + self._cached_mean
            return x_den
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode}.")


# -----------------------------
# Embeddings (PV-friendly)
# -----------------------------
class TokenEmbedding(nn.Module):
    """
    Conv1d token embedding (captures local ramps, cloud transients).
    Input: (B, L, C) -> Output: (B, L, d_model)
    """
    def __init__(self, c_in: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L) -> conv -> (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding.
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        # (1, L, d_model)
        return self.pe[:length].unsqueeze(0)


class PVEmbedding(nn.Module):
    """
    PV-friendly: TokenEmbedding (Conv) + PositionalEmbedding.
    """
    def __init__(
        self,
        c_in: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
        conv_kernel: int = 3,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.token = TokenEmbedding(c_in, d_model, kernel_size=conv_kernel)
        self.pos = PositionalEmbedding(d_model, max_len) if use_pos_emb else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        out = self.token(x)
        if self.pos is not None:
            out = out + self.pos(out.size(1))
        return self.drop(out)


# -----------------------------
# Phase Query (Q replaced by periodic phase matrix)
# -----------------------------
class PhaseQueryGenerator(nn.Module):
    """
    Generate Q from cycle phase, chosen by cycle_index.

    We generate a deterministic sinusoidal phase embedding:
        phase = 2π * (pos / period) + 2π * (cycle_index / period)
    then map [sin(phase), cos(phase), sin(2*phase), cos(2*phase), ...] into d_model.

    cycle_index: int tensor of shape (B,) or scalar int
    """
    def __init__(
        self,
        d_model: int,
        period: int,
        max_len: int,
        harmonics: int = 4,   # number of harmonics used
    ):
        super().__init__()
        self.d_model = d_model
        self.period = period
        self.max_len = max_len
        self.harmonics = harmonics

        # phase feature dim = 2 * harmonics
        feat_dim = 2 * harmonics
        self.proj = nn.Linear(feat_dim, d_model, bias=True)

        # pre-store positions [0..max_len-1]
        self.register_buffer("pos_idx", torch.arange(max_len, dtype=torch.float), persistent=False)

    def forward(self, cycle_index: torch.Tensor, length: int, start_pos: int = 0) -> torch.Tensor:
        """
        Returns Q: (B, length, d_model)
        """
        if length + start_pos > self.max_len:
            raise ValueError(f"length({length})+start_pos({start_pos}) exceeds max_len({self.max_len})")

        # cycle_index -> (B, 1)
        if not torch.is_tensor(cycle_index):
            cycle_index = torch.tensor(cycle_index, dtype=torch.long, device=self.pos_idx.device)

        if cycle_index.dim() == 0:
            cycle_index = cycle_index.view(1)
        if cycle_index.dim() == 1:
            cycle_index = cycle_index.view(-1, 1)  # (B, 1)
        else:
            raise ValueError("cycle_index must be scalar or shape (B,)")

        B = cycle_index.size(0)

        # positions: (1, L)
        pos = self.pos_idx[start_pos:start_pos + length].view(1, -1)  # (1, L)

        # phase: (B, L)
        # convert cycle_index to phase offset
        cycle_offset = (cycle_index.float() % self.period) / float(self.period)  # (B,1) in [0,1)
        phase = 2.0 * math.pi * (pos / float(self.period) + cycle_offset)  # (B, L)

        # build harmonic features: (B, L, 2*H)
        feats = []
        for k in range(1, self.harmonics + 1):
            feats.append(torch.sin(k * phase))
            feats.append(torch.cos(k * phase))
        feat = torch.stack(feats, dim=-1)  # (B, L, 2H)

        q = self.proj(feat)  # (B, L, d_model)
        return q


# -----------------------------
# 1D Deformable Attention (Self/Cross)
# -----------------------------
def _linear_interpolate_1d(values: torch.Tensor, idx_float: torch.Tensor) -> torch.Tensor:
    """
    values: (B, H, L, D)
    idx_float: (B, H, L, P) positions in [0, L-1] (float)
    returns: (B, H, L, P, D) interpolated
    """
    B, H, L, D = values.shape
    P = idx_float.size(-1)

    idx0 = torch.floor(idx_float).clamp(0, L - 1).long()  # (B,H,L,P)
    idx1 = (idx0 + 1).clamp(0, L - 1)                     # (B,H,L,P)
    w1 = (idx_float - idx0.float()).unsqueeze(-1)         # (B,H,L,P,1)
    w0 = 1.0 - w1                                         # (B,H,L,P,1)

    # gather along L dimension
    # expand to gather D
    gather0 = values.gather(dim=2, index=idx0.unsqueeze(-1).expand(B, H, L, P, D))
    gather1 = values.gather(dim=2, index=idx1.unsqueeze(-1).expand(B, H, L, P, D))

    out = w0 * gather0 + w1 * gather1
    return out


class DeformableAttention1D(nn.Module):
    """
    Deformable attention in 1D.
    - For self-attn: query_len = key_len = L
    - For cross-attn: query_len = Lq, key_len = Lk

    Q is external (here: phase query), K/V from x (embedding/memory).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_points = num_points

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)  # still project phase-Q for capacity

        # offsets predicted from Q: (B,Lq, H*P)
        self.offset_proj = nn.Linear(d_model, n_heads * num_points)

        # attention weight logits over points: (B,Lq,H,P)
        self.attn_w_proj = nn.Linear(d_model, n_heads * num_points)

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        """
        q_in:  (B, Lq, d_model) phase query
        kv_in: (B, Lk, d_model) source (self: same as encoder states; cross: memory)

        return: (B, Lq, d_model)
        """
        B, Lq, _ = q_in.shape
        _, Lk, _ = kv_in.shape
        H, D = self.n_heads, self.head_dim
        P = self.num_points

        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        # reshape to heads
        q = q.view(B, Lq, H, D).transpose(1, 2)  # (B,H,Lq,D)
        k = k.view(B, Lk, H, D).transpose(1, 2)  # (B,H,Lk,D)
        v = v.view(B, Lk, H, D).transpose(1, 2)  # (B,H,Lk,D)

        # offsets and point weights from original q_in (not head-split)
        offsets = self.offset_proj(q_in).view(B, Lq, H, P).permute(0, 2, 1, 3)  # (B,H,Lq,P)
        attn_logits = self.attn_w_proj(q_in).view(B, Lq, H, P).permute(0, 2, 1, 3)  # (B,H,Lq,P)

        # base index: align query positions to key positions
        # if Lq != Lk, map query index linearly into key index range
        if Lq == Lk:
            base = torch.arange(Lq, device=q_in.device, dtype=torch.float).view(1, 1, Lq, 1)  # (1,1,Lq,1)
        else:
            base = torch.linspace(0, Lk - 1, steps=Lq, device=q_in.device, dtype=torch.float).view(1, 1, Lq, 1)

        # apply offsets (can be negative/positive), then clamp
        idx = (base + offsets).clamp(0.0, float(Lk - 1))  # (B,H,Lq,P)

        # sample K and V at idx
        k_s = _linear_interpolate_1d(k, idx)  # (B,H,Lq,P,D)
        v_s = _linear_interpolate_1d(v, idx)  # (B,H,Lq,P,D)

        # point-wise attention: dot(q, k_s) + learned point logits
        # q: (B,H,Lq,D) -> (B,H,Lq,1,D)
        q_exp = q.unsqueeze(3)
        dot = (q_exp * k_s).sum(dim=-1) / math.sqrt(D)  # (B,H,Lq,P)
        scores = dot + attn_logits
        weights = F.softmax(scores, dim=-1)  # (B,H,Lq,P)
        weights = self.drop(weights)

        # weighted sum of v_s
        out = (weights.unsqueeze(-1) * v_s).sum(dim=3)  # (B,H,Lq,D)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, Lq, H * D)  # (B,Lq,d_model)
        out = self.out_proj(out)
        return self.drop(out)


# -----------------------------
# Encoder / Decoder Blocks
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        if activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DeformableEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = DeformableAttention1D(d_model, n_heads, num_points=num_points, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_phase: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d), q_phase: (B,L,d)
        a = self.attn(q_phase, x)
        x = self.norm1(x + self.drop(a))
        f = self.ffn(x)
        x = self.norm2(x + self.drop(f))
        return x


class DeformableDecoderLayer(nn.Module):
    """
    Future phase queries do deformable cross-attn to encoder memory.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross = DeformableAttention1D(d_model, n_heads, num_points=num_points, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, mem: torch.Tensor, q_phase_future: torch.Tensor) -> torch.Tensor:
        # y: (B,Lp,d) current decoder state (initialized as q_phase_future or learnable)
        # mem: (B,L,d)
        a = self.cross(q_phase_future, mem)
        y = self.norm1(y + self.drop(a))
        f = self.ffn(y)
        y = self.norm2(y + self.drop(f))
        return y


# -----------------------------
# Full PV Forecasting Model
# -----------------------------
class Model(nn.Module):
    """
    Input:
        x_enc: (B, L, C)
        cycle_index: scalar or (B,)
    Output:
        y_pred: (B, pred_len, out_dim)
    """
    def __init__(
        self,configs,
        num_points: int = 4,  # deformable attn sampling points nums
        use_pos_emb: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        harmonics: int = 4,   # phase Q harmonics
        use_learnable_future_base: bool = True,  # decoder 额外可学习基底，提高表达
    ):
        super().__init__()
        self.c_in = configs.enc_in
        self.out_dim = configs.c_out
        self.max_len = configs.seq_len + configs.pred_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.period = configs.period
        self.d_layers = configs.d_layers
        self.e_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_model * 4
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel
        # ReVIN on raw input channels
        self.revin = RevIN(num_features=self.c_in, eps=revin_eps, affine=revin_affine)

        # Embedding
        self.embed = PVEmbedding(
            c_in=self.c_in,
            d_model=self.d_model,
            max_len=self.max_len,
            dropout=self.dropout,
            conv_kernel=self.kernel_size,
            use_pos_emb=use_pos_emb,
        )

        # Phase Q generators (encoder / decoder)
        self.phase_q = PhaseQueryGenerator(d_model=self.d_model, period=self.period, max_len=self.max_len, harmonics=harmonics)

        # Encoder
        self.encoder = nn.ModuleList([
            DeformableEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                num_points=num_points,
                dropout=self.dropout,
            )
            for _ in range(self.e_layers)
        ])
        self.enc_norm = nn.LayerNorm(self.d_model)

        # Decoder
        self.decoder = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                num_points=num_points,
                dropout=self.dropout,
            )
            for _ in range(self.d_layers)
        ])
        self.dec_norm = nn.LayerNorm(self.d_model)

        # Optional learnable base for future tokens
        self.use_learnable_future_base = use_learnable_future_base
        if use_learnable_future_base:
            self.future_base = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))
            nn.init.trunc_normal_(self.future_base, std=0.02)
        else:
            self.register_parameter("future_base", None)

        # Output head
        self.proj = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x_enc: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (B, L, C)
        cycle_index: scalar or (B,)
        """
        B, L, C = x_enc.shape
        if L + self.pred_len > self.max_len:
            raise ValueError(f"L({L}) + pred_len({self.pred_len}) > max_len({self.max_len})")

        # 1) ReVIN normalize on raw channels
        x_norm = self.revin(x_enc, mode="norm")  # (B,L,C)

        # 2) Embedding
        x = self.embed(x_norm)  # (B,L,d)

        # 3) Encoder phase Q (same length L, start_pos=0)
        q_phase_enc = self.phase_q(cycle_index=cycle_index, length=L, start_pos=0)  # (B,L,d)

        # 4) Encoder: deformable self-attn with phase-Q
        for layer in self.encoder:
            x = layer(x, q_phase_enc)
        mem = self.enc_norm(x)  # (B,L,d)

        # 5) Decoder future phase Q (length pred_len, start_pos=L)
        q_phase_dec = self.phase_q(cycle_index=cycle_index, length=self.pred_len, start_pos=L)  # (B,Lp,d)

        # init decoder state
        if self.use_learnable_future_base:
            y = self.future_base.expand(B, -1, -1) + q_phase_dec
        else:
            y = q_phase_dec

        # 6) Decoder: deformable cross-attn (future phase queries -> encoder mem)
        for layer in self.decoder:
            y = layer(y, mem, q_phase_dec)
        y = self.dec_norm(y)  # (B,Lp,d)

        # 7) Project to target
        out = self.proj(y)  # (B, pred_len, out_dim)

        # 8) ReVIN denorm:
        #    ReVIN是对输入C维做的；如果 out_dim == C，可直接反归一化；
        #    若 out_dim != C（例如只预测1维功率），这里默认你把目标功率放在输入的某一列，
        #    建议：训练时让 out_dim == 1，并把目标列单独做ReVIN或把C=1输入模型。
        if out.shape[-1] == C:
            out = self.revin(out, mode="denorm")
        return out
