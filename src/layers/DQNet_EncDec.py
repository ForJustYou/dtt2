import numpy as np
from timm.models.layers import trunc_normal_
from src.utils.functions import num_patches
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
import math


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, cycle_index=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, cycle_index=cycle_index)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
class MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class DeformAtten1D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.seq_len = seq_len
        self.d_model = d_model 
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, stride=self.stride, padding=pad_size),
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.d_model, self.seq_len))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0,2,1) # B, C, L

        q = self.proj_q(x) # B, C, L

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.n_groups)

        grouped_queries = group(q)

        offset = self.proj_offset(grouped_queries) # B * g 1 Lg
        offset = rearrange(offset, 'b 1 n -> b n')

        def grid_sample_1d(feats, grid, *args, **kwargs):
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value = 0.)
            feats = rearrange(feats, '... -> ... 1')
            out = F.grid_sample(feats, grid, **kwargs) 
            return rearrange(out, '... 1 -> ...')
        
        def normalize_grid(arange, dim = 1, out_dim = -1):
            n = arange.shape[-1]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0:
            offset = offset.tanh().mul(self.offset_range_factor)

        grid = torch.arange(offset.shape[-1], device = device)
        vgrid = grid + offset
        vgrid_scaled = normalize_grid(vgrid)
        x_sampled = grid_sample_1d(group(x),vgrid_scaled,
            mode = 'bilinear', padding_mode = 'zeros', align_corners = False)[:,:,:L]

        x_sampled = rearrange(x_sampled,'(b g) d n -> b (g d) n', g = self.n_groups)
        q = q.reshape(B * self.n_heads, self.n_head_channels, L)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)
        v = self.proj_v(x_sampled)
        v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, self.n_head_channels, L)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1) # softmax: attention[0,0,:].sum() = 1

        out = torch.einsum('b i j , b j d -> b i d', attention, v) 
        
        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B))


class DeformAtten2D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, patch_len, d_route, n_heads, kernel=5, n_groups=4, d_model=None, stride=1, seq_len=1, cycle=168, cycle_mode="q") -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.d_model = d_model
        self.seq_len = seq_len
        self.stride = stride
        self.patch_len = patch_len
        self.d_route = d_route #1
        self.n_groups = n_groups
        self.n_group_channels = self.d_route // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_route // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.cycle = cycle
        self.cycle_mode = cycle_mode
        self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle, self.d_model), requires_grad=True)

        self.proj_q = nn.Conv2d(self.d_route, self.d_route, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.d_route, self.d_route, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.d_route, self.d_route, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_route, self.d_route)
        kernel_size = kernel
        self.step = 1
        pad_size = kernel_size // 2 if kernel_size != self.step else 0
        self.proj_offset = nn.Sequential( 
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.step, padding=pad_size),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.scale_factor = self.d_route ** -0.5  # 1/np.sqrt(dim)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.d_route, self.patch_len, 1))
        trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x, cycle_index):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        #416 1 7 64
        cycle_index = cycle_index.long()
        gather_index = cycle_index % self.cycle
        query_input = self.temporalQuery[gather_index]  # (b, c, s)
        query = query_input.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        query = rearrange(query, 'b n c l -> (b n) l c').unsqueeze(-3)
        if self.cycle_mode == "q" or self.cycle_mode == "qk":
            q = self.proj_q(query) # B, 1, H, W
        elif self.cycle_mode == "None":
            q = self.proj_q(x) # B, C, H, W
        else:
            raise ValueError("Unknown cycle_mode {}".format(self.cycle_mode))
        offset = self.proj_offset(q) # B, 2, H, W

        if self.offset_range_factor >= 0:
            offset = offset.tanh().mul(self.offset_range_factor)

        def create_grid_like(t, dim = 0):
            h, w, device = *t.shape[-2:], t.device

            grid = torch.stack(torch.meshgrid(
                torch.arange(w, device = device),
                torch.arange(h, device = device),
            indexing = 'xy'), dim = dim)

            grid.requires_grad = False
            grid = grid.type_as(t)
            return grid
        
        def normalize_grid(grid, dim = 1, out_dim = -1):
            # normalizes a grid to range from -1 to 1
            h, w = grid.shape[-2:]
            grid_h, grid_w = grid.unbind(dim = dim)

            grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
            grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

            return torch.stack((grid_h, grid_w), dim = out_dim)

     
        grid =create_grid_like(offset)
        vgrid = grid + offset
        vgrid_scaled = normalize_grid(vgrid)
        x_sampled = F.grid_sample(x,vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)[:,:,:H,:W]
        x_sampled = rearrange(x_sampled, '(b g) c h w -> b (g c) h w', g=self.n_groups)
        if self.cycle_mode == "qk":
            k = self.proj_k(query).reshape(B * self.n_heads, H, W)
        else:
            k = self.proj_k(x_sampled).reshape(B * self.n_heads, H, W)
        v = self.proj_v(x_sampled)
        v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, H, W)
        q = q.reshape(B * self.n_heads, H, W)
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b i j , b j d -> b i d', attention, v)
        
        return self.proj_out(out.reshape(B, H, W, C))


class CrossDeformAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, droprate,
                 n_days=1, window_size=4, patch_len=7, stride=3, cycle=168, cycle_mode="q") -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        # 1d size: B*n_days, subseq_len, C
        # 2d size: B*num_patches, 1, patch_len, C
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)
        self.cycle = cycle
        self.cycle_mode = cycle_mode
        self.layer_norm =  nn.LayerNorm(d_model)

        # 1D
        # Deform attention
        self.deform_attn = DeformAtten1D(self.subseq_len, d_model, n_heads, dropout, kernel=window_size) 
        self.attn_layers1d = nn.ModuleList([self.deform_attn])

        self.mlps1d = nn.ModuleList(
            [ 
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers1d))
            ]
        )
        self.drop_path1d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers1d))
            ]
        )
        #######################################
        # 2D
        d_route = 1
        self.deform_attn2d = DeformAtten2D(
            self.patch_len,
            d_route,
            n_heads=1,
            kernel=window_size,
            n_groups=1,
            d_model=d_model,
            seq_len=seq_len,
            stride=stride,
            cycle=cycle,
            cycle_mode=cycle_mode,
        )
        self.write_out = nn.Linear(self.num_patches*self.patch_len, self.seq_len)

        self.attn_layers2d = nn.ModuleList([self.deform_attn2d])

        self.mlps2d = nn.ModuleList(
            [ 
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers2d))
            ]
        )
        self.drop_path2d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers2d))
            ]
        )

        self.fc = nn.Linear(2*d_model, d_model)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None, cycle_index=None):
        n_day = self.n_days 
        B, L, C = x.shape
        
        assert cycle_index.dim() == 2
        cycle_index_patch = cycle_index.unfold(dimension=1, size=self.patch_len, step=self.stride)
        cycle_index_flat = cycle_index_patch.reshape(-1, self.patch_len)

        x = self.layer_norm(x)
        padding_len = (n_day - (L % n_day)) % n_day
        x_padded = torch.cat((x, x[:, [0], :].expand(-1, padding_len, -1)), dim=1)
        x_1d = rearrange(x_padded, 'b (seg_num ts_d) d_model -> (b ts_d) seg_num d_model', ts_d=n_day) 
        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:,:L,:]

        x_unfold = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        x_2d = rearrange(x_unfold, 'b n c l -> (b n) l c').unsqueeze(-3)
        x_2d = rearrange(x_2d, 'b c h w -> b h w c')

        for d, attn_layer in enumerate(self.attn_layers2d):
            x0 = x_2d
            x_2d = attn_layer(x_2d,cycle_index_flat)
            x_2d = self.drop_path2d[d](x_2d) + x0
            x0 = x_2d
            x_2d = self.mlps2d[d](self.layer_norm(x_2d.permute(0,1,3,2))).permute(0,1,3,2)
            x_2d = self.drop_path2d[d](x_2d) + x0
        x_2d = rearrange(x_2d, 'b h w c -> b c h w')
        x_2d = rearrange(x_2d, '(b n) 1 l c -> b (n l) c', b=B)
        x_2d = self.write_out(x_2d.permute(0,2,1)).permute(0,2,1)

        x = torch.concat([x_1d, x_2d], dim=-1)
        x = self.fc(x)
        return x, None
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Local_Temporal_Embedding(nn.Module):
    def __init__(self, d_inp, d_model, padding, sub_groups=8, dropout=0.1):
        super(Local_Temporal_Embedding, self).__init__()

        d_out = d_model // sub_groups if d_model % sub_groups == 0 else d_model // sub_groups + 1
        self.sub_seqlen = d_inp
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(d_inp, d_out, bias=False)
                
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        B, L, C = x.shape
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.sub_seqlen, step=self.sub_seqlen)
        x = rearrange(x, 'b l g c -> (b g) l c')
        x = self.value_embedding(x)
        x = rearrange(x, '(b g) l c -> b l (g c)', b = B)[:,:,:self.d_model]
        x = x + self.position_embedding(x)
        return self.dropout(x)


class CoarseToFineDecoder(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, coarse_factor=4, tcn_kernel=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.coarse_factor = max(1, int(coarse_factor))
        self.coarse_len = max(1, math.ceil(pred_len / self.coarse_factor))

        self.coarse_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, self.coarse_len),
        )

        padding = tcn_kernel // 2
        self.refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=tcn_kernel, padding=padding),
            nn.LeakyReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=tcn_kernel, padding=padding),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        coarse = self.coarse_proj(x)
        if self.coarse_len == self.pred_len:
            up = coarse
        else:
            up = F.interpolate(coarse, size=self.pred_len, mode='linear', align_corners=False)
        refined = up + self.refine(up)
        return refined.permute(0, 2, 1)
