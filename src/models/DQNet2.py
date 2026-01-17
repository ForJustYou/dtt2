import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from src.layers.MLP import MLP
from src.utils.functions import num_patches
import numpy as np

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

    def forward(self, x, mask=None,Q=None,K=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0,2,1) # B, C, L

        q = self.proj_q(x) if Q is None else self.proj_q(Q.permute(0,2,1)) # B, C, L

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
        if K is not None:
            k = self.proj_k(K.permute(0,2,1)).reshape(B * self.n_heads, self.n_head_channels, L)
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
    def __init__(self, patch_len, d_route, n_heads, kernel=5, n_groups=4, d_model=None,
                 stride=1, seq_len=1) -> None:
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


    def forward(self, x,Q=None,K=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        #416 1 7 64
        q = self.proj_q(x) if Q is None else self.proj_q(Q.permute(0, 3, 1, 2))
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
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, H, W)
        if K is not None:
            k = self.proj_k(K.permute(0,3,1,2)).reshape(B * self.n_heads, H, W)
        v = self.proj_v(x_sampled)
        v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, H, W)
        q = q.reshape(B * self.n_heads, H, W) 
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b i j , b j d -> b i d', attention, v)
        
        return self.proj_out(out.reshape(B, H, W, C))


class DeformAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, droprate,
                 n_days=1, window_size=4, patch_len=7, stride=3) -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        # 1d size: B*n_days, subseq_len, C
        # 2d size: B*num_patches, 1, patch_len, C
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)
        self.layer_norm =  nn.LayerNorm(d_model)

        # 1D
        # Deform attention
        self.deform_attn = DeformAtten1D(self.subseq_len, d_model, n_heads, dropout, kernel=window_size) 
        self.attn_layers1d = nn.ModuleList([self.deform_attn])

        self.mlps1d = nn.ModuleList([ MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers1d))])
        self.drop_path1d = nn.ModuleList([ DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers1d))])
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
            stride=stride
        )
        self.write_out = nn.Linear(self.num_patches*self.patch_len, self.seq_len)

        self.attn_layers2d = nn.ModuleList([self.deform_attn2d])

        self.mlps2d = nn.ModuleList([ MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers2d)) ])
        self.drop_path2d = nn.ModuleList([DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers2d)) ])

        self.fc = nn.Linear(2*d_model, d_model)
    
    def forward(self, x, attn_mask=None, tau=None, delta=None,Q=None,K=None):
        
        def deform1d(x) -> torch.Tensor:
            B, L, C = x.shape
            x = self.layer_norm(x)
            padding_len = (self.n_days - (L % self.n_days)) % self.n_days
            x_padded = torch.cat((x, x[:, :padding_len, :].expand(-1, padding_len, -1)), dim=1)
            x_1d = rearrange(x_padded, 'b (seg_num ts_d) d_model -> (b ts_d) seg_num d_model', ts_d=self.n_days) 
            return x_1d
        n_day = self.n_days 
        B, L, C = x.shape
        x_1d = deform1d(x)
        Q_1d = deform1d(Q) if Q is not None else None
        K_1d = deform1d(K) if K is not None else None
        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d, Q=Q_1d, K=K_1d)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:,:L,:]

        def deform2d(x) -> torch.Tensor:
            x_unfold = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
            x_2d = rearrange(x_unfold, 'b n c l -> (b n) l c').unsqueeze(-3)
            x_2d = rearrange(x_2d, 'b c h w -> b h w c')
            return x_2d
        x_2d = deform2d(x)
        Q_2d = deform2d(Q) if Q is not None else None
        K_2d = deform2d(K) if K is not None else None
        for d, attn_layer in enumerate(self.attn_layers2d):
            x0 = x_2d
            x_2d = attn_layer(x_2d,Q_2d,K_2d)
            x_2d = self.drop_path2d[d](x_2d) + x0
            x0 = x_2d
            x_2d = self.mlps2d[d](self.layer_norm(x_2d.permute(0,1,3,2))).permute(0,1,3,2)
            x_2d = self.drop_path2d[d](x_2d) + x0
        x_2d = rearrange(x_2d, 'b h w c -> b c h w')
        x_2d = rearrange(x_2d, '(b n) 1 l c -> b (n l) c', b=B)
        x_2d = self.write_out(x_2d.permute(0,2,1)).permute(0,2,1)

        x = torch.concat([x_1d, x_2d], dim=-1)
        x = self.fc(x)
        return x_1d
    
class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel
        self.cycle_len = configs.cycle
        self.cycle_mode = configs.cycle_mode
        self.use_revin = True  # 是否使用RevIN，默认True 

         # 周期性时间查询参数：(cycle_len, enc_in)，按周期索引提取
        self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        
        #通道融合
        self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        # 输入映射到隐藏维度
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

         # Deformable Attention

        dpr = [x.item() for x in torch.linspace(self.dropout, self.dropout, self.e_layers)]
        self.deform_attn = DeformAttn(
            seq_len=configs.seq_len,
            d_model=self.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            droprate=dpr[1],
            n_days=configs.n_reshape,
            window_size=configs.kernel,
            patch_len=configs.patch_len,
            stride=configs.stride
        )

        self.deform_input = nn.Linear(self.enc_in, self.d_model)

        self.deform_output = nn.Linear(self.d_model, self.enc_in)

        # 简单的前馈模块（两层 GELU）
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
          # MLP layer
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        # 输出映射到预测长度
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, cycle_index=None) -> torch.Tensor:
        cycle_index = cycle_index.long()

        if self.use_revin:
            seq_mean = torch.mean(x_enc, dim=1, keepdim=True)
            seq_var = torch.var(x_enc, dim=1, keepdim=True) + 1e-5
            x_enc = (x_enc - seq_mean) / torch.sqrt(seq_var)
        
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
        # 通道融合
        x_enc = x_enc.permute(0, 2, 1)  # (b, c, s)
        channel = self.channelAggregator(query_input, x_enc, x_enc)[0]

        if self.cycle_mode == 'None':
            deform_out = self.deform_attn(self.deform_input(x_enc.permute(0, 2, 1)))
        elif self.cycle_mode == 'q':
            deform_out = self.deform_attn(self.deform_input(x_enc.permute(0, 2, 1)),
                                      self.deform_input(query_input.permute(0, 2, 1)))
        else:
            deform_out = self.deform_attn(self.deform_input(x_enc.permute(0, 2, 1)),
                                        self.deform_input(query_input.permute(0, 2, 1)),
                                      self.deform_input(query_input.permute(0, 2, 1)))
                                      
        deform_out = self.deform_output(deform_out).permute(0, 2, 1)
        x_enc = x_enc + channel + deform_out
        # 应用模型
        x_enc = self.input_proj(x_enc)
        hidden = self.model(x_enc)

        output = hidden + x_enc

        output = self.mlp(output).permute(0, 2, 1)
        # 可选：实例反归一化（RevIN）
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        return output
