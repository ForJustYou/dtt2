import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 基础配置（序列长度、预测长度、特征维度等）
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        # 消融开关：是否使用时间查询与通道聚合
        self.use_revin = True  # 是否使用RevIN，默认True
        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True

        # 周期性时间查询参数：(cycle_len, enc_in)，按周期索引提取
        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        # 通道聚合：以序列长度为注意力维度，对通道维做注意力融合
        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        # 输入映射到隐藏维度
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        # 简单的前馈模块（两层 GELU）
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # 输出映射到预测长度
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, cycle_index=None):
        cycle_index = cycle_index.long()
        x = x_enc  # (b, s, c)
        # 可选：实例归一化（RevIN）
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 维度转换：b,s,c -> b,c,s 方便后续按通道处理
        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            # 根据周期索引循环取出时间查询向量，并与输入在序列维对齐
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
            if self.channel_aggre:
                # 用注意力在通道维整合 query 和输入，捕捉周期模式
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                # 仅通道聚合，不加时间查询
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        # 输入投影：融合后的特征映射到 d_model
        input = self.input_proj(x_input+channel_information)

        # 前馈变换
        hidden = self.model(input)

        # 残差输出并恢复到 b,s,c
        output = self.output_proj(hidden+input).permute(0, 2, 1)

        # 可选：实例反归一化（RevIN）
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output

