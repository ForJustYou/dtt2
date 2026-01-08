import torch
import torch.nn as nn

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = 'mlp'
        self.d_model = configs.d_model
        self.use_revin = True

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, cycle_index=None):
        # x_enc: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        if cycle_index is None:
            cycle_index = torch.zeros(x_enc.shape[0], device=x_enc.device, dtype=torch.long)
        else:
            cycle_index = cycle_index.long()

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x_enc, dim=1, keepdim=True)
            seq_var = torch.var(x_enc, dim=1, keepdim=True) + 1e-5
            x_enc = (x_enc - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x_enc = x_enc - self.cycleQueue(cycle_index, self.seq_len)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
