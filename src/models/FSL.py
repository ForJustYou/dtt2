import torch.nn as nn
from src.models.DeformTime import Model as DeformTime

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.encoder = DeformTime(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
