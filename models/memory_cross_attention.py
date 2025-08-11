import torch
from einops import rearrange
from torch import nn
from .position_encoding import PositionEmbeddingSine2D
from util.misc import NestedTensor


class memory_cross_attention(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(memory_cross_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.act1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1024_256 = nn.Linear(1024, d_model)

    def forward(self, encoder_hidden_states_c, query_tokens):
        """
        :param encoder_hidden_states_c: (bt, l, c) c=1024
        :param query_tokens: (q, b, c)
        """
        encoder_hidden_states_c = self.linear1024_256(encoder_hidden_states_c)
        bt, l, c = encoder_hidden_states_c.shape
        q, b, _ = query_tokens.shape
        t = bt // b
        query_tokens = query_tokens.repeat_interleave(t, dim=1).permute(1, 0, 2)  # [bt q c]
        query2, _ = self.cross_attention(query_tokens, encoder_hidden_states_c, encoder_hidden_states_c)
        query_tokens = query_tokens + self.dropout1(query2)
        query_tokens = self.norm1(query_tokens)
        query2 = self.linear2(self.dropout2(self.act1(self.linear1(query_tokens))))
        query_tokens = query_tokens + self.dropout3(query2)
        query_tokens = self.norm2(query_tokens)
        return query_tokens
