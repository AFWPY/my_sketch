# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_channels):
        super(MultiHeadAttention, self).__init__()
        assert num_channels % num_heads == 0, "num_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_channel = num_channels // num_heads

        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)
        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

    def get_kqv_matrix(self, fm, linears):
        ret = linears(fm)
        return ret

    def reshape_and_transpose(self, component_matrix, B, H, W):
        m = self.num_heads
        component_matrix = torch.reshape(component_matrix, (B, H * W, m, self.d_channel))
        component_matrix = rearrange(component_matrix, 'b hw m d_channel -> (b m) hw d_channel')
        return component_matrix

    def forward(self, key, value, query):
        B, C, H, W = query.size()

        # Reshape query for multi-head attention
        query = rearrange(query, 'b c h w -> b (h w) c')
        key = rearrange(key, 'b c h w -> b (h w) c')
        value = rearrange(value, 'b c h w -> b (h w) c')

        # Compute query, key, value matrices
        query_matrix = self.get_kqv_matrix(query, self.linears_query)
        key_matrix = self.get_kqv_matrix(key, self.linears_key)
        value_matrix = self.get_kqv_matrix(value, self.linears_value)

        # Reshape and transpose for multi-head attention
        query_matrix = self.reshape_and_transpose(query_matrix, B, H, W)
        key_matrix = self.reshape_and_transpose(key_matrix, B, H, W)
        value_matrix = self.reshape_and_transpose(value_matrix, B, H, W)

        # Compute attention scores
        d_k = self.d_channel
        scores = torch.matmul(query_matrix, key_matrix.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        # Apply attention to value matrix
        out = torch.matmul(p_attn, value_matrix)
        out = rearrange(out, '(b m) hw c -> b hw (c m)', m=self.num_heads)
        residual = rearrange(query_matrix, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        # add & norm
        out = self.layer_norm(out + residual)  # b hw mc]
        out = self.multihead_concat_fc(out)  
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)

        return out


class ComponentAttentionModule(nn.Module):
    def __init__(self, num_heads=4, num_channels=512, num_layers=8):
        super(ComponentAttentionModule, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(num_heads, num_channels) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, input, q_cb):
        # Apply each multi-head attention layer in sequence
        for layer in self.layers:
            input = layer(input, input, q_cb)
        return input

# Example usage:
# component_attention_module = ComponentAttentionModule()
# output = component_attention_module(input_feature_map, query_cb_feature_map)