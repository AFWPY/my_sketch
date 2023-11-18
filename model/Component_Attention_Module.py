# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ComponentAttentiomModule(nn.Module):
    def __init__(self, num_heads=8, num_channels=256):
        super(ComponentAttentiomModule, self).__init__()
        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)
        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

    def get_kqv_matrix(self, fm, linears):
        # matmul with style featuremaps and content featuremaps
        ret = linears(fm)
        return ret

    def get_content_query(self, content_feature_map):
        B, C, H, W = content_feature_map.shape
        m = self.num_heads
        d_channel = C // m
        query_component_matrix = rearrange(content_feature_map, 'b c h w -> b (h w) c')
        query_component_matrix = self.get_kqv_matrix(query_component_matrix, self.linears_query)
        query_component_matrix = torch.reshape(query_component_matrix, (B, H * W, m, d_channel))
        query_component_matrix = rearrange(query_component_matrix, 'b hw m d_channel -> (b m) hw d_channel')

        return query_component_matrix

    def get_component_key_value(self, input, keys=False):
        B, C, H, W = input.shape
        m = self.num_heads
        d_channel = C // m

        if keys:
            key_component_matrix = rearrange(input, 'b c h w -> b (h w) c')
            key_component_matrix = self.get_kqv_matrix(key_component_matrix, self.linears_query)
            key_component_matrix = torch.reshape(key_component_matrix, (B, H * W, m, d_channel))
            key_component_matrix = rearrange(key_component_matrix, 'b hw m d_channel -> (b m) hw d_channel')
            return key_component_matrix
        else:
            value_component_matrix = rearrange(input, 'b c h w -> b (h w) c')
            value_component_matrix = self.get_kqv_matrix(value_component_matrix, self.linears_value)
            value_component_matrix = torch.reshape(value_component_matrix, (B, H * W, m, d_channel))
            value_component_matrix = rearrange(value_component_matrix, 'b hw m d_channel -> (b m) hw d_channel')
            return value_component_matrix

    def cross_attention(self, input ,q_cb , mask=None, dropout=None):
        # get query key value
        B, C, H, W = input.shape
        content_query = self.get_content_query(q_cb)  # [(b m) (h w) c]
        components_key = self.get_component_key_value(input, keys=True)  # [(b m) (h w) c]
        style_components_value = self.get_component_key_value(input)  # [(b m) (h w) c]

        # q=k=v.shape [(b m) n d_channel]
        residual = content_query
        d_k = content_query.size(-1)
        scores = torch.matmul(content_query, components_key.transpose(-2, -1)) / math.sqrt(d_k)  # [(b m) (h w) n]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        out = torch.matmul(p_attn, style_components_value)
        out = rearrange(out, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        residual = rearrange(residual, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        # add & norm
        out = self.layer_norm(out + residual)  # b hw mc]
        out = self.multihead_concat_fc(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)

        return out  # [B C H W]

    def forward(self, input,q_cb):
        output = self.cross_attention(input,q_cb)

        return output

