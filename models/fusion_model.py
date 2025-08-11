import collections
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange, repeat


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head=16, d_model=1024, d_k=64, d_v=64):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=128, embed_dim=1024, norm_layer=None)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        )
        self.cross_attention = MultiHeadAttention()
        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.GELU()
        )

    def forward(self, img, text):
        x_1 = self.down_sample(img)  # (b*f 128 h/4 w/4)
        x = self.patch_embed(x_1)  # (b*f 1024 h/16 w/16)
        _, _, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.cross_attention(x, text, text)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.up_sample(x)  # (b*f 128 h/4 w/4)
        x = x + x_1
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = FusionModel()
    model.to(device)
    img = torch.zeros(5, 3, 224, 224).to(device)
    text = torch.zeros(5, 77, 1024).to(device)
    output = model(img, text)
    print(output.shape)  # (5,128,56,56)
