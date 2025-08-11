import sys

import torch
from diffusers.models.resnet import TemporalConvLayer
from torch import nn
from einops import rearrange


class FirstFeatureExtractor(nn.Module):
    def __init__(self):
        super(FirstFeatureExtractor, self).__init__()
        # self.transformer_temporal = TransformerTemporalModel(in_channels=64)
        self.conv_temporal = TemporalConvLayer(in_dim=128, out_dim=128)
        # self.transformer_spatio = BasicTransformerBlock(dim=64, num_attention_heads=8, attention_head_dim=16)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        b, num_frames, _, _, _ = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        identity = self.down_sample(x)
        h, w = identity.shape[-2:]
        x = self.conv_temporal(identity, num_frames=num_frames)
        # x = rearrange(x, "bf c h w -> bf (h w) c")
        # x = self.transformer_spatio(x)
        # x = rearrange(x, "bf (h w) c -> bf c h w", h=h, w=w)
        # x = self.transformer_temporal(x)
        x = x + identity
        return x


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = FirstFeatureExtractor()
    # model.to(device)
    # data = torch.zeros(1, 5, 3, 224, 224).to(device)
    # output = model(data)
    # print(output.shape)  # [5,128,56,56]
    query_embed = nn.Embedding(5, 256)
    print(query_embed.weight.shape)
