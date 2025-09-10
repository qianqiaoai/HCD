import sys
import torch
from diffusers.models.resnet import TemporalConvLayer
from torch import nn
from einops import rearrange
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import inverse_sigmoid
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

class MSO(nn.Module):
    def __init__(self, mask_dim=16, img_dim=[96, 192], out_dim=16):
        super().__init__()

        self.mask_dim = mask_dim
        self.img_dim = img_dim
        self.out_dim = out_dim

        self.conv1_1div8 = nn.Conv2d(mask_dim+img_dim[1], mask_dim, kernel_size=3, padding=1)
        self.conv2_1div8 = nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1)

        self.conv1_1div4 = nn.Conv2d(mask_dim + img_dim[0], mask_dim, kernel_size=3, padding=1)
        self.conv2_1div4 = nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1)

    # TODO: add image on channel.  deconv to upsample
    def forward(self, pred_masks, image_features):
        image_features = [x.tensors for x in image_features]  # 1/4 & 1/8
        # merge with 1/8 image
        assert pred_masks.shape[-1] == image_features[-1].shape[-1], "First size wrong."
        x = torch.cat([pred_masks, image_features[-1]], dim=1)
        pred_masks += self.conv2_1div8(F.relu(self.conv1_1div8(F.relu(x))))

        # merge with 1/4 image
        pred_masks = F.interpolate(pred_masks, size=(image_features[-2].shape[-2], image_features[-2].shape[-1]), mode='bilinear', align_corners=False)
        assert pred_masks.shape[-1] == image_features[-2].shape[-1], "Second size wrong."
        x = torch.cat([pred_masks, image_features[-2]], dim=1)
        pred_masks += self.conv2_1div4(F.relu(self.conv1_1div4(F.relu(x))))

        return pred_masks

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = FirstFeatureExtractor()
    # model.to(device)
    # data = torch.zeros(1, 5, 3, 224, 224).to(device)
    # output = model(data)
    # print(output.shape)  # [5,128,56,56]
    query_embed = nn.Embedding(5, 256)
    print(query_embed.weight.shape)
