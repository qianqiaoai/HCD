"""
Obtaining features using Unet
Modified from DETR, SgMg and VD-IT (https://github.com/facebookresearch/detr, https://github.com/bo-miao/SgMg, https://github.com/buxiangzhiren/VD-IT) 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .unet_3d_condition_it import UNet3DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from transformers import AutoProcessor,CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
from pathlib import Path
from .temporal_mask_refinement import FirstFeatureExtractor
from util.misc import NestedTensor
from typing import Dict, List
from models.position_encoding import build_position_encoding

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


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in=1024, d_hid=512):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x += residual
        x = self.layer_norm(x)
        return x


class ResidualMLP(nn.Module):
    def __init__(self):
        super(ResidualMLP, self).__init__()
        self.linear1 = nn.Linear(1280, 1024)
        self.linear2 = nn.Linear(1024, 1024)

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(out1)

        return out2 + out1


class Crossattn(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, ):
        super().__init__()
        self.cros_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, img1, img2):
        dec_output = self.cros_attn(img1, img2, img2)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class FeatureExtractor(torch.nn.Module):
    """
    FeatureExtractor for FPN
    """

    def __init__(
            self,
            pretrained_model_path='./weight',
            num_channels=[128, 256, 512, 1024]
    ):
        super().__init__()
        print(pretrained_model_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        self.dim_tokens_enc = [512, 960, 1600, 1920]
        self.layer_dims = num_channels
        self.unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        self.cross_attn = Crossattn()
        self.first_feature = FirstFeatureExtractor()

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1] * 2,
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[1] * 2,
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2] * 2,
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[2] * 2,
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_postprocess = nn.ModuleList([
            # self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

        self.cv_model = CLIPVisionModelWithProjection.from_pretrained(f"{pretrained_model_path}/clip")
        self.projection = ResidualMLP()
        self.cv_processor = AutoProcessor.from_pretrained(f"{pretrained_model_path}/clip")
        self.freeze_models(
            [self.vae, self.text_encoder, self.unet, self.cv_model])
        self.unet.eval()
        self.text_encoder.eval()
        self.cv_model.eval()

    def get_prompt_ids(self, prompt):
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return prompt_ids

    def freeze_models(self, models_to_freeze):
        for model in models_to_freeze:
            if model is not None: model.requires_grad_(False)

    def tensor_to_vae_latent(self, t):
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215
        return latents

    def forward(self, batch):

        pixel_values = batch["pixel_values"]  # 获取像素值 范围为[-1,1]
        video_length = pixel_values.shape[1]  # num_frames
        original_pixel_values = ((pixel_values + 1) * 127.5).to(dtype=torch.int)  # 从归一化中恢复图像,范围为0-255
        original_pixel_values = rearrange(original_pixel_values, "b f c h w -> (b f) c h w")
        original_pixel_values = self.cv_processor(images=original_pixel_values,  # 图像clip预处理,返回图片大小(224,224)
                                                  return_tensors="pt")["pixel_values"].cuda()
        # pixel_values_ = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        pixel_values_ = (pixel_values + 1) / 2
        first_feature = self.first_feature(pixel_values_)
        image_tokens = self.cv_model(original_pixel_values).last_hidden_state  #[bf,257,1280]

        encoder_hidden_states = self.get_prompt_ids(batch["captions"])  # 文本clip编码预处理
        encoder_hidden_states = self.text_encoder(encoder_hidden_states.cuda())[0]  # [b,77,1024]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=video_length, dim=0)  # [b*f,77,1024]
        image_tokens = self.projection(image_tokens)  # 1280->1024[b*f,257,1024]
        encoder_hidden_states_c = self.cross_attn(encoder_hidden_states, image_tokens)  # 模态融合[b*f,77,1024]

        latents = self.tensor_to_vae_latent(pixel_values)  # 用vae压缩 latents(b,4,f,h/8,w/8)

        # Get video length
        video_length = latents.shape[2]
        bsz = latents.shape[0]

        # Sample 0
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        timesteps = torch.zeros_like(timesteps)

        _, ret_ft_unet = self.unet(latents, timesteps,
                                   encoder_hidden_states=encoder_hidden_states_c).sample
        final_ret_ft = []
        for ei in ret_ft_unet:  # unet提取的多尺度特征
            final_ret_ft.append(ei)

        my_decout_f = []
        for i in range(len(final_ret_ft) // 2):
            try:
                my_decout_f.append(torch.cat([final_ret_ft[i], final_ret_ft[len(final_ret_ft) - i - 1]], dim=1))
            except:
                a = final_ret_ft[i]
                b = final_ret_ft[len(final_ret_ft) - i - 1]
                a = a[:, :, 0:b.size(2), 0:b.size(3)]
                my_decout_f.append(torch.cat([a, b], dim=1))
        layers = my_decout_f
        my_decout = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        my_decout.insert(0, first_feature)
        return my_decout

class Backbone(nn.Module):
    def __init__(self, position_embedding, pretrained_model_path='./weight', strides=[4,8,16,32], num_channels=[128, 256, 512, 1024]):
        """
        strides: x4,x8,x16,x32
        """
        super().__init__()
        self.strides=strides
        self.num_channels=num_channels
        self.diff_feature_extract = FeatureExtractor(pretrained_model_path=pretrained_model_path, num_channels=num_channels)
        self.position_embedding=position_embedding
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, tensor_list: NestedTensor, captions: str):
        _, t = tensor_list.tensors.shape[:2]
        tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')
        tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')
        n_imgs = rearrange(tensor_list.tensors, '(b t) c h w -> b t c h w', t=t)
        o_imgs = n_imgs * self.std.cuda() + self.mean.cuda()
        dn_imgs = o_imgs * 2 - 1
        batch = {}
        batch["pixel_values"] = dn_imgs
        batch["captions"] = captions
        xs1 = self.diff_feature_extract(batch)
        xs = {}
        for i, eft in enumerate(xs1):
            xs[str(i)] = eft
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append( NestedTensor(x, mask))
        # position encoding
        for x in out:
            pos.append(self.position_embedding(x).to(x.tensors.dtype))
        return out, pos


def build_modelscope_backbone(args):
    position_embedding = build_position_encoding(args)
    model = Backbone(position_embedding, pretrained_model_path=args.pretrained_model_path)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = FeatureExtractor()
    model.cuda()
    batch = {"pixel_values": torch.zeros((1, 5, 3, 224, 224)), "captions": "a word"}
    batch["pixel_values"] = batch["pixel_values"].to(device)
    print(batch["pixel_values"].device)
    my_decout = model(batch)
    for feature in my_decout:
        print(feature.shape)
