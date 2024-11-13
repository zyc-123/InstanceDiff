import sys
if __name__=="__main__":
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')
    __package__='models.modules'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip
from .modified_clip import ContextDecoder

import functools

from .attention import SpatialTransformer

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)

# from .attention import SpatialTransformer

artifact_type = ['speckle in OCT', 'speckle in ultra sound', 'noise in cryo-EM image', 'noise in low dose CT',
                 'Gaussian noise in MRI']
type_map_ind = {'speckle in OCT': 0, 'speckle in ultra sound': 1, 'noise in cryo-EM image': 2,
                'noise in low dose CT': 3, 'Gaussian noise in MRI': 4}

from ..BiomedCLIP import get_tokenizer

### discarded
class ScoreMapModule(nn.Module):
    def __init__(self,
                 text_encoder_context_length=42,
                 token_embed_dim=768, #512,
                 CLIP_Type="BiomedCLIP",
                 text_dim=512,
                 visual_dim=64) -> None:
        super(ScoreMapModule, self).__init__()
        self.context_length = 10

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, visual_dim),
        )
        context_length = text_encoder_context_length - self.context_length
        # context aware module
        self.context_decoder = ContextDecoder(
            context_length=context_length,
            transformer_width=visual_dim,
            transformer_heads=4,
            transformer_layers=3,
            visual_dim=visual_dim,
            text_dim=visual_dim,
            out_dim=visual_dim,
            dropout=0.0
        )
        if CLIP_Type=="BiomedCLIP":
            tokenizer = get_tokenizer("./models/BiomedCLIP",
                                      r"./models/BiomedCLIP/BiomedCLIP_config.json")
            self.texts = torch.cat(
                [tokenizer(c, context_length=self.context_length) for c in artifact_type]).cuda()  # len == 10
            self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
        else:
            self.texts = torch.cat([clip.tokenize(c, context_length=self.context_length) for c in artifact_type]).cuda()
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(visual_dim) * 1e-4)

    def forward(self, vision_feat, text_encoder, names, if_return_emb=False):
        self.texts = self.texts.to(vision_feat.device)
        B, C, H, W = vision_feat.shape
        vision_feat = vision_feat.reshape(B, C, -1).permute(0, 2, 1)  # B, N, C
        # encode text by CLIP, stack of self-attention module
        text_feat = text_encoder(self.texts, self.contexts).expand(B, -1, -1)

        # vision_feat = self.pre_vis_ln(vision_feat)
        text_feat = self.text_proj(text_feat)
        text_diff = self.context_decoder(text_feat, vision_feat)
        text_embeddings = text_feat + self.gamma * text_diff
        # normalization
        text_embeddings = F.normalize(text_embeddings, dim=2, p=2)

        vision_feat = vision_feat.permute(0, 2, 1).reshape((B, -1, H, W))
        vision_feat = F.normalize(vision_feat, dim=1, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', vision_feat, text_embeddings)
        indices = [type_map_ind[name] for name in names]
        score_map = score_map[torch.arange(B), indices, :, :].unsqueeze(1)
        if if_return_emb:
            text_emb = text_embeddings[torch.arange(B), indices, :].unsqueeze(1)
            return score_map, text_emb
            pass
        else:
            return score_map


class LearnableForwardUNet_MultiScoreMap_Emb(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4],
                 context_dim=512,
                 text_module='scoremap',
                 CLIP_ScoreMapModuleList=None,
                 score_map_ngf=16,
                 score_map_ch_mult=[1, 1, 2, 4],
                 use_image_context=False, use_degra_context=False):
        """
        Initialize function
        Args:
            text_module(str): Specify text information fusion components; 'scoremap', or 'attention'
            CLIP_ScoreMapModule(class): must be specified when text_module=='scoremap'
        Returns:
            void
        """
        super().__init__()
        self.ch_mult = ch_mult
        self.score_map_ch_mult = score_map_ch_mult
        self.depth = len(ch_mult)
        self.context_dim = -1 if context_dim is None else context_dim
        self.score_map_ngf = score_map_ngf
        self.use_image_context = use_image_context
        self.use_degra_context = use_degra_context

        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc, nf, 7)

        # time embeddings
        time_dim = nf * 4
        self.random_or_learned_sinusoidal_cond = False
        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.text_module = text_module
        # if self.context_dim > 0 and use_degra_context:
        if use_degra_context:
            self.prompt = nn.Parameter(torch.rand(1, time_dim))
            self.text_mlp = nn.Sequential(
                nn.Linear(context_dim, time_dim), NonLinearity(),
                nn.Linear(time_dim, time_dim))
            self.prompt_mlp = nn.Linear(time_dim, time_dim)
        if text_module == 'attention':
            pass
        elif text_module == 'scoremap':
            self.CLIP_ScoreMapModuleList = CLIP_ScoreMapModuleList
        else:
            print(
                f'LearnableFDUnet.py::LearnableForwardUNet::__init__(): No implemented text_module named {text_module}')

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        ch_mult = [1] + ch_mult

        for i in range(self.depth):
            dim_in = nf * ch_mult[i]
            dim_out = nf * ch_mult[i + 1]

            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            dim_head_in = dim_in // num_heads_in

            if use_image_context:
                att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head,
                                                                                    depth=1, context_dim=context_dim)
                att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head,
                                                                                   depth=1, context_dim=context_dim)
            else:
                att_down = LinearAttention(dim_in)  # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out)  # if i < 2 else Attention(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, att_down)),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))

            if text_module == 'scoremap':
                self.ups.insert(0, nn.ModuleList([
                    block_class(dim_in=dim_out + dim_in + score_map_ngf*ch_mult[i], dim_out=dim_out, time_emb_dim=time_dim),
                    block_class(dim_in=dim_out + dim_in + score_map_ngf*ch_mult[i], dim_out=dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, att_up)),
                    Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
                ]))
            else:
                self.ups.insert(0, nn.ModuleList([
                    block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                    block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, att_up)),
                    Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
                ]))

        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        if self.use_image_context:
            self.mid_attn = Residual(PreNorm(mid_dim, SpatialTransformer(mid_dim, num_heads_mid, dim_head, depth=1,
                                                                         context_dim=context_dim)))
        else:
            self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time, names, text_encoder=None, text_context=None, score_map_detach=False, image_context=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time])
        time = time.to(xt.device)

        # x = xt - cond
        x = xt
        x = torch.cat([x, cond], dim=1)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)

        if self.text_module == 'attention':
            prompt_embedding = torch.softmax(self.text_mlp(text_context), dim=1) * self.prompt
            prompt_embedding = self.prompt_mlp(prompt_embedding)
            t = t + prompt_embedding

        h = []
        score_maps = []
        for i in range(len(self.downs)):
            if self.text_module == 'scoremap':
                score_map, text_emb = self.CLIP_ScoreMapModuleList[i](x, text_encoder, names, if_return_emb=True)
                if score_map_detach:
                    score_maps.append(score_map.detach())
                else:
                    score_maps.append(score_map)
                if self.use_degra_context and i == 0:
                    t = t + text_emb


            b1, b2, attn, downsample = self.downs[i]
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x, context=image_context)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context=image_context)
        x = self.mid_block2(x, t)

        for i in range(len(self.ups)):
            b1, b2, attn, upsample = self.ups[i]
            # for b1, b2, attn, upsample in self.ups:

            if self.text_module == 'scoremap':
                score_map = score_maps[-i-1]
                x = torch.cat([score_map.repeat(1, self.score_map_ngf*self.score_map_ch_mult[len(self.ups)-1-i], 1, 1), x, h.pop()], dim=1)
            else:
                x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)

            if self.text_module == 'scoremap':
                x = torch.cat([score_map.repeat(1, self.score_map_ngf*self.score_map_ch_mult[len(self.ups)-1-i], 1, 1), x, h.pop()], dim=1)
            else:
                x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)

            x = attn(x, context=image_context)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        indices = [type_map_ind[name] for name in names]
        x = x[torch.arange(x.shape[0]), indices, :, :].unsqueeze(1)

        return x, score_maps


def create_LearnableForwardUNet_MultiScoreMap(opt, CLIP_ScoreMapModule):
    in_nc = opt['in_nc']
    out_nc = opt['out_nc']
    nf = opt['nf']
    ch_mult = opt['ch_mult']
    context_dim = opt['context_dim']
    text_module = opt['text_module']
    score_map_ngf = opt['score_map_ngf']
    score_map_ch_mult = opt['score_map_ch_mult']
    use_image_context = opt['use_image_context']
    use_degra_context = opt['use_degra_context']
    return LearnableForwardUNet_MultiScoreMap_Emb(in_nc, out_nc, nf,
                                ch_mult=ch_mult,
                                context_dim=context_dim,
                                text_module=text_module,
                                CLIP_ScoreMapModuleList=CLIP_ScoreMapModule,
                                score_map_ngf=score_map_ngf,
                                score_map_ch_mult=score_map_ch_mult,
                                use_degra_context=use_degra_context,
                                use_image_context=use_image_context)


if __name__ == '__main__':
    from .modified_clip import CLIPTextContextEncoder

    text_encoder = CLIPTextContextEncoder(
        context_length=42,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pretrain_pth='./clip_model/RN101.pt'
    ).cuda()
    text_encoder.init_weights()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    chan_mult = [1, 1, 2, 4]

    CLIP_ScoreMapModuleList = nn.ModuleList([ScoreMapModule(visual_dim=64*chan_mult[i]).cuda() for i in range(4)]).cuda()
    net = LearnableForwardUNet_MultiScoreMap_Emb(2, 2, 64,
                                             CLIP_ScoreMapModuleList=CLIP_ScoreMapModuleList,
                                             score_map_ch_mult=chan_mult).cuda()

    x = torch.ones((2, 1, 224, 224)).cuda()
    cond = torch.ones((2, 1, 224, 224)).cuda()
    names = ['speckle in OCT', 'speckle in ultra sound']
    out, score_map = net(x, cond, 50, names, text_encoder)
    print(out.shape)
