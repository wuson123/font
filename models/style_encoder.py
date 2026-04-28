import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)


def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
  for y in ys:
      x = x - proj(x, y)
  return x



class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
    
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                          kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
    
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


class StyleEncoder(ModelMixin, ConfigMixin):
    """
    This class is to encode the style image to image embedding.
    Downsample scale is 32.
    For example:
        Input: Shape[Batch, 3, 128, 128]
        Output: Shape[Batch, 255, 4, 4]
    """
    @register_to_config
    def __init__(
        self, 
        G_ch=64, 
        G_wide=True, 
        resolution=128,
        G_kernel_size=3, 
        G_attn='64_32_16_8', 
        n_classes=1000,
        num_G_SVs=1, 
        num_G_SV_itrs=1, 
        G_activation=nn.ReLU(inplace=False),
        nf_mlp = 512, 
        nEmbedding = 256, 
        input_nc = 3,
        output_nc = 3
    ):
        super(StyleEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16


        
        self.out_channel_nultipiler = 1
        self.arch = style_encoder_textedit_addskip_arch(
          self.ch, 
          self.out_channel_nultipiler,
          input_nc
        )[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(
              SNConv2d,
              kernel_size=3, padding=1,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )
            self.which_linear = functools.partial(
              SNLinear,
              num_svs=num_G_SVs, 
              num_itrs=num_G_SV_itrs,
              eps=self.SN_eps
            )
        self.blocks = []

        self.blocks = nn.ModuleList([
          nn.ModuleList(block) for block in self.blocks
        ])
        last_layer = nn.Sequential(
          nn.InstanceNorm2d(self.arch['out_channels'][-1]),
          self.activation,
          nn.Conv2d(
            self.arch['out_channels'][-1],
            self.arch['out_channels'][-1],
            kernel_size=1,
            stride=1
          )
        )
        self.blocks.append(last_layer)
        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self,x):        
        h = x
        residual_features = []
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)        
        h = self.blocks[-1](h)
        style_emd = h        
        h = F.adaptive_avg_pool2d(h,(1,1))
        h = h.view(h.size(0),-1)
        
        return style_emd,h,residual_features
