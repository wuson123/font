from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin,
                                           register_to_config)
from diffusers.utils import BaseOutput, logging
from src.modules.pinn_module import PINNModule, AttentionBlock
from src.modules.scdca import SCDCA
from typing import Any, Dict, Optional

logger = logging.get_logger(__name__)

# ============================================================
# [新增] Skeleton 分支：Head + Loss
# ============================================================
class SkeletonHead(nn.Module):
    """
    输入:  [B, C, H, W]  (来自 UNet 上采样结束后的特征)
    输出:  [B, 1, H, W]  (logits)
    """
    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(8, mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(8, mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  [B,1,H,W]
    targets: [B,1,H,W] in {0,1} 或 soft [0,1]
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def skeleton_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_bce: float = 1.0,
    w_dice: float = 1.0,
) -> torch.Tensor:
    """
    BCE + Dice（骨架非常稀疏，Dice 很关键）
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return w_bce * bce + w_dice * dice



# [改] UNetOutput：新增 skeleton_logits 字段（兼容 return_dict=False）

@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor
    skeleton_logits: Optional[torch.FloatTensor] = None
    offset_out_sum: Optional[torch.FloatTensor] = None
    scdca_loss: Optional[torch.FloatTensor] = None  # 新增


# @dataclass
# class UNetOutput(BaseOutput):
#     sample: torch.FloatTensor

class UNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = None,
            up_block_types: Tuple[str] = None,
            attention_head_dim: int = 8,
            channel_attn: bool = False,
            content_encoder_downsample_size: int = 4,
            content_start_channel: int = 16,
            reduction: int = 32,
    ):
        super().__init__()

        # 在 unet 的初始化添加 AttentionBlock
        self.attention_block = AttentionBlock(in_channels=512, reduction_ratio=32)

        self.content_encoder_downsample_size = content_encoder_downsample_size
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input 输入卷积
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time 时间步嵌入
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]

        self.scdca_layers = nn.ModuleList([])  # 新增：每个上采样层配一个对齐模块

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            content_channel = content_start_channel * (2 ** (content_encoder_downsample_size - i - 1))

            print("Load the up block ", up_block_type)
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,  # larger 1 than the down block
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                upblock_index=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            # 为当前层添加 SCDCA
            self.scdca_layers.append(SCDCA(in_channels=output_channel, t_embed_dim=time_embed_dim))

        # out 输出卷积
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # [新增] Skeleton head：输入通道 block_out_channels[0]（通常=320）
        # 在 post-process 之前预测骨架最干净
        # ✅ 用 conv_in 的输出通道数作为 skel_head 输入通道，保证永远匹配 early_feat
        skel_in_ch = self.conv_in.out_channels
        self.skel_head = SkeletonHead(in_ch=skel_in_ch, mid_ch=min(64, skel_in_ch))


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            content_encoder_downsample_size: int = 4,
     
            # 新增 return_skeleton：训练打开，采样可关
            return_skeleton: bool = False,
    ) -> Union[UNetOutput, Tuple]:

 
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time嵌入
        timesteps = timestep  # only one time
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # 2. pre-process 输入预处理
        sample = self.conv_in(sample)

        # 3. down下采样
        down_block_res_samples = (sample,)
        for index, downsample_block in enumerate(self.down_blocks):
            if (hasattr(downsample_block, "attentions") and downsample_block.attentions is not None) or hasattr(
                    downsample_block, "content_attentions"):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    index=index,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        # 4. mid 中间模块
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                index = content_encoder_downsample_size,
                encoder_hidden_states = encoder_hidden_states
            )
    

        # 在上采样循环开始前，通过深层特征插值初步预测一个全局结构图
        target_size = down_block_res_samples[0].shape[-2:]
        early_feat = F.interpolate(sample, size=target_size, mode='bilinear', align_corners=True)
        struct_map = early_feat[:, :1].detach()
        struct_map = (struct_map - struct_map.min()) / (struct_map.max() - struct_map.min() + 1e-6)

        # 5. up 上采样
        offset_out_sum = 0
        # 新增
        total_scdca_loss = 0.0


        # 核心集成 SCDCA 特征对齐
        # 将 struct_map 缩放到当前特征层级
        curr_s_map = F.interpolate(struct_map, size=sample.shape[-2:], mode='bilinear', align_corners=True)
        # 使用 SCDCA 对 sample 进行结构约束下的内容对齐
        sample, l_struct, _ = self.scdca_layers[i](
            F_x=sample,
            F_c=sample,
            t=timesteps,
            struct_map=curr_s_map
        )
        total_scdca_loss += l_struct

        # 新增 Skeleton logits（在输出卷积前预测）
        skel_logits = None
        if return_skeleton:
            skel_logits = self.skel_head(sample)  # [B,1,H,W]

        # 6. post-process 输出后处理
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # if not return_dict:
        #     return (sample, offset_out_sum)
        if not return_dict:
            return (sample, offset_out_sum, skel_logits, total_scdca_loss)

        return UNetOutput(
            sample=sample,
            skeleton_logits=skel_logits,
            scdca_loss=total_scdca_loss,
            offset_out_sum=offset_out_sum
        )

        #return UNetOutput(sample=sample)
