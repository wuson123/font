import math
import torch
import torch.nn as nn
from diffusers import ModelMixin
from diffusers.configuration_utils import (
    ConfigMixin,
    register_to_config
)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.modules.multimodal_aggregator import MultimodalAggregator


class DiffuserModel(ModelMixin, ConfigMixin):
   

    @register_to_config
    def __init__(
            self,
            unet,
            style_encoder,
            content_encoder,
            attention_block,
            use_multimodal_aggregator=False,
            fusion_type="concat",
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.attention_block = attention_block

        self.use_multimodal_aggregator = use_multimodal_aggregator
        self.fusion_type = fusion_type

        if use_multimodal_aggregator:
            style_out_channels = style_encoder.arch['out_channels'][-1]
            content_out_channels = content_encoder.arch['out_channels'][-1]

            if fusion_type == "concat":
                self.multimodal_aggregator = MultimodalAggregator(
                    input_dims=[style_out_channels, content_out_channels],
                    hidden_dim=style_out_channels,
                    output_dim=style_out_channels * 2,  # concat 模式下输出维度是输入的两倍
                    fusion_type=fusion_type,
                    use_residual=True,
                )
            else:
                self.multimodal_aggregator = MultimodalAggregator(
                    input_dims=[style_out_channels, content_out_channels],
                    hidden_dim=style_out_channels,
                    output_dim=style_out_channels,
                    fusion_type=fusion_type,
                    use_residual=True,
                )

    def forward(
            self,
            x_t,
            timesteps,
            style_images,
            content_encoder_downsample_size,
            return_skeleton: bool = False,
    ):
        style_img_feature, _, _ = self.style_encoder(style_images)
        content_img_feature, content_residual_features = self.content_encoder(content_images)

        if self.use_multimodal_aggregator:
            batch_size, channels, height, width = style_img_feature.shape

            if self.fusion_type == "concat":
                concat_features = torch.cat([style_img_feature, content_img_feature], dim=1)
                fused_features = self.multimodal_aggregator.fusion(concat_features)
            else:
                style_flat = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
                content_flat = content_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

                fused_flat = self.multimodal_aggregator(
                    modal_features=[style_flat, content_flat]
                )
                fused_features = fused_flat.permute(0, 2, 1).reshape(batch_size, channels, height, width)

            style_img_feature = fused_features

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channel)


        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
            return_dict=False,
            return_skeleton=return_skeleton,
        )

        skel_logits = None
        scdca_loss = None

        if isinstance(out, (tuple, list)):
            if len(out) == 4:
                noise_pred, offset_out_sum, skel_logits, scdca_loss = out
            elif len(out) == 3:
                noise_pred, offset_out_sum, skel_logits = out
            else:
                noise_pred, offset_out_sum = out
        else:
            noise_pred = out.sample
            offset_out_sum = getattr(out, "offset_out_sum", 0.0)
            skel_logits = getattr(out, "skeleton_logits", None)
            scdca_loss = getattr(out, "scdca_loss", None)

        return noise_pred, offset_out_sum, skel_logits, scdca_loss


class DiffuserModelDPM(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            unet,
            style_encoder,
            content_encoder,
            use_multimodal_aggregator=False,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.use_multimodal_aggregator = use_multimodal_aggregator
        self.fusion_type = fusion_type

        if use_multimodal_aggregator:
            style_out_channels = style_encoder.arch['out_channels'][-1]
            content_out_channels = content_encoder.arch['out_channels'][-1]

            if fusion_type == "concat":
                self.multimodal_aggregator = MultimodalAggregator(
                    input_dims=[style_out_channels, content_out_channels],
                    output_dim=style_out_channels * 2,  # concat 模式下输出维度是输入的两倍
                    fusion_type=fusion_type,
                    use_residual=True,
                )
            else:
                self.multimodal_aggregator = MultimodalAggregator(
                    input_dims=[style_out_channels, content_out_channels],
                    output_dim=style_out_channels,
                    fusion_type=fusion_type,
                    use_residual=True,
                )

    def forward(
            self,
            x_t,
            timesteps,
            cond,
            content_encoder_downsample_size,
            version,
            return_skeleton: bool = False,
    ):
        content_images = cond[0]
        style_images = cond[1]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        content_img_feture, content_residual_features = self.content_encoder(content_images)

        if self.use_multimodal_aggregator:
            batch_size, channels, height, width = style_img_feature.shape

            if self.fusion_type == "concat":
                concat_features = torch.cat([style_img_feature, content_img_feture], dim=1)
                fused_features = self.multimodal_aggregator.fusion(concat_features)
            else:
                style_flat = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
                content_flat = content_img_feture.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

                fused_flat = self.multimodal_aggregator(
                    modal_features=[style_flat, content_flat]
                )
                fused_features = fused_flat.permute(0, 2, 1).reshape(batch_size, channels, height, width)

            style_img_feature = fused_features

        content_residual_features.append(content_img_feture)

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
            return_skeleton=return_skeleton,
        )
        noise_pred = out[0]
        if return_skeleton:
            struct_map = out[2] if isinstance(out, (tuple, list)) and len(out) > 2 else None
            return noise_pred, struct_map

        return noise_pred