import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor
from torchvision.models.swin_transformer import SwinTransformerBlock


@PreTrainedConfig.register_subclass("folded_act")
@dataclass
class FoldedACTConfig(ACTConfig):
    spatial_folding: bool = False
    swin_after_tokenization: bool = True
    swin_before_folding: bool = False
    camera_overlay: bool = False
    sep_swin_for_views: bool = False


def fold_camera_features(features: Tensor) -> Tensor:
    """Pack the two scaled diagonal quadrant sums along height."""
    height, width = features.shape[-2:]
    if height % 2 or width % 2:
        features = F.pad(features, (0, width % 2, 0, height % 2))
    top, bottom = features.chunk(2, dim=-2)
    top_left, top_right = top.chunk(2, dim=-1)
    bottom_left, bottom_right = bottom.chunk(2, dim=-1)
    return torch.cat((top_left + bottom_right, top_right + bottom_left), dim=-2) / math.sqrt(2)


class FoldedACT(ACT):
    def __init__(self, config: ACTConfig) -> None:
        use_swin = (
            isinstance(config, FoldedACTConfig)
            and bool(config.image_features)
            and (config.swin_after_tokenization or (config.spatial_folding and config.swin_before_folding))
        )
        if isinstance(config, FoldedACTConfig):
            if config.camera_overlay:
                if len(config.image_features) not in (2, 4):
                    raise ValueError("Camera overlay requires two or four input cameras.")
            if config.sep_swin_for_views and not use_swin:
                raise ValueError("Separate Swin blocks require Swin and at least one input camera.")
        super().__init__(config)
        if isinstance(config, FoldedACTConfig) and config.camera_overlay and config.image_features:
            self._widen_backbone_conv1(num_stacked_views=2)
        self.pre_fold_swin = None
        self.view_swin_blocks = torch.nn.ModuleList()
        if use_swin and isinstance(config, FoldedACTConfig):
            num_visual_streams = (
                len(config.image_features) // 2 if config.camera_overlay else len(config.image_features)
            )
            if config.sep_swin_for_views:
                self.view_swin_blocks.extend(self._make_swin() for _ in range(num_visual_streams))
            else:
                self.pre_fold_swin = self._make_swin()

    def _widen_backbone_conv1(self, num_stacked_views: int = 2) -> None:
        conv1 = self.backbone.conv1
        if not isinstance(conv1, torch.nn.Conv2d):
            raise TypeError(f"Expected Conv2d at backbone.conv1, got {type(conv1).__name__}")
        padding = conv1.padding if isinstance(conv1.padding, str) else (conv1.padding[0], conv1.padding[1])
        widened = torch.nn.Conv2d(
            in_channels=conv1.in_channels * num_stacked_views,
            out_channels=conv1.out_channels,
            kernel_size=(conv1.kernel_size[0], conv1.kernel_size[1]),
            stride=(conv1.stride[0], conv1.stride[1]),
            padding=padding,
            dilation=(conv1.dilation[0], conv1.dilation[1]),
            groups=conv1.groups,
            bias=conv1.bias is not None,
            padding_mode=conv1.padding_mode,
        )
        with torch.no_grad():
            widened.weight.copy_(conv1.weight.repeat(1, num_stacked_views, 1, 1) / num_stacked_views)
            if conv1.bias is not None and widened.bias is not None:
                widened.bias.copy_(conv1.bias)
        self.backbone.conv1 = widened

    def _make_swin(self) -> SwinTransformerBlock:
        return SwinTransformerBlock(
            dim=self.config.dim_model,
            num_heads=self.config.n_heads,
            window_size=[4, 4],
            shift_size=[2, 2],
        )

    def _camera_grid(self, image: Tensor, view_index: int) -> Tensor:
        features = self.backbone(image)["feature_map"]
        projected = self.encoder_img_feat_input_proj(features)
        positions = self.encoder_cam_feat_pos_embed(features).to(projected)
        tokens = projected.flatten(2).permute(2, 0, 1)
        positioned = tokens + positions.flatten(2).permute(2, 0, 1)
        grid = positioned.permute(1, 2, 0).reshape_as(projected)
        swin = self.view_swin_blocks[view_index] if self.view_swin_blocks else self.pre_fold_swin
        if swin is not None:
            grid = swin(grid.movedim(1, -1)).movedim(-1, 1)
        return grid

    def _sample_latent(self, batch, images):
        batch_size = images[0].shape[0]
        if not (self.config.use_vae and self.training):
            return images[0].new_zeros(batch_size, self.config.latent_dim), None, None

        cls_token = self.vae_encoder_cls_embed.weight.expand(batch_size, -1, -1)
        prefix = [cls_token]
        if self.config.robot_state_feature:
            prefix.append(self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
        action_tokens = self.vae_encoder_action_input_proj(batch[ACTION])
        action_padding = batch["action_is_pad"]
        prefix_padding = action_padding.new_zeros(batch_size, len(prefix))
        encoded = self.vae_encoder(
            torch.cat([*prefix, action_tokens], dim=1).transpose(0, 1),
            pos_embed=self.get_buffer("vae_encoder_pos_enc").transpose(0, 1),
            key_padding_mask=torch.cat((prefix_padding, action_padding), dim=1),
        )
        mean, log_variance = self.vae_encoder_latent_output_proj(encoded[0]).chunk(2, dim=-1)
        latent = mean + torch.exp(log_variance / 2) * torch.randn_like(mean)
        return latent, mean, log_variance

    def forward(self, batch):
        if not (
            isinstance(self.config, FoldedACTConfig)
            and (
                self.config.camera_overlay
                or self.config.spatial_folding
                or self.pre_fold_swin is not None
                or self.view_swin_blocks
            )
            and self.config.image_features
        ):
            return super().forward(batch)

        images = batch[OBS_IMAGES]
        latent, mean, log_variance = self._sample_latent(batch, images)
        prefix = [self.encoder_latent_input_proj(latent)]
        if self.config.robot_state_feature:
            prefix.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            prefix.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.camera_overlay:
            camera_inputs = [
                torch.cat((first, second), dim=1)
                for first, second in zip(images[::2], images[1::2], strict=True)
            ]
        else:
            camera_inputs = images
        camera_grids = [self._camera_grid(image, index) for index, image in enumerate(camera_inputs)]
        if self.config.spatial_folding:
            camera_grids = [fold_camera_features(grid) for grid in camera_grids]
        camera_tokens = torch.cat([grid.flatten(2).permute(2, 0, 1) for grid in camera_grids], dim=0)
        encoder_tokens = torch.cat((torch.stack(prefix), camera_tokens), dim=0)
        prefix_positions = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        camera_positions = prefix_positions.new_zeros(camera_tokens.shape[0], 1, self.config.dim_model)
        encoder_positions = torch.cat((prefix_positions, camera_positions), dim=0)
        memory = self.encoder(encoder_tokens, pos_embed=encoder_positions)

        decoder_input = encoder_positions.new_zeros(
            self.config.chunk_size, images[0].shape[0], self.config.dim_model
        )
        decoded = self.decoder(
            decoder_input,
            memory,
            encoder_pos_embed=encoder_positions,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        return self.action_head(decoded.transpose(0, 1)), (mean, log_variance)


class FoldedACTPolicy(ACTPolicy):
    config_class = FoldedACTConfig
    name = "folded_act"

    def __init__(self, config: ACTConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = FoldedACT(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()