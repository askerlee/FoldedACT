import math
from dataclasses import dataclass
from functools import reduce
from operator import add

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttention, SwinTransformerBlock

from .splits_and_configs import CAM_FRONT, CAM_LEFT, CAM_RIGHT, CAM_TOP


@PreTrainedConfig.register_subclass("folded_act")
@dataclass
class FoldedACTConfig(ACTConfig):
    spatial_folding: bool = False
    swin_after_tokenization: bool = True
    swin_before_folding: bool = False
    camera_overlay: bool = False
    four_cams_overlay_ablation: str | None = None
    early_camera_overlay: bool = False
    sep_swin_for_views: bool = False
    swin_mlp_ratio: float = 4.0

    @property
    def camera_groups(self) -> tuple[tuple[int, ...], ...]:
        camera_keys = tuple(self.image_features)
        if self.four_cams_overlay_ablation == "alt-sum":
            return (
                tuple(camera_keys.index(key) for key in (CAM_TOP, CAM_FRONT, CAM_RIGHT)),
                (camera_keys.index(CAM_LEFT),),
            )
        group_size = 2 if self.camera_overlay else 1
        indices = tuple(range(len(camera_keys)))
        return tuple(indices[start : start + group_size] for start in range(0, len(indices), group_size))


class SwinAttentionOnlyBlock(torch.nn.Module):
    """Shifted-window self-attention residual block without the trailing MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: list[int], shift_size: list[int]) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.norm1(x))


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
            if config.four_cams_overlay_ablation == "alt-sum" and (
                not config.camera_overlay
                or config.early_camera_overlay
                or set(config.image_features) != {CAM_TOP, CAM_FRONT, CAM_LEFT, CAM_RIGHT}
            ):
                raise ValueError("alt-sum requires post-Swin overlay with top, front, left, and right cameras.")
            if config.camera_overlay:
                if len(config.image_features) not in (2, 4):
                    raise ValueError("Camera overlay requires two or four input cameras.")
                if not config.early_camera_overlay and not use_swin:
                    raise ValueError("Late camera overlay requires Swin.")
            if config.sep_swin_for_views and not use_swin:
                raise ValueError("Separate Swin blocks require Swin and at least one input camera.")
            if config.swin_mlp_ratio < 0:
                raise ValueError("swin_mlp_ratio must be non-negative.")
        super().__init__(config)
        use_early_overlay = (
            isinstance(config, FoldedACTConfig)
            and config.camera_overlay
            and config.early_camera_overlay
            and bool(config.image_features)
        )
        if use_early_overlay:
            self._widen_backbone_conv1(num_stacked_views=2)
        if config.image_features:
            self.backbone.to(memory_format=torch.channels_last)
            self.encoder_img_feat_input_proj.to(memory_format=torch.channels_last)
        self.pre_fold_swin = None
        self.view_swin_blocks = torch.nn.ModuleList()
        if use_swin and isinstance(config, FoldedACTConfig):
            num_visual_streams = (
                len(config.image_features) // 2 if use_early_overlay else len(config.image_features)
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

    def _make_swin(self) -> torch.nn.Module:
        mlp_ratio = self.config.swin_mlp_ratio if isinstance(self.config, FoldedACTConfig) else 4.0
        if mlp_ratio == 0:
            return SwinAttentionOnlyBlock(
                dim=self.config.dim_model,
                num_heads=self.config.n_heads,
                window_size=[4, 4],
                shift_size=[2, 2],
            )
        return SwinTransformerBlock(
            dim=self.config.dim_model,
            num_heads=self.config.n_heads,
            window_size=[4, 4],
            shift_size=[2, 2],
            mlp_ratio=mlp_ratio,
        )

    def _project_camera_grid(self, images: Tensor) -> Tensor:
        features = self.backbone(images.contiguous(memory_format=torch.channels_last))["feature_map"]
        projected = self.encoder_img_feat_input_proj(features)
        positions = self.encoder_cam_feat_pos_embed(features).to(projected)
        return projected + positions

    def _camera_grid(self, grid: Tensor, view_index: int) -> Tensor:
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
            if self.config.image_features:
                images = [image.contiguous(memory_format=torch.channels_last) for image in batch[OBS_IMAGES]]
                batch = {**batch, OBS_IMAGES: images}
            return super().forward(batch)

        images = batch[OBS_IMAGES]
        latent, mean, log_variance = self._sample_latent(batch, images)
        prefix = [self.encoder_latent_input_proj(latent)]
        if self.config.robot_state_feature:
            prefix.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            prefix.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.camera_overlay and self.config.early_camera_overlay:
            camera_inputs = [
                torch.cat((first, second), dim=1)
                for first, second in zip(images[::2], images[1::2], strict=True)
            ]
        else:
            camera_inputs = list(images)
        if len(camera_inputs) > 1 and all(image.shape == camera_inputs[0].shape for image in camera_inputs):
            projected_grids = self._project_camera_grid(torch.cat(camera_inputs, dim=0)).chunk(
                len(camera_inputs), dim=0
            )
        else:
            projected_grids = [self._project_camera_grid(image) for image in camera_inputs]
        camera_grids = [self._camera_grid(grid, index) for index, grid in enumerate(projected_grids)]
        if self.config.camera_overlay and not self.config.early_camera_overlay:
            camera_grids = [
                reduce(add, (camera_grids[index] for index in group)) / math.sqrt(len(group))
                for group in self.config.camera_groups
            ]
        prefix_positions = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        camera_pos_grids = [self.encoder_cam_feat_pos_embed(grid).to(prefix_positions) for grid in camera_grids]
        if self.config.spatial_folding:
            camera_grids = [fold_camera_features(grid) for grid in camera_grids]
            camera_pos_grids = [fold_camera_features(pos) for pos in camera_pos_grids]
        camera_tokens = torch.cat([grid.flatten(2).permute(2, 0, 1) for grid in camera_grids], dim=0)
        encoder_tokens = torch.cat((torch.stack(prefix), camera_tokens), dim=0)
        camera_positions = torch.cat([pos.flatten(2).permute(2, 0, 1) for pos in camera_pos_grids], dim=0)
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