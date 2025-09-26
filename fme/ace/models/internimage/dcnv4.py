import dataclasses
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from InternImage_master.segmentation.mmseg_custom.models.backbones.intern_image import (
    InternImage as SegInternImage,
)


class FeaturePyramidDecoder(nn.Module):
    """Minimal feature pyramid decoder for InternImage outputs."""

    def __init__(
        self,
        feature_channels: Sequence[int],
        embed_dim: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(channels, embed_dim, kernel_size=1) for channels in feature_channels]
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x: Optional[torch.Tensor] = None
        for feature, lateral in zip(reversed(features), reversed(self.lateral_convs)):
            y = lateral(feature)
            if x is None:
                x = y
            else:
                x = F.interpolate(
                    x,
                    size=y.shape[-2:],
                    mode=self.upsample_mode,
                    align_corners=False,
                )
                x = x + y
        if x is None:
            raise ValueError("Feature list for decoder is empty")

        x = self.output_conv(x)
        x = F.interpolate(
            x, size=output_size, mode=self.upsample_mode, align_corners=False
        )
        return x


class InternImageDCNv4(nn.Module):
    """ACE-compatible InternImage encoder-decoder with DCNv4 kernels."""

    def __init__(
        self,
        params: "InternImageDCNv4Config",
        in_chans: int,
        out_chans: int,
        img_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        depths = tuple(params.depths)
        groups = tuple(params.groups)
        out_indices = tuple(params.out_indices)
        level2_ids = (
            tuple(params.level2_post_norm_block_ids)
            if params.level2_post_norm_block_ids is not None
            else None
        )

        self.backbone = SegInternImage(
            core_op=params.core_op,
            channels=params.channels,
            depths=list(depths),
            groups=list(groups),
            mlp_ratio=params.mlp_ratio,
            drop_rate=params.drop_rate,
            drop_path_rate=params.drop_path_rate,
            drop_path_type=params.drop_path_type,
            act_layer=params.act_layer,
            norm_layer=params.norm_layer,
            layer_scale=params.layer_scale,
            offset_scale=params.offset_scale,
            post_norm=params.post_norm,
            with_cp=params.with_cp,
            dw_kernel_size=params.dw_kernel_size,
            level2_post_norm=params.level2_post_norm,
            level2_post_norm_block_ids=list(level2_ids) if level2_ids is not None else None,
            res_post_norm=params.res_post_norm,
            center_feature_scale=params.center_feature_scale,
            use_dcn_v4_op=True,
            out_indices=out_indices,
            in_chans=in_chans,
        )

        feature_channels = [int(params.channels * 2**idx) for idx in out_indices]
        self.decoder = FeaturePyramidDecoder(
            feature_channels=feature_channels,
            embed_dim=params.decoder_embed_dim,
            out_channels=out_chans,
            upsample_mode=params.upsample_mode,
        )
        self.img_shape = img_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        features = self.backbone(x)
        return self.decoder(features, spatial_size)


@dataclasses.dataclass
class InternImageDCNv4Config:
    """Configuration container for the InternImageDCNv4 model."""

    core_op: str = "DCNv4"
    channels: int = 64
    depths: tuple[int, int, int, int] = (3, 4, 18, 5)
    groups: tuple[int, int, int, int] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.2
    drop_path_type: str = "linear"
    act_layer: str = "GELU"
    norm_layer: str = "LN"
    layer_scale: Optional[float] = None
    offset_scale: float = 1.0
    post_norm: bool = False
    with_cp: bool = False
    dw_kernel_size: Optional[int] = None
    level2_post_norm: bool = False
    level2_post_norm_block_ids: Optional[tuple[int, ...]] = None
    res_post_norm: bool = False
    center_feature_scale: bool = False
    out_indices: tuple[int, ...] = (0, 1, 2, 3)
    decoder_embed_dim: int = 256
    upsample_mode: str = "bilinear"
