import dataclasses
from typing import Optional, Tuple

from fme.ace.models.internimage.dcnv4 import InternImageDCNv4, InternImageDCNv4Config
from fme.ace.registry.registry import ModuleConfig, ModuleSelector


@ModuleSelector.register("InternImageDCNv4")
@dataclasses.dataclass
class InternImageDCNv4Builder(ModuleConfig):
    """Builder to expose the InternImage DCNv4 model through the ACE registry."""

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
    level2_post_norm_block_ids: Optional[Tuple[int, ...]] = None
    res_post_norm: bool = False
    center_feature_scale: bool = False
    out_indices: tuple[int, ...] = (0, 1, 2, 3)
    decoder_embed_dim: int = 256
    upsample_mode: str = "bilinear"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
    ) -> InternImageDCNv4:
        params = InternImageDCNv4Config(**dataclasses.asdict(self))
        return InternImageDCNv4(
            params=params,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
        )
