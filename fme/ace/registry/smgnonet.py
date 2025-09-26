import dataclasses
from typing import Literal, Optional

from fme.ace.models.modulus.smgnonet import SphericalMultigridNeuralOperatorNet
from fme.ace.registry.registry import ModuleConfig, ModuleSelector


@ModuleSelector.register("SphericalMultigridNeuralOperatorNet")
@dataclasses.dataclass
class SphericalMultigridNeuralOperatorBuilder(ModuleConfig):
    """
    Configuration for the SMgNO (Spherical Multigrid Neural Operator) architecture.
    
    Based on the paper: "A spherical multigrid neural operator for global weather forecasting"
    https://doi.org/10.1038/s41598-025-96208-y
    """

    spectral_transform: str = "sht"
    filter_type: str = "linear"
    operator_type: str = "diagonal"
    scale_factor: int = 1
    embed_dim: int = 256
    num_layers: int = 12
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    
    # SMgNO specific parameters
    max_levels: int = 3
    smoothing_iterations: int = 2
    use_cshfs: bool = True
    mlp_ratio: float = 2.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    rank: float = 1.0
    factorization: Optional[str] = None
    checkpoint_smoothing: bool = False
    coarsest_embed_ratio: float = 0.5

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
    ):
        # Set the channel parameters in the config object
        self.N_in_channels = n_in_channels
        self.N_out_channels = n_out_channels
        
        smgno_net = SphericalMultigridNeuralOperatorNet(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
        )

        return smgno_net
