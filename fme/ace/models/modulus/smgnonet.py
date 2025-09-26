# flake8: noqa
# Spherical Multigrid Neural Operator Network (SMgNO)
# Implementation based on the paper: "A spherical multigrid neural operator for global weather forecasting"
# https://doi.org/10.1038/s41598-025-96208-y

from functools import partial
from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch.utils.checkpoint import checkpoint

from .initialization import trunc_normal_
from .layers import MLP, DropPath, RealFFT2, SpectralAttention2d
from .s2convolutions import SpectralAttentionS2, SpectralConvS2

# layer normalization
try:
    from apex.normalization import FusedLayerNorm
    apex_imported = True
except ImportError:
    apex_imported = False


def pad_lon_circular_lat_zero(x: torch.Tensor, pad_h: int = 1, pad_w: int = 1) -> torch.Tensor:
    """
    Apply periodic padding only along longitude (width), and zero padding along latitude (height).
    - pad_w: padding size for left/right (longitude)
    - pad_h: padding size for top/bottom (latitude)
    """
    # pad longitude (left/right) with circular
    x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    # pad latitude (top/bottom) with constant zero
    x = F.pad(x, (0, 0, pad_h, pad_h), mode="constant", value=0.0)
    return x


class ConvolutionBasedSphericalHarmonicFunctions(nn.Module):
    """
    Convolution based on Spherical Harmonic Functions (CSHFs)
    Implementation of Eq. (7) from the paper:
    θ̃ - θ̃' = k_α * (k_θ * g)(λ, φ) ≈ Σ θ̃'(l) * Y_lm(λ, φ) + α * g(λ, φ)
    
    This implements convolution in the spherical harmonic domain with
    learnable truncation compensation.
    """
    
    def __init__(
        self, 
        forward_transform,
        inverse_transform,
        in_channels, 
        out_channels,
        use_spectral_conv=True,
        learnable_alpha=True,
        operator_type="diagonal",
        rank: float = 1.0,
        factorization: Optional[str] = None,
    ):
        super(ConvolutionBasedSphericalHarmonicFunctions, self).__init__()
        
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_spectral_conv = use_spectral_conv
        self.rank = rank
        self.factorization = factorization
        
        # Use SpectralConvS2 for proper spherical convolution
        if use_spectral_conv:
            self.spectral_conv = SpectralConvS2(
                forward_transform,
                inverse_transform,
                in_channels,
                out_channels,
                operator_type=operator_type,
                bias=True,
                rank=rank,
                factorization=factorization,
                use_tensorly=False if factorization is None else True
            )
        else:
            # Fallback to standard convolution with periodic padding
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 0)  # No padding initially
            
        # Projection to align input g (in_channels) to conv_output channels (out_channels)
        self.g_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Learnable parameter alpha (per-channel) from Eq. (7), as a scalar pulse strength at poles
        if learnable_alpha:
            self.alpha_pole = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('alpha_pole', None)
        
    def forward(self, x):
        """
        Forward pass implementing Eq. (7)
        
        θ̃ ≈ Σ θ̃'(l) * Y_lm(λ, φ) + α * g(λ, φ)
        """
        if self.use_spectral_conv:
            # Apply spherical convolution in spectral domain
            conv_output = self.spectral_conv(x)
            # SpectralConvS2 returns (output, residual) when scale_residual is True
            if isinstance(conv_output, tuple):
                conv_output, _ = conv_output
        else:
            # Apply standard convolution with periodic padding in longitude only
            x_padded = pad_lon_circular_lat_zero(x, pad_h=1, pad_w=1)
            conv_output = self.conv(x_padded)
        
        # Add learnable pulse α · g(λ, φ) for truncation error compensation (per-channel scalar at poles)
        # Here g(λ, φ) ≈ input x (local field), modulated by a tiny polar Gaussian mask (impulse-like).
        if hasattr(self, 'alpha_pole') and self.alpha_pole is not None:
            nlat, nlon = conv_output.shape[-2:]
            lat_indices = torch.arange(nlat, device=conv_output.device, dtype=conv_output.dtype)
            # Tiny Gaussian at north pole (idx=0) and south pole (idx=nlat-1)
            sigma = 1.5
            gauss_n = torch.exp(-0.5 * (lat_indices / sigma) ** 2)
            gauss_s = torch.exp(-0.5 * ((lat_indices - (nlat - 1)) / sigma) ** 2)
            lat_mask = (gauss_n + gauss_s)
            # Normalize mask to [0,1]
            lat_mask = lat_mask / (lat_mask.max() + 1e-6)
            lat_mask = lat_mask.view(1, 1, nlat, 1)
            alpha = torch.tanh(self.alpha_pole).view(1, -1, 1, 1)
            # Ensure x matches conv_output spatial size if fallback path used
            g = x
            if g.shape[-2:] != (nlat, nlon):
                g = F.interpolate(g, size=(nlat, nlon), mode="bilinear", align_corners=False)
            # Project g to match out_channels for safe addition
            g = self.g_proj(g)
            conv_output = conv_output + alpha * lat_mask * g
            
        return conv_output


class MultigridSmoothingOperator(nn.Module):
    """
    Semi-iterative smoothing operator implementing Algorithm 1 lines 5-9
    
    Algorithm 1 (lines 5-9):
    for i = 1 to ν^l do
        u^{l,i} = u^{l,i-1} + k^l · W · σ(S^l[u^{l,i-1}] - u^{l,i-1})
    end for
    """
    
    def __init__(
        self,
        forward_transform,
        inverse_transform, 
        embed_dim,
        smoothing_iterations=2,
        use_cshfs=True,
        rank: float = 1.0,
        factorization: Optional[str] = None,
        checkpoint_smoothing: bool = False,
    ):
        super(MultigridSmoothingOperator, self).__init__()
        
        self.embed_dim = embed_dim
        self.smoothing_iterations = smoothing_iterations  # ν^l in Algorithm 1
        self.use_cshfs = use_cshfs
        self.checkpoint_smoothing = checkpoint_smoothing
        
        # Smoothing operator S^l using CSHFs
        if use_cshfs:
            self.S = ConvolutionBasedSphericalHarmonicFunctions(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                use_spectral_conv=True,
                learnable_alpha=True,
                rank=rank,
                factorization=factorization,
            )
        else:
            # Standard convolution with padding=1 so output shape matches input
            self.S = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            
        # Learnable parameter k^l from Algorithm 1
        self.k = nn.Parameter(torch.ones(1))
        
        # Projection matrix W from Algorithm 1
        # Apply channel-wise since we're working with spatial data
        self.W = nn.Conv2d(embed_dim, embed_dim, 1, bias=True)
        
        # Activation function σ (GELU as mentioned in paper)
        self.activation = nn.GELU()
        
    def _smooth_once(self, u: torch.Tensor) -> torch.Tensor:
        if self.use_cshfs:
            Su = self.S(u)
        else:
            Su = self.S(u)
        smoothing_diff = Su - u
        projected = self.W(smoothing_diff)
        activated = self.activation(projected)
        return u + self.k * activated

    def forward(self, u_prev):
        """
        Semi-iterative smoothing operation following Algorithm 1
        
        Algorithm 1 (lines 5-9):
        for i = 1 to ν^l do
            u^{l,i} = u^{l,i-1} + k^l · W · σ(S^l[u^{l,i-1}] - u^{l,i-1})
        end for
        
        Args:
            u_prev: u^{l,i-1} (previous solution at level l)
        Returns:
            u^{l,ν^l}: smoothed solution after ν^l iterations
        """
        u = u_prev

        for _ in range(self.smoothing_iterations):
            if self.training and self.checkpoint_smoothing:
                u = checkpoint(self._smooth_once, u)
            else:
                u = self._smooth_once(u)

        return u


class MultigridRestrictionOperator(nn.Module):
    """
    Restriction operator for transferring from fine to coarse grid
    Uses 3x3 convolution with 2x2 stride as mentioned in the paper
    
    Important: Uses periodic padding in longitudinal direction to ensure continuity
    """
    
    def __init__(self, channels):
        super(MultigridRestrictionOperator, self).__init__()
        # No padding in conv since we'll apply periodic padding manually
        self.restriction = nn.Conv2d(
            channels, channels, 
            kernel_size=3, 
            stride=2, 
            padding=0  # No padding, we'll handle it manually
        )
        
    def forward(self, x):
        # Apply periodic padding only in longitude; zero in latitude
        x_padded = pad_lon_circular_lat_zero(x, pad_h=1, pad_w=1)
        return self.restriction(x_padded)


class MultigridProlongationOperator(nn.Module):
    """
    Prolongation operator for transferring from coarse to fine grid
    Uses pixel shuffle to avoid checkerboard artifacts as mentioned in the paper
    """
    
    def __init__(self, channels):
        super(MultigridProlongationOperator, self).__init__()
        # Use pixel shuffle instead of transposed convolution to avoid checkerboard artifacts
        self.upscale_conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.upscale_conv(x)
        x = self.pixel_shuffle(x)
        return x


class SFNOCoarsestGridSolver(nn.Module):
    """
    SFNO-inspired solver for the coarsest grid.
    Uses a reduced embedding width to match the "low-resolution SFNO" described in the paper.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        coarse_embed_dim: Optional[int] = None,
        operator_type="diagonal",
        rank: float = 1.0,
        factorization: Optional[str] = None,
    ):
        super(SFNOCoarsestGridSolver, self).__init__()

        self.embed_dim = embed_dim
        self.coarse_embed_dim = coarse_embed_dim or embed_dim

        self.input_norm = nn.InstanceNorm2d(embed_dim, affine=True, track_running_stats=False)
        if self.coarse_embed_dim != embed_dim:
            self.down_proj = nn.Conv2d(embed_dim, self.coarse_embed_dim, 1, bias=False)
            self.up_proj = nn.Conv2d(self.coarse_embed_dim, embed_dim, 1, bias=False)
        else:
            self.down_proj = nn.Identity()
            self.up_proj = nn.Identity()

        self.solver = SpectralConvS2(
            forward_transform,
            inverse_transform,
            self.coarse_embed_dim,
            self.coarse_embed_dim,
            operator_type=operator_type,
            bias=True,
            rank=rank,
            factorization=factorization,
            use_tensorly=False if factorization is None else True
        )

        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        x = self.down_proj(x)
        solver_output = self.solver(x)
        if isinstance(solver_output, tuple):
            x, _ = solver_output
        else:
            x = solver_output
        x = self.activation(x)
        x = self.up_proj(x)
        # Preserve a lightweight residual path for stability
        if isinstance(residual, torch.Tensor) and residual.shape == x.shape:
            x = x + residual
        return x


class MultigridVCycle(nn.Module):
    """
    V-cycle multigrid structure implementing Algorithm 1 from the paper
    
    In the neural network context, this serves as a hierarchical feature processor
    that mimics the multigrid PDE solver structure. The V-cycle provides:
    1. Multi-scale feature processing (fine to coarse to fine)
    2. Efficient computation through hierarchical operations
    3. Better gradient flow through skip connections at different scales
    """
    
    def __init__(
        self,
        transforms_by_level,  # List of (forward_transform, inverse_transform) for each level
        embed_dim,
        max_levels=3,
        smoothing_iterations=2,
        use_cshfs=True,
        rank: float = 1.0,
        factorization: Optional[str] = None,
        checkpoint_smoothing: bool = False,
        coarsest_embed_ratio: float = 0.5,
    ):
        super(MultigridVCycle, self).__init__()
        
        self.max_levels = max_levels
        self.embed_dim = embed_dim
        self.coarsest_embed_ratio = coarsest_embed_ratio
        
        # System operators (A^l) and Smoothing operators (S^l) for each level
        self.system_operators = nn.ModuleList()
        self.smoothing_operators = nn.ModuleList()
        
        # Restriction and prolongation operators
        self.restriction_operators = nn.ModuleList()
        self.prolongation_operators = nn.ModuleList()
        
        for level in range(max_levels):
            forward_transform, inverse_transform = transforms_by_level[level]
            
            # System operator A^l using CSHFs
            if use_cshfs:
                system_op = ConvolutionBasedSphericalHarmonicFunctions(
                    forward_transform,
                    inverse_transform,
                    embed_dim,
                    embed_dim,
                    use_spectral_conv=True,
                    learnable_alpha=True,
                    rank=rank,
                    factorization=factorization,
                )
            else:
                # Standard convolution with periodic padding
                system_op = nn.Conv2d(embed_dim, embed_dim, 3, 1, 0)
            self.system_operators.append(system_op)
            
            # Smoothing operator
            smoother = MultigridSmoothingOperator(
                forward_transform,
                inverse_transform,
                embed_dim,
                smoothing_iterations,
                use_cshfs,
                rank=rank,
                factorization=factorization,
                checkpoint_smoothing=checkpoint_smoothing,
            )
            self.smoothing_operators.append(smoother)
            
            # Restriction and prolongation (except for coarsest level)
            if level < max_levels - 1:
                self.restriction_operators.append(MultigridRestrictionOperator(embed_dim))
                self.prolongation_operators.append(MultigridProlongationOperator(embed_dim))
        
        # SFNO solver for coarsest grid
        coarsest_forward, coarsest_inverse = transforms_by_level[-1]
        coarse_embed_dim = max(8, int(embed_dim * coarsest_embed_ratio))
        self.coarsest_solver = SFNOCoarsestGridSolver(
            coarsest_forward,
            coarsest_inverse,
            embed_dim,
            coarse_embed_dim=coarse_embed_dim,
            rank=rank,
            factorization=factorization,
        )
        
    def forward(self, u, f, level=0):
        """
        V-cycle multigrid solver implementing Algorithm 1
        
        Args:
            u: u^l (current solution at level l)
            f: f^l (right-hand side at level l)  
            level: current grid level l
        Returns:
            u: updated solution
        """
        if level == self.max_levels - 1:
            # Coarsest grid: solve for correction e^l from residual f (= r at coarsest)
            # Return correction directly
            return self.coarsest_solver(f)
        
        # Pre-smoothing: ν^l iterations (Algorithm 1, lines 5-9)
        u = self.smoothing_operators[level](u)
        
        # Compute residual: r^l = f^l - A^l * u^l (Algorithm 1, line 10)
        Au = self.system_operators[level](u)
        r = f - Au
        
        # Restrict residual to coarser grid: r^{l+1} = R_l^{l+1} * r^l (Algorithm 1, line 11)
        r_coarse = self.restriction_operators[level](r)
        
        # Solve on coarser grid recursively: e^{l+1} = V-cycle(0, r^{l+1}) (Algorithm 1, line 12)
        u_coarse0 = torch.zeros_like(r_coarse)
        correction = self.forward(u_coarse0, r_coarse, level + 1)
        
        # Prolongate correction back to fine grid: e^l = P_{l+1}^l * e^{l+1} (Algorithm 1, line 13)
        correction_fine = self.prolongation_operators[level](correction)
        
        # Apply correction: u^l = u^l + e^l (Algorithm 1, line 14)
        u = u + correction_fine
        
        # Post-smoothing: ν^l more iterations (Algorithm 1, lines 15-19)
        u = self.smoothing_operators[level](u)
        
        return u


class SphericalMultigridNeuralOperatorBlock(nn.Module):
    """
    SMgNO Block combining multigrid framework with spectral operations
    """
    
    def __init__(
        self,
        transforms_by_level,
        embed_dim,
        max_levels=3,
        smoothing_iterations=2,
        use_cshfs=True,
        rank: float = 1.0,
        factorization: Optional[str] = None,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_factory=None,
        checkpoint_smoothing: bool = False,
        coarsest_embed_ratio: float = 0.5,
    ):
        super(SphericalMultigridNeuralOperatorBlock, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Normalization
        self.norm1 = norm_factory() if norm_factory is not None else nn.Identity()
        
        # Multigrid V-cycle
        self.multigrid = MultigridVCycle(
            transforms_by_level,
            embed_dim,
            max_levels,
            smoothing_iterations,
            use_cshfs,
            rank=rank,
            factorization=factorization,
            checkpoint_smoothing=checkpoint_smoothing,
            coarsest_embed_ratio=coarsest_embed_ratio,
        )
        
        # Residual connection
        self.skip_connection = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop_rate
        )
        
        # Normalization
        self.norm2 = norm_factory() if norm_factory is not None else nn.Identity()
        
        # Dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        # Store residual for skip connection
        residual = x
        
        # First normalization
        x_norm = self.norm1(x)
        
        # Apply multigrid V-cycle as the main operator
        # Initialize with zero solution and use normalized input as RHS
        u_init = torch.zeros_like(x_norm)
        x_mg = self.multigrid(u_init, x_norm)
        
        # Skip connection with original input
        x = x_mg + self.skip_connection(residual)
        x = self.drop_path(x)
        
        # Second normalization and MLP (following transformer-like structure)
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        x = x + self.drop_path(x_mlp)
        
        return x


class SphericalMultigridNeuralOperatorNet(nn.Module):
    """
    Spherical Multigrid Neural Operator Network (SMgNO)
    
    Implementation of the SMgNO model from the paper:
    "A spherical multigrid neural operator for global weather forecasting"
    https://doi.org/10.1038/s41598-025-96208-y
    
    Key contributions from the paper:
    1. CSHFs (Convolution based on Spherical Harmonic Functions) - Eq. (7)
       - Solves spherical data distortion issues
       - Uses learnable truncation error compensation
    2. Multigrid framework for computational efficiency
       - V-cycle structure with hierarchical processing
       - SFNO integration in coarsest grid for accuracy
    3. Semi-iterative smoothing (Algorithm 1)
       - Replaces residual correction with iterative approach
    4. Periodic padding in longitudinal direction
       - Ensures east-west boundary continuity
    5. Pixel shuffle instead of transposed convolution
       - Avoids checkerboard artifacts
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    img_shape : tuple, optional
        Shape of the input channels, by default (721, 1440)
    scale_factor : int, optional
        Scale factor to use, by default 1
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of SMgNO blocks, by default 12
    max_levels : int, optional
        Maximum number of multigrid levels, by default 3
    smoothing_iterations : int, optional
        Number of smoothing iterations per level, by default 2
    use_cshfs : bool, optional
        Whether to use CSHFs, by default True
    mlp_ratio : float, optional
        Ratio of MLP hidden dimension to embedding dimension, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Drop path rate, by default 0.0
    activation_function : str, optional
        Activation function to use, by default "gelu"
    normalization_layer : str, optional
        Type of normalization layer, by default "instance_norm"
    big_skip : bool, optional
        Whether to use big skip connections, by default True
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    encoder_layers : int, optional
        Number of encoder layers, by default 1
    checkpointing : int, optional
        Checkpointing level, by default 0
    """
    
    def __init__(
        self,
        params,
        spectral_transform: str = "sht",
        img_shape: Tuple[int, int] = (721, 1440),
        scale_factor: int = 1,
        in_chans: int = 2,
        out_chans: int = 2,
        embed_dim: int = 256,
        num_layers: int = 12,
        max_levels: int = 3,
        smoothing_iterations: int = 2,
        use_cshfs: bool = True,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        activation_function: str = "gelu",
        normalization_layer: str = "instance_norm",
        big_skip: bool = True,
        pos_embed: bool = True,
        encoder_layers: int = 1,
        checkpointing: int = 0,
        hard_thresholding_fraction: float = 1.0,
        checkpoint_smoothing: bool = False,
        coarsest_embed_ratio: float = 0.5,
    ):
        super(SphericalMultigridNeuralOperatorNet, self).__init__()
        
        # Store parameters
        self.params = params
        self.spectral_transform = getattr(params, "spectral_transform", spectral_transform)
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else img_shape
        )
        self.scale_factor = getattr(params, "scale_factor", scale_factor)
        self.in_chans = getattr(params, "N_in_channels", in_chans)
        self.out_chans = getattr(params, "N_out_channels", out_chans)
        self.embed_dim = getattr(params, "embed_dim", embed_dim)
        self.num_layers = getattr(params, "num_layers", num_layers)
        self.max_levels = getattr(params, "max_levels", max_levels)
        self.smoothing_iterations = getattr(params, "smoothing_iterations", smoothing_iterations)
        self.use_cshfs = getattr(params, "use_cshfs", use_cshfs)
        self.big_skip = getattr(params, "big_skip", big_skip)
        self.use_pos_embed = getattr(params, "pos_embed", pos_embed)
        self.encoder_layers = getattr(params, "encoder_layers", encoder_layers)
        self.checkpointing = getattr(params, "checkpointing", checkpointing)
        self.hard_thresholding_fraction = getattr(params, "hard_thresholding_fraction", hard_thresholding_fraction)
        self.checkpoint_smoothing = getattr(params, "checkpoint_smoothing", checkpoint_smoothing)
        self.coarsest_embed_ratio = getattr(params, "coarsest_embed_ratio", coarsest_embed_ratio)
        self.rank = getattr(params, "rank", 1.0)
        self.factorization = getattr(params, "factorization", None)
        
        data_grid = getattr(params, "data_grid", "legendre-gauss")
        
        # Compute downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)
        
        # Compute maximum frequencies
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)
        
        # Setup spectral transforms for different grid levels
        self.transforms_by_level = []
        
        for level in range(self.max_levels):
            # Coarser grids for higher levels
            level_factor = 2 ** level
            h_level = max(self.h // level_factor, 16)  # Minimum grid size
            w_level = max(self.w // level_factor, 32)
            
            modes_lat_level = max(modes_lat // level_factor, 8)
            # Ensure |m| <= l by clamping mmax <= lmax and Nyquist in lon
            modes_lon_level = min(modes_lat_level, max((w_level // 2), 1))
            
            if self.spectral_transform == "sht":
                # Handle torch_harmonics API differences across versions
                try:
                    forward_transform = th.RealSHT(
                        h_level, w_level,
                        lmax=modes_lat_level,
                        mmax=modes_lon_level,
                        grid=data_grid,
                    ).float()
                    inverse_transform = th.InverseRealSHT(
                        h_level, w_level,
                        lmax=modes_lat_level,
                        mmax=modes_lon_level,
                        grid=data_grid,
                    ).float()
                except TypeError:
                    # Fallbacks: older versions may not support mmax or grid
                    try:
                        forward_transform = th.RealSHT(
                            h_level, w_level, lmax=modes_lat_level
                        ).float()
                        inverse_transform = th.InverseRealSHT(
                            h_level, w_level, lmax=modes_lat_level
                        ).float()
                    except TypeError:
                        forward_transform = th.RealSHT(h_level, w_level).float()
                        inverse_transform = th.InverseRealSHT(h_level, w_level).float()
                
                # Normalize attribute access for nlat/nlon across versions
                for t in (forward_transform, inverse_transform):
                    if not hasattr(t, "nlat") or not hasattr(t, "nlon"):
                        # Attempt to infer
                        try:
                            t.nlat, t.nlon = h_level, w_level
                        except Exception:
                            pass
            else:
                raise ValueError(f"Unsupported spectral transform: {self.spectral_transform}")
                
            self.transforms_by_level.append((forward_transform, inverse_transform))
        
        # Determine activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        elif activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")
        
        # Encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True))
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]
        
        # Normalization factories (avoid arg collisions inside blocks)
        if normalization_layer == "layer_norm":
            def make_norm():
                return nn.LayerNorm(normalized_shape=(self.h, self.w), eps=1e-6)
        elif normalization_layer == "instance_norm":
            def make_norm():
                return nn.InstanceNorm2d(self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif normalization_layer == "none":
            def make_norm():
                return nn.Identity()
        else:
            raise NotImplementedError(f"Normalization {normalization_layer} not implemented.")
        
        # SMgNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            block = SphericalMultigridNeuralOperatorBlock(
                self.transforms_by_level,
                self.embed_dim,
                self.max_levels,
                self.smoothing_iterations,
                self.use_cshfs,
                rank=self.rank,
                factorization=self.factorization,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_factory=make_norm,
                checkpoint_smoothing=self.checkpoint_smoothing,
                coarsest_embed_ratio=self.coarsest_embed_ratio
            )
            self.blocks.append(block)
        
        # Decoder
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True))
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)
        
        # Positional embedding
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.h, self.w))
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or (apex_imported and isinstance(m, FusedLayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """Helper"""
        return {"pos_embed"}
    
    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
    def forward(self, x):
        # Save big skip
        if self.big_skip:
            residual = x
        
        # Encoder
        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)
        
        # Add positional embedding
        if self.use_pos_embed and hasattr(self, 'pos_embed'):
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Forward through SMgNO blocks
        x = self._forward_features(x)
        
        # Big skip connection
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)
        
        # Decoder
        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)
        
        return x
