import dataclasses
import pathlib
from typing import Literal, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    lr_init_tensor: float = 0.02  # `lr_init` in the original code.
    lr_init_mlp: float = 1e-3  # `lr_basis` in the original code.
    lr_decay_iters: Optional[int] = None  # If unset, defaults to n_iters.
    lr_decay_target_ratio: float = 0.1
    lr_upsample_reset: bool = True  # Reset learning rate after upsampling.


@dataclasses.dataclass(frozen=True)
class TensorfConfig:
    run_dir: pathlib.Path

    # Input data directory.
    dataset_path: pathlib.Path

    # Dataset type.
    dataset_type: Literal["blender", "nerfstudio"] = "blender"

    # Training options.
    minibatch_size: int = 4096
    n_iters: int = 30000

    # Optimizer configuration.
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    # Loss options.
    # TODO: these are not yet implemented :')
    # l1_weight_initial: float = 0.0
    # l1_weight_rest: float = 0.0
    # ortho_weight: float = 0.0
    # tv_weight_density: float = 0.0
    # tv_weight_app: float = 0.0

    initial_aabb_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    initial_aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Per-axis tensor decomposition components.
    appearance_feat_dim: int = 24  # n_lambd_sh
    density_feat_dim: int = 8  # n_lambd_sigma

    # Fourier feature frequency counts for both the interpolated feature vector and view
    # direction count; these are used in the appearance MLP.
    feature_n_freqs: int = 6  # fea_pe
    viewdir_n_freqs: int = 6  # view_pe

    # Grid parameters; we define the initial and final grid dimensions as well as when
    # to upsample or update the alpha mask.
    density_grid_dim_init: int = 64  # cbrt(N_voxel_init)
    density_grid_dim_final: int = 64  # cbrt(N_voxel_final)
    app_grid_dim_init: int = 128  # cbrt(N_voxel_init)
    app_grid_dim_final: int = 300  # cbrt(N_voxel_final)
    upsamp_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)

    # TODO: unimplemented.
    # update_alphamask_iters: Tuple[int, ...] = (2000, 4000)

    #
    # ---------------------------
    # Below, features that aren't present in the original TensoRF paper. This is mostly
    # focused on mixed precision and real scenes.
    #

    # If enabled, we use mixed-precision training. This seems to work and speeds up
    # training throughput by a significant factor, but is disabled by default because we
    # haven't fully evaluated stability, impact on convergence, hyperparameters, etc.
    #
    # Important: if mixed precision is enabled, the loss scale should generally be set
    # to something high!
    mixed_precision: bool = False

    # Loss scale for preventing gradient underflow.
    #
    # Applied always but useful mostly for mixed-precision training, where we observe a
    # tradeoff where a higher value will produce lower errors and improve convergence,
    # but can run slower despite a nearly identical computation graph. (possibly due to
    # some reduced sparsity of gradients?)
    loss_scale: float = 1.0

    # For unbounded scenes, we apply MipNeRF-360-inspired scene contraction. Useful for
    # real data.
    bounded_scene: bool = True
    scene_scale: float = 1.0

    # Add NeRF in the wild-inspired camera embeddings.
    camera_embeddings: bool = False

    # Near and far limits for rendering.
    render_near: float = 0.05
    render_far: float = 200.0
    train_ray_sample_multiplier: float = 1.0

    # Proposal networks.
    density_from_appearance_mlp: bool = False
