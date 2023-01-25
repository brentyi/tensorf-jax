"""Training script for lego dataset.

For helptext, try running:
```
python train_lego.py --help
```
"""

import functools
import pathlib

import fifteen
import tyro

import tensorf.train_config
import tensorf.training

if __name__ == "__main__":
    # Open PDB after runtime errors.
    fifteen.utils.pdb_safety_net()

    # Default configuration for lego dataset.
    default_config = tensorf.train_config.TensorfConfig(
        run_dir=pathlib.Path(f"./runs/lego-{fifteen.utils.timestamp()}"),
        dataset_path=pathlib.Path("./data/nerf_synthetic/lego"),
        dataset_type="blender",
        n_iters=30000,
        initial_aabb_min=(-0.6585, -1.1833, -0.4651),
        initial_aabb_max=(0.6636, 1.1929, 1.0512),
        appearance_feat_dim=48,
        density_feat_dim=16,
        feature_n_freqs=2,
        viewdir_n_freqs=2,
        density_grid_dim_init=128,
        density_grid_dim_final=300,
        app_grid_dim_init=128,
        app_grid_dim_final=300,
        upsamp_iters=(2000, 3000, 4000, 5500, 7000),
    )

    # Run training loop! Note that we can set a default value for a function via
    # `functools.partial()`.
    tyro.cli(
        functools.partial(
            tensorf.training.run_training_loop,
            config=default_config,
        )
    )
