"""Training script for lego dataset.

For helptext, try running:
```
python train_lego.py --help
```
"""

import pathlib

import fifteen
import tyro

import tensorf.train_config
import tensorf.training

if __name__ == "__main__":
    # Open PDB after runtime errors.
    fifteen.utils.pdb_safety_net()

    # Default configuration for lego dataset.
    lego_config = tensorf.train_config.TensorfConfig(
        run_dir=pathlib.Path(f"./runs/scene-contraction-{fifteen.utils.timestamp()}"),
        dataset_path=pathlib.Path("./data/stanislaus_field"),
        dataset_type="nerfstudio",
        n_iters=30000,
        initial_aabb_min=(-2.0, -2.0, -2.0),
        initial_aabb_max=(2.0, 2.0, 2.0),
        appearance_feat_dim=48,
        density_feat_dim=32,
        feature_n_freqs=6,
        viewdir_n_freqs=6,
        grid_dim_init=128,
        grid_dim_final=512,
        upsamp_iters=(5000, 7500, 20000),
        scene_contraction=True,
    )

    # Parse arguments.
    config = tyro.cli(
        tensorf.train_config.TensorfConfig,
        default=lego_config,
    )

    # Run training loop!
    tensorf.training.run_training_loop(config)
