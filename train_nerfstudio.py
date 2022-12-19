"""Training script for real scenes stored using the nerfstudio format.

For helptext, try running:
```
python train_nerfstudio.py --help
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

    # Default configuration for nerfstudio dataset.
    default_config = tensorf.train_config.TensorfConfig(
        run_dir=pathlib.Path(f"./runs/nerfstudio-{fifteen.utils.timestamp()}"),
        dataset_path=pathlib.Path("./data/dozer"),
        dataset_type="nerfstudio",
        n_iters=30000,
        # Note that the aabb is ignored when scene contraction is on.
        initial_aabb_min=(-2.0, -2.0, -2.0),
        initial_aabb_max=(2.0, 2.0, 2.0),
        appearance_feat_dim=48,
        density_feat_dim=32,
        feature_n_freqs=6,
        viewdir_n_freqs=6,
        grid_dim_init=128,
        grid_dim_final=300,
        upsamp_iters=(2_500, 5_000, 10_000),
        scene_contraction=True,
        camera_embeddings=True,
        render_near=0.05,
        render_far=200.0,
        train_ray_sample_multiplier=3.0,
        minibatch_size=2048,
    )

    # Run training loop! Note that we can set a default value for a function via
    # `functools.partial()`.
    tyro.cli(
        functools.partial(
            tensorf.training.run_training_loop,
            config=default_config,
        )
    )
