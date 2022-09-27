"""Visualization helper.

Loads a radiance field and rotates a camera around it, rendering a viewpoint from each
angle.

For a summary of options:
```
python render_360.py --help
```
"""

import dataclasses
import pathlib

import dcargs
import fifteen
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from jax import numpy as jnp
from PIL import Image

import tensorf.data
import tensorf.render
import tensorf.train_config
import tensorf.training


@dataclasses.dataclass
class Args:
    run_dir: pathlib.Path
    """Path to training run outputs."""

    output_dir: pathlib.Path = pathlib.Path("./renders")
    """Renders will be saved to `[output_dir]/image_[i].png`."""

    mode: tensorf.render.RenderMode = tensorf.render.RenderMode.RGB
    """Render mode: RGB or depth."""

    frames: int = 10
    """Number of frames to render."""

    density_samples_per_ray: int = 512
    appearance_samples_per_ray: int = 128
    ray_batch_size: int = 4096 * 4


def main(args: Args) -> None:
    experiment = fifteen.experiments.Experiment(data_dir=args.run_dir)
    config = experiment.read_metadata("config", tensorf.train_config.TensorfConfig)

    # Make sure output directory exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the training state from a checkpoint.
    train_state = tensorf.training.TrainState.initialize(
        config=config,
        grid_dim=config.grid_dim_final,
        prng_key=jax.random.PRNGKey(0),
    )
    train_state = experiment.restore_checkpoint(train_state)
    assert train_state.step > 0

    # Load the training dataset... we're only going to use this to grab a camera.
    train_views = tensorf.data.load_blender_dataset(
        dataset_root=config.dataset_path, split="train", progress_bar=True
    )
    camera = train_views[0].camera
    del train_views

    for i in range(args.frames):
        print(f"Rendering frame {i + 1}/{args.frames}")

        # Render & save image.
        rendered = tensorf.render.render_rays_batched(
            appearance_mlp=train_state.appearance_mlp,
            learnable_params=train_state.learnable_params,
            aabb=train_state.aabb,
            rays_wrt_world=camera.pixel_rays_wrt_world(),
            prng_key=jax.random.PRNGKey(0),
            config=tensorf.render.RenderConfig(
                near=0.1,
                far=10.0,
                mode=args.mode,
                density_samples_per_ray=args.density_samples_per_ray,
                appearance_samples_per_ray=args.appearance_samples_per_ray,
            ),
            batch_size=args.ray_batch_size,
        )
        if len(rendered.shape) == 3:
            # RGB: (H, W, 3)
            image = onp.array(rendered)
            image = onp.clip(image * 255.0, 0.0, 255.0).astype(onp.uint8)
        else:
            # Depth: (H, W)
            image = onp.array(rendered)

            # Visualization heuristics for "depths".
            image = 1.0 / onp.maximum(image, 1e-4)
            image -= 0.15
            image *= 5.0
            image = onp.clip(image * 255.0, 0.0, 255.0).astype(onp.uint8)
            image = onp.tile(image[:, :, None], reps=(1, 1, 3))
        Image.fromarray(image).save(args.output_dir / f"image_{i:03}.png")

        # Rotate camera.
        camera = jdc.replace(
            camera,
            T_camera_world=camera.T_camera_world
            @ jaxlie.SE3.from_rotation(
                jaxlie.SO3.from_z_radians(2 * jnp.pi / args.frames)
            ),
        )


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.cli(Args))
