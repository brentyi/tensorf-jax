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

import fifteen
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import tyro
from jax import numpy as jnp
from PIL import Image
from typing_extensions import Literal, assert_never

import tensorf.cameras
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

    render_width: int = 400
    render_height: int = 400
    render_fov_x: float = onp.pi / 2.0

    rotation_axis: Literal["world_z", "camera_up"] = "camera_up"


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
    if config.dataset_type == "blender":
        train_views = tensorf.data.load_blender_dataset(
            config.dataset_path, split="train", progress_bar=True
        )
    elif config.dataset_type == "nerfstudio":
        train_views = tensorf.data.load_nerfstudio_dataset(
            config.dataset_path, progress_bar=True
        )
    else:
        assert_never(config.dataset_type)

    initial_T_camera_world = train_views[0].camera.T_camera_world
    initial_T_camera_world = jaxlie.SE3.from_rotation_and_translation(
        initial_T_camera_world.rotation(),
        initial_T_camera_world.translation() * 0.8,
    )
    camera = tensorf.cameras.Camera.from_fov(
        T_camera_world=initial_T_camera_world,
        image_width=args.render_width,
        image_height=args.render_height,
        fov_x_radians=args.render_fov_x,
    )

    # Get rotation axis.
    if args.rotation_axis == "world_z":
        rotation_axis = onp.array([0.0, 0.0, 1.0])
    elif args.rotation_axis == "camera_up":
        # In the OpenCV convention, the "camera up" is -Y.
        up_vectors = onp.array(
            [
                view.camera.T_camera_world.rotation().inverse()
                @ onp.array([0.0, -1.0, 0.0])
                for view in train_views
            ]
        )
        rotation_axis = onp.mean(up_vectors, axis=0)
        rotation_axis /= onp.linalg.norm(rotation_axis)
    else:
        assert_never(args.rotation_axis)

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
                near=0.05,
                far=50.0,
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
                jaxlie.SO3.exp(2 * jnp.pi / args.frames * rotation_axis)
            ),
        )


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(tyro.cli(Args))
