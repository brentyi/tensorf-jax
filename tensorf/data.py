import json
import pathlib
from typing import List, Literal

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import PIL.Image
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated

from . import cameras


@jdc.pytree_dataclass
class RegisteredRgbaView(jdc.EnforcedAnnotationsMixin):
    """Structure containing 2D image + camera pairs."""

    image_rgba: Annotated[
        jnp.ndarray,
        jnp.floating,  # Range of contents is [0, 1].
    ]
    camera: cameras.Camera


@jdc.pytree_dataclass
class RenderedRays(jdc.EnforcedAnnotationsMixin):
    """Structure containing individual 3D rays in world space + colors."""

    colors: Annotated[jnp.ndarray, (3,), jnp.floating]
    rays_wrt_world: cameras.Rays3D


def rendered_rays_from_views(views: List[RegisteredRgbaView]) -> RenderedRays:
    """Convert a list of registered 2D views into a pytree containing individual
    rendered rays."""

    out = []
    for view in views:
        height = view.camera.image_height
        width = view.camera.image_width

        rays = view.camera.pixel_rays_wrt_world()
        assert rays.get_batch_axes() == (height, width)

        rgba = view.image_rgba
        assert rgba.shape == (height, width, 4)

        # Add white background color; this is what the standard alpha compositing over
        # operator works out to with an opaque white background.
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4])

        out.append(
            RenderedRays(
                colors=rgb.reshape((-1, 3)),
                rays_wrt_world=cameras.Rays3D(
                    origins=rays.origins.reshape((-1, 3)),
                    directions=rays.directions.reshape((-1, 3)),
                ),
            )
        )

    out_concat: RenderedRays = jax.tree_map(
        lambda *children: onp.concatenate(children, axis=0), *out
    )

    # Shape of rays should (N,3), colors should be (N,4), etc.
    assert len(out_concat.rays_wrt_world.get_batch_axes()) == 1
    return out_concat


def load_blender_dataset(
    dataset_root: pathlib.Path,
    split: Literal["test", "train", "val"],
    progress_bar: bool = False,
) -> List[RegisteredRgbaView]:
    """Load a NeRF training dataset as a list of registered views of a scene."""

    with open(dataset_root / f"transforms_{split}.json") as f:
        metadata = json.load(f)
    fov_x_radians: float = metadata["camera_angle_x"]

    out = []

    # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
    # convention.
    T_blendercam_camera = jaxlie.SE3.from_rotation(jaxlie.SO3.from_x_radians(onp.pi))

    for frame in (
        tqdm(metadata["frames"], desc=f"Loading {dataset_root.stem}")
        if progress_bar
        else metadata["frames"]
    ):
        # Expected keys in each frame.
        # TODO: what is rotation?
        assert frame.keys() == {"file_path", "rotation", "transform_matrix"}

        image = onp.array(PIL.Image.open(dataset_root / f"{frame['file_path']}.png"))
        assert image.dtype == onp.uint8
        height, width = image.shape[:2]

        # Note that this is RGBA!
        assert image.shape == (height, width, 4)

        # [0, 255] => [0, 1]
        image = (image / 255.0).astype(onp.float32)

        # Compute extrinsics.
        T_world_blendercam = jaxlie.SE3.from_matrix(
            onp.array(frame["transform_matrix"], dtype=onp.float32)
        )
        T_world_camera = T_world_blendercam @ T_blendercam_camera

        T_camera_world = T_world_camera.inverse()

        out.append(
            RegisteredRgbaView(
                image_rgba=image,  # type: ignore
                camera=cameras.Camera.from_fov(
                    T_camera_world=T_camera_world,
                    image_width=width,
                    image_height=height,
                    fov_x_radians=fov_x_radians,
                ),
            )
        )
    return out


if __name__ == "__main__":
    views = load_blender_dataset(
        pathlib.Path("./data/nerf_synthetic/lego/"), "train", True
    )
    rays = rendered_rays_from_views(views)
    print(rays.get_batch_axes())
