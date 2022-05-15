import concurrent.futures
import json
import pathlib
from typing import Iterable, List, Literal, TypeVar

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

    # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
    # convention.
    T_blendercam_camera = jaxlie.SE3.from_rotation(jaxlie.SO3.from_x_radians(onp.pi))

    # Read metadata: image paths, transformation matrices, FOV.
    with open(dataset_root / f"transforms_{split}.json") as f:
        metadata: dict = json.load(f)

    image_paths: List[pathlib.Path] = []
    transform_matrices: List[onp.ndarray] = []
    for frame in metadata["frames"]:
        # Expected keys in each frame. TODO: what is rotation?
        assert frame.keys() == {"file_path", "rotation", "transform_matrix"}

        image_paths.append(dataset_root / f"{frame['file_path']}.png")
        transform_matrices.append(
            onp.array(frame["transform_matrix"], dtype=onp.float32)
        )
        assert transform_matrices[-1].shape == (4, 4)  # Should be in SE(3).
    fov_x_radians: float = metadata["camera_angle_x"]
    del metadata

    # Parallelized image loading + data preprocessing.
    out = []
    for image, transform_matrix in zip(
        _threaded_image_fetcher(image_paths),
        _optional_tqdm(
            transform_matrices,
            enable=progress_bar,
            desc=f"Loading {dataset_root.stem}",
        ),
    ):
        assert image.dtype == onp.uint8
        height, width = image.shape[:2]

        # Note that this is RGBA!
        assert image.shape == (height, width, 4)

        # [0, 255] => [0, 1]
        image = (image / 255.0).astype(onp.float32)

        # Compute extrinsics.
        T_world_blendercam = jaxlie.SE3.from_matrix(transform_matrix)
        T_camera_world = (T_world_blendercam @ T_blendercam_camera).inverse()
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


T = TypeVar("T", bound=Iterable)


def _optional_tqdm(iterable: T, enable: bool, desc: str = "") -> T:
    """Wraps an iterable with tqdm if `enable=True`."""
    return tqdm(iterable, desc) if enable else iterable  # type: ignore


def _threaded_image_fetcher(paths: Iterable[pathlib.Path]) -> Iterable[onp.ndarray]:
    """Maps an iterable over image paths to an iterable over image arrays, which are
    opened via PIL.

    Helpful for parallelizing IO."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for image in executor.map(
            lambda p: onp.array(PIL.Image.open(p)),
            paths,
            chunksize=4,
        ):
            yield image


if __name__ == "__main__":
    views = load_blender_dataset(
        pathlib.Path("./data/nerf_synthetic/lego/"), "train", True
    )
    rays = rendered_rays_from_views(views)
    print(rays.get_batch_axes())
