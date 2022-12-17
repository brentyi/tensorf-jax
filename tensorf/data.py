import concurrent.futures
import json
import pathlib
from typing import Iterable, List, Literal, TypeVar

import cv2
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import PIL.Image
from jax import numpy as jnp
from optax._src.alias import transform
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


def load_nerfstudio_dataset(
    dataset_root: pathlib.Path,
    progress_bar: bool = False,
) -> List[RegisteredRgbaView]:
    """Load a NeRF training dataset as a list of registered views of a scene.

    Naive. Attempts to load everything into memory at once, ignores multiple scales, etc.

    Note: this is mostly the same as the blender dataset loading below. Could refactor."""

    # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
    # convention.
    T_blendercam_camera = jaxlie.SE3.from_rotation(jaxlie.SO3.from_x_radians(onp.pi))

    # Read metadata: image paths, transformation matrices, FOV.
    with open(dataset_root / f"transforms.json") as f:
        metadata: dict = json.load(f)

    image_paths: List[pathlib.Path] = []
    transform_matrices: List[onp.ndarray] = []
    for frame in metadata["frames"]:
        # Expected keys in each frame.
        assert frame.keys() == {"file_path", "transform_matrix"}

        image_paths.append(dataset_root / frame["file_path"])
        transform_matrices.append(
            onp.array(frame["transform_matrix"], dtype=onp.float32)
        )
        assert transform_matrices[-1].shape == (4, 4)  # Should be in SE(3).

    camera_matrix = onp.eye(3)
    camera_matrix[0, 0] = metadata["fl_x"]
    camera_matrix[1, 1] = metadata["fl_y"]
    camera_matrix[0, 2] = metadata["cx"]
    camera_matrix[1, 2] = metadata["cy"]
    dist_coeffs = onp.array([metadata[k] for k in ("k1", "k2", "p1", "p2")])

    new_camera_matrix, _new_size = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (metadata["w"], metadata["h"]), alpha=0.0
    )
    del metadata

    # Compute pose bounding box.
    positions = onp.array(transform_matrices)[:, :3, 3]
    aabb_min = positions.min(axis=0)
    aabb_max = positions.max(axis=0)
    del positions
    assert aabb_min.shape == aabb_max.shape == (3,)

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

        # Note that this is RGB!
        assert image.shape == (*image.shape[:2], 3)

        # Center and scale scene. In the future we might also auto-orient it.
        transform_matrix = transform_matrix.copy()
        transform_matrix[:3, 3] -= aabb_min
        transform_matrix[:3, 3] -= (aabb_max - aabb_min) / 2.0
        transform_matrix[:3, 3] /= (aabb_max - aabb_min).max() / 2.0

        # Compute extrinsics.
        T_world_blendercam = jaxlie.SE3.from_matrix(transform_matrix)
        T_camera_world = (T_world_blendercam @ T_blendercam_camera).inverse()

        image_undistorted = cv2.undistort(
            image, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # [0, 255] => [0, 1]
        image_undistorted = (image_undistorted / 255.0).astype(onp.float32)

        image_undistorted_rgba = onp.concatenate(
            [image_undistorted, onp.ones((*image.shape[:2], 1))], axis=-1
        )
        assert image_undistorted_rgba.shape == (*image.shape[:2], 4)
        height, width = image_undistorted.shape[:2]

        out.append(
            RegisteredRgbaView(
                image_rgba=image_undistorted_rgba,  # type: ignore
                camera=cameras.Camera(
                    K=new_camera_matrix,  # type: ignore
                    T_camera_world=T_camera_world,
                    image_width=width,
                    image_height=height,
                ),
            )
        )

    return out


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
