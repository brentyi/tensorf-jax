from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import json
import pathlib
from typing import Any, Dict, Iterable, List, Literal, Protocol, TypeVar

import cv2
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import PIL.Image
from jax import numpy as jnp
from optax._src.alias import transform
from tqdm.auto import tqdm
from typing_extensions import Annotated, assert_never

from . import cameras


class NerfDataset(Protocol):
    def get_training_rays(self) -> RenderedRays:
        ...

    def get_cameras(self) -> List[cameras.Camera]:
        ...


def make_dataset(
    dataset_type: Literal["blender", "nerfstudio"],
    dataset_root: pathlib.Path,
    scene_scale: float,
) -> NerfDataset:
    if dataset_type == "blender":
        assert scene_scale == 1.0
        return BlenderDataset(dataset_root)
    elif dataset_type == "nerfstudio":
        return NerfstudioDataset(dataset_root, scene_scale)
    else:
        assert_never(dataset_type)


@dataclasses.dataclass(frozen=True)
class NerfstudioDataset:
    dataset_root: pathlib.Path
    scene_scale: float

    def get_training_rays(self) -> RenderedRays:
        metadata = self._get_metadata()

        camera_model = metadata["camera_model"]
        if metadata["camera_model"] == "OPENCV":
            dist_coeffs = onp.array(
                [metadata.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")]
            )
        elif metadata["camera_model"] == "OPENCV_FISHEYE":
            dist_coeffs = onp.array([metadata[k] for k in ("k1", "k2", "k3", "k4")])
        else:
            assert False, f"Unsupported camera model {metadata['camera_model']}."

        image_paths = tuple(
            map(
                lambda frame: self.dataset_root / frame["file_path"],
                metadata["frames"],
            )
        )
        out: List[RenderedRays] = []
        for i, image, camera in zip(
            range(len(image_paths)),
            _threaded_image_fetcher(image_paths),
            tqdm(
                self.get_cameras(),
                desc=f"Loading {self.dataset_root.stem}",
            ),
        ):
            h, w = image.shape[:2]
            orig_h, orig_w = h, w

            # Resize image to an arbitrary target dimension.
            target_pixels = 1200 * 1600
            scale = 1.0
            if h * w > target_pixels:
                scale = onp.sqrt(target_pixels / (h * w))
                h = int(h * scale)
                w = int(w * scale)
                image = cv2.resize(image, (w, h))

            image = (image / 255.0).astype(onp.float32)

            # (2, w, h) => (h, w, 2) => (h * w, 2)
            orig_image_points = (
                onp.mgrid[:w, :h].T.reshape((h * w, 2))
                / onp.array([h, w])
                * onp.array([orig_h, orig_w])
            )

            if camera_model == "OPENCV":
                ray_directions = cv2.undistortPoints(
                    src=orig_image_points,
                    cameraMatrix=camera.K,
                    distCoeffs=dist_coeffs,
                ).squeeze(axis=1)
            elif camera_model == "OPENCV_FISHEYE":
                ray_directions = cv2.fisheye.undistortPoints(
                    distorted=orig_image_points[:, None, :],
                    K=camera.K,
                    D=dist_coeffs,
                ).squeeze(axis=1)
            else:
                assert False

            assert ray_directions.shape == (h * w, 2)
            ray_directions = onp.concatenate(
                [ray_directions, onp.ones((h * w, 1))], axis=-1
            )
            ray_directions /= onp.linalg.norm(ray_directions, axis=-1, keepdims=True)
            assert ray_directions.shape == (h * w, 3)

            T_world_camera = camera.T_camera_world.inverse()
            out.append(
                RenderedRays(
                    colors=image.reshape((-1, 3)),  # type: ignore
                    rays_wrt_world=cameras.Rays3D(
                        origins=onp.tile(  # type: ignore
                            T_world_camera.translation()[None, :], (h * w, 1)
                        ),
                        directions=ray_directions @ onp.array(T_world_camera.rotation().as_matrix().T),  # type: ignore
                        camera_indices=onp.full(
                            shape=(h * w),
                            fill_value=i,
                            dtype=onp.uint32,
                        ),  # type: ignore
                    ),
                )
            )

        return jax.tree_map(lambda *leaves: onp.concatenate(leaves, axis=0), *out)

    def get_cameras(self) -> List[cameras.Camera]:
        # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
        # convention.
        T_blendercam_camera = jaxlie.SE3.from_rotation(
            jaxlie.SO3.from_x_radians(onp.pi)
        )

        metadata = self._get_metadata()

        transform_matrices: List[onp.ndarray] = []
        for frame in metadata["frames"]:
            # Expected keys in each frame.
            assert frame.keys() == {"file_path", "transform_matrix"}

            transform_matrices.append(
                onp.array(frame["transform_matrix"], dtype=onp.float32)
            )
            assert transform_matrices[-1].shape == (4, 4)  # Should be in SE(3).

        camera_matrix = onp.eye(3)
        camera_matrix[0, 0] = metadata["fl_x"]
        camera_matrix[1, 1] = metadata["fl_y"]
        camera_matrix[0, 2] = metadata["cx"]
        camera_matrix[1, 2] = metadata["cy"]

        # Compute pose bounding box.
        positions = onp.array(transform_matrices)[:, :3, 3]
        aabb_min = positions.min(axis=0)
        aabb_max = positions.max(axis=0)
        del positions
        assert aabb_min.shape == aabb_max.shape == (3,)

        out = []
        for transform_matrix in transform_matrices:
            # Center and scale scene. In the future we might also auto-orient it.
            transform_matrix = transform_matrix.copy()
            transform_matrix[:3, 3] -= aabb_min
            transform_matrix[:3, 3] -= (aabb_max - aabb_min) / 2.0
            transform_matrix[:3, 3] /= (aabb_max - aabb_min).max() / 2.0
            transform_matrix[:3, 3] *= self.scene_scale

            # Compute extrinsics.
            T_world_blendercam = jaxlie.SE3.from_matrix(transform_matrix)
            T_camera_world = (T_world_blendercam @ T_blendercam_camera).inverse()

            out.append(
                cameras.Camera(
                    K=camera_matrix,  # type: ignore
                    T_camera_world=T_camera_world,
                    image_width=metadata["w"],
                    image_height=metadata["h"],
                )
            )

        return out

    def _get_metadata(self) -> Dict[str, Any]:
        """Read metadata: image paths, transformation matrices, FOV."""
        with open(self.dataset_root / f"transforms.json") as f:
            metadata: dict = json.load(f)
        return metadata


@dataclasses.dataclass(frozen=True)
class BlenderDataset:
    dataset_root: pathlib.Path

    def get_training_rays(self) -> RenderedRays:
        return rendered_rays_from_views(self._registered_views)

    def get_cameras(self) -> List[cameras.Camera]:
        return [v.camera for v in self._registered_views]

    @functools.cached_property
    def _registered_views(self) -> List[RegisteredRgbaView]:
        metadata = self._get_metadata()

        image_paths: List[pathlib.Path] = []
        transform_matrices: List[onp.ndarray] = []
        for frame in metadata["frames"]:
            assert frame.keys() == {"file_path", "rotation", "transform_matrix"}

            image_paths.append(self.dataset_root / f"{frame['file_path']}.png")
            transform_matrices.append(
                onp.array(frame["transform_matrix"], dtype=onp.float32)
            )
            assert transform_matrices[-1].shape == (4, 4)  # Should be in SE(3).
        fov_x_radians: float = metadata["camera_angle_x"]
        del metadata

        # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
        # convention.
        T_blendercam_camera = jaxlie.SE3.from_rotation(
            jaxlie.SO3.from_x_radians(onp.pi)
        )

        out = []
        for image, transform_matrix in zip(
            _threaded_image_fetcher(image_paths),
            tqdm(
                transform_matrices,
                desc=f"Loading {self.dataset_root.stem}",
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

    def _get_metadata(self) -> Dict[str, Any]:
        """Read metadata: image paths, transformation matrices, FOV."""
        split = "train"
        with open(self.dataset_root / f"transforms_{split}.json") as f:
            metadata: dict = json.load(f)
        return metadata


# Helpers.


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
    for i, view in enumerate(views):
        height = view.camera.image_height
        width = view.camera.image_width

        rays = view.camera.pixel_rays_wrt_world(camera_index=i)
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
                    camera_indices=rays.camera_indices.reshape((-1,)),
                ),
            )
        )

    out_concat: RenderedRays = jax.tree_map(
        lambda *children: onp.concatenate(children, axis=0), *out
    )

    # Shape of rays should (N,3), colors should be (N,4), etc.
    assert len(out_concat.rays_wrt_world.get_batch_axes()) == 1
    return out_concat


T = TypeVar("T", bound=Iterable)


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
