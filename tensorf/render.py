from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Optional, Tuple, cast

import flax
import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated

from . import cameras, networks, tensor_vm, utils


class RenderMode(enum.Enum):
    # Note: we currently only support rendering distances from the camera origin, which
    # is a bit different from depth. (the latter is typically a local Z value)
    RGB = enum.auto()
    DIST_MEDIAN = enum.auto()
    DIST_MEAN = enum.auto()


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    near: float
    far: float
    mode: RenderMode

    density_samples_per_ray: int
    """Number of points to sample densities at."""

    appearance_samples_per_ray: int
    """Number of points to sample appearances at."""


@jdc.pytree_dataclass
class LearnableParams:
    """Structure containing learnable parameters required for rendering."""

    appearance_mlp_params: flax.core.FrozenDict
    appearance_tensor: tensor_vm.TensorVM
    density_tensors: Tuple[tensor_vm.TensorVM, ...]
    scene_contraction: jdc.Static[bool]


def render_rays_batched(
    appearance_mlp: networks.FeatureMlp,
    learnable_params: LearnableParams,
    aabb: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    prng_key: Optional[jax.random.KeyArray],
    config: RenderConfig,
    *,
    batch_size: int = 4096,
    use_tqdm: bool = True,
) -> onp.ndarray:
    """Render rays. Supports arbitrary batch axes (helpful for inputs with both height
    and width leading axes), and automatically splits rays into batches to prevent
    out-of-memory errors.

    Possibly this could just take the training state directly as input."""
    batch_axes = rays_wrt_world.get_batch_axes()
    rays_wrt_world = (
        cameras.Rays3D(  # TODO: feels like this could be done less manually!
            origins=rays_wrt_world.origins.reshape((-1, 3)),
            directions=rays_wrt_world.directions.reshape((-1, 3)),
            camera_indices=rays_wrt_world.camera_indices.reshape((-1,)),
        )
    )
    (total_rays,) = rays_wrt_world.get_batch_axes()

    processed = 0
    out = []

    for i in (tqdm if use_tqdm else lambda x: x)(
        range(math.ceil(total_rays / batch_size))
    ):
        batch = jax.tree_map(
            lambda x: x[processed : min(total_rays, processed + batch_size)],
            rays_wrt_world,
        )
        processed += batch_size
        out.append(
            render_rays(
                appearance_mlp=appearance_mlp,
                learnable_params=learnable_params,
                aabb=aabb,
                rays_wrt_world=batch,
                prng_key=prng_key,
                config=config,
            )
        )
    out_concatenated = onp.concatenate(out, axis=0)

    # Reshape, with generalization to both (*,) for depths and (*, 3) for RGB.
    return out_concatenated.reshape(batch_axes + out_concatenated.shape[1:])


@jdc.jit
def render_rays(
    appearance_mlp: jdc.Static[networks.FeatureMlp],
    learnable_params: LearnableParams,
    aabb: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    prng_key: jax.random.KeyArray,
    config: jdc.Static[RenderConfig],
    *,
    dtype: jdc.Static[Any] = jnp.float32,
) -> jnp.ndarray:
    """Render a set of rays.

    Output should have shape `(ray_count, 3)`."""

    # Cast everything to the desired dtype.
    learnable_params, aabb, rays_wrt_world = jax.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(jnp.floating, dtype) else x,
        (learnable_params, aabb, rays_wrt_world),
    )

    (ray_count,) = rays_wrt_world.get_batch_axes()

    sample_prng_key, render_rgb_prng_key = jax.random.split(prng_key)

    if learnable_params.scene_contraction:
        # Contracted scene: sample linearly for close samples, then start spacing
        # samples out.
        #
        # An occupancy grid or proposal network would really help us here!
        close_samples_per_ray = config.density_samples_per_ray // 2
        far_samples_per_ray = config.density_samples_per_ray - close_samples_per_ray

        close_ts = jnp.linspace(config.near, config.near + 1.0, close_samples_per_ray)

        # Some heuristics for sampling far points, which should be close to sampling
        # linearly in disparity when k=1. This is probably reasonable, but it'd be a
        # good idea to look at what real NeRF codebases do.
        far_start = config.near + 1.0 + 1.0 / close_samples_per_ray
        k = 10.0
        far_deltas = (
            1.0
            / (
                1.0
                - onp.linspace(  # onp here is important for float64.
                    0.0,
                    1.0 - 1 / ((config.far - far_start) / k + 1),
                    far_samples_per_ray,
                )
            )
            - 1.0
        ) * onp.linspace(1.0, k, far_samples_per_ray)
        far_ts = far_start + far_deltas

        ts = jnp.tile(jnp.concatenate([close_ts, far_ts])[None, :], reps=(ray_count, 1))

        # Compute step sizes.
        step_sizes = jnp.roll(ts, -1, axis=-1) - ts  # Naive. Could be improved
        step_sizes = step_sizes.at[:, -1].set(step_sizes[:, -2])

        # Jitter samples.
        sample_jitter = jax.random.uniform(
            sample_prng_key, shape=(ray_count, config.density_samples_per_ray)
        )
        ts = ts + step_sizes * sample_jitter

        # Compute points in world space.
        points = (
            rays_wrt_world.origins[:, None, :]
            + ts[:, :, None] * rays_wrt_world.directions[:, None, :]
        )

        # Contract points to cube.
        norm = jnp.linalg.norm(points, ord=jnp.inf, axis=-1, keepdims=True)
        points = jnp.where(norm <= 1.0, points, (2.0 - 1.0 / norm) * points / norm)
        assert points.shape == (ray_count, config.density_samples_per_ray, 3)
        points = jnp.moveaxis(points, -1, 0)
    else:
        # Bounded scene: we sample points uniformly between the camera origin and bounding
        # box limit.
        points, ts, step_sizes = jax.vmap(
            lambda ray: sample_points_along_ray_within_bbox(
                ray_wrt_world=ray,
                aabb=aabb,
                samples_per_ray=config.density_samples_per_ray,
                prng_key=sample_prng_key,
            ),
            out_axes=(1, 0, 0),
        )(rays_wrt_world)
        step_sizes = jnp.tile(step_sizes[:, None], (1, config.density_samples_per_ray))

    assert points.shape == (3, ray_count, config.density_samples_per_ray)
    assert ts.shape == (ray_count, config.density_samples_per_ray)
    assert step_sizes.shape == (ray_count, config.density_samples_per_ray)

    # Normalize points to [-1, 1].
    points = (
        (points - aabb[0][:, None, None]) / (aabb[1] - aabb[0])[:, None, None] - 0.5
    ) * 2.0
    assert points.shape == (3, ray_count, config.density_samples_per_ray)

    # Pull interpolated density features out of tensor decomposition.
    density_feat = learnable_params.density_tensors.interpolate(points)
    assert density_feat.shape == (
        density_feat.shape[0],
        ray_count,
        config.density_samples_per_ray,
    )

    # Density from features.
    sigmas = jax.nn.softplus(jnp.sum(density_feat, axis=0) + 10.0)
    assert sigmas.shape == (ray_count, config.density_samples_per_ray)

    # Compute segment probabilities for each ray.
    probs = compute_segment_probabilities(sigmas, step_sizes)
    assert (
        probs.get_batch_axes()
        == probs.p_exits.shape
        == probs.p_terminates.shape
        == (ray_count, config.density_samples_per_ray)
    )

    if config.mode is RenderMode.RGB:
        # Get RGB array.
        rgb, unbias_coeff = _rgb_from_points(
            rays_wrt_world=rays_wrt_world,
            probs=probs,
            learnable_params=learnable_params,
            points=points,
            appearance_mlp=appearance_mlp,
            config=config,
            prng_key=render_rgb_prng_key,
            dtype=dtype,
        )
        assert rgb.shape == (ray_count, config.density_samples_per_ray, 3)
        assert unbias_coeff.shape == (ray_count,)

        # No need to backprop through the unbiasing coefficient! This can also cause
        # instability in mixed-precision mode.
        unbias_coeff = jax.lax.stop_gradient(unbias_coeff)

        # One thing I don't have intuition for: is there something special about RGB
        # that makes this weighted average/expected value meaningful? Is this
        # because RGB is additive? Can we just do this with any random color space?
        expected_rgb = (
            jnp.sum(rgb * probs.p_terminates[:, :, None], axis=-2)
            * unbias_coeff[:, None]
        )
        assert expected_rgb.shape == (ray_count, 3)

        # Add white background.
        assert probs.p_exits.shape == (ray_count, config.density_samples_per_ray)
        background_color = jnp.ones(3, dtype=dtype)
        expected_rgb_with_background = (
            expected_rgb + probs.p_exits[:, -1:] * background_color
        )
        assert expected_rgb_with_background.shape == (ray_count, 3)
        return expected_rgb_with_background

    elif config.mode is RenderMode.DIST_MEDIAN:
        # Compute depth via median.
        sample_distances = jnp.concatenate(
            [ts, jnp.full((ray_count, 1), jnp.inf, dtype=dtype)], axis=-1
        )
        p_not_alive_padded = jnp.concatenate(
            [1.0 - probs.p_exits, jnp.ones((ray_count, 1), dtype=dtype)], axis=-1
        )
        assert sample_distances.shape == p_not_alive_padded.shape

        median_mask = p_not_alive_padded > 0.5
        median_mask = (
            jnp.zeros_like(median_mask)
            .at[..., 1:]
            .set(jnp.logical_xor(median_mask[..., :-1], median_mask[..., 1:]))
        )

        # Output is medians.
        depths = jnp.sum(median_mask * sample_distances, axis=-1)
        return depths

    elif config.mode is RenderMode.DIST_MEAN:
        # Compute depth via expected value.
        sample_distances = jnp.concatenate([ts, ts[:, -1:]], axis=-1)
        p_terminates_padded = jnp.concatenate(
            [probs.p_terminates, probs.p_exits[:, -1:]], axis=-1
        )
        assert sample_distances.shape == p_terminates_padded.shape
        return jnp.sum(p_terminates_padded * sample_distances, axis=-1)

    else:
        assert False
