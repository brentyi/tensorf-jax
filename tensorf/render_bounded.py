from typing import Any, Optional, Tuple, cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from . import cameras, networks, render_common, utils


def render_rays(
    appearance_mlp: jdc.Static[networks.FeatureMlp],
    learnable_params: render_common.LearnableParams,
    aabb: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    prng_key: jax.random.KeyArray,
    config: jdc.Static[render_common.RenderConfig],
    *,
    dtype: jdc.Static[Any] = jnp.float32,
) -> jnp.ndarray:
    """Render a set of rays.

    Output should have shape `(ray_count, 3)`."""

    # For bounded scenes, we match the original TensoRF implementation and support only
    # one density tensor.
    assert len(learnable_params.density_tensors) == 1
    assert len(config.density_samples_per_ray) == 1
    (density_samples_per_ray,) = config.density_samples_per_ray

    # Cast everything to the desired dtype.
    learnable_params, aabb, rays_wrt_world = jax.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(jnp.floating, dtype) else x,
        (learnable_params, aabb, rays_wrt_world),
    )
    (ray_count,) = rays_wrt_world.get_batch_axes()

    sample_prng_key, render_rgb_prng_key = jax.random.split(prng_key)

    # Bounded scene: we sample points uniformly between the camera origin and bounding
    # box limit.
    points, ts, step_sizes = jax.vmap(
        lambda ray: _sample_points_along_ray_within_bbox(
            ray_wrt_world=ray,
            aabb=aabb,
            samples_per_ray=density_samples_per_ray,
            prng_key=sample_prng_key,
        ),
        out_axes=(1, 0, 0),
    )(rays_wrt_world)
    step_sizes = jnp.tile(step_sizes[:, None], (1, density_samples_per_ray))

    assert points.shape == (3, ray_count, density_samples_per_ray)
    assert ts.shape == (ray_count, density_samples_per_ray)
    assert step_sizes.shape == (ray_count, density_samples_per_ray)

    # Normalize points to [-1, 1].
    points = (
        (points - aabb[0][:, None, None]) / (aabb[1] - aabb[0])[:, None, None] - 0.5
    ) * 2.0
    assert points.shape == (3, ray_count, density_samples_per_ray)

    # Pull interpolated density features out of tensor decomposition.
    density_feat = learnable_params.density_tensors[0].interpolate(points)
    assert density_feat.shape == (
        density_feat.shape[0],
        ray_count,
        density_samples_per_ray,
    )

    # Density from features.
    prerectified_sigmas = jnp.sum(density_feat, axis=0)
    assert prerectified_sigmas.shape == (ray_count, density_samples_per_ray)

    # Compute segment probabilities for each ray.
    probs = render_common.SegmentProbabilities.compute(
        prerectified_sigmas,
        step_sizes,
        ts,
        near=config.near,
        far=config.far,
    )
    assert (
        probs.get_batch_axes()
        == probs.p_exits.shape
        == probs.p_terminates.shape
        == (ray_count, density_samples_per_ray)
    )

    if config.mode is render_common.RenderMode.RGB:
        #  The original PyTorch implementation speeds up training by only rendering RGB
        #  values for which the termination probability (sample weight) exceeds a
        #  threshold, but this requires some boolean masking and dynamic shapes which,
        #  alas, are quite difficult in JAX. To reduce the number of appearance
        #  computations needed, we instead resort to a weighted sampling approach.

        render_probs, render_indices = probs.resample_subset(
            num_samples=config.appearance_samples_per_ray,
            prng=render_rgb_prng_key,
        )
        assert render_probs.get_batch_axes() == (
            ray_count,
            config.appearance_samples_per_ray,
        )

        render_points = jnp.take_along_axis(
            arr=points, indices=render_indices[None, :, :], axis=-1
        )
        assert render_points.shape == (3, ray_count, config.appearance_samples_per_ray)

        mlp_out = render_common.get_appearance_mlp_out(
            appearance_mlp,
            learnable_params,
            render_points,
            rays_wrt_world,
        )
        rgb_per_point = render_common.direct_rgb_from_mlp(mlp_out)
        return render_common.render_from_mlp_out(
            rgb_per_point,
            render_probs,
            render_common.RenderMode.RGB,
        )

    else:
        return probs.render_distance(config.mode)


def _sample_points_along_ray_within_bbox(
    ray_wrt_world: cameras.Rays3D,
    aabb: jnp.ndarray,
    samples_per_ray: int,
    prng_key: Optional[jax.random.KeyArray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return points along a ray.

    Outputs are:
    - Points of shape `(3, samples_per_ray)`.
    - Distances from the ray origin of shape `(samples_per_ray,)`.
    - A scalar step size."""
    assert ray_wrt_world.get_batch_axes() == ()
    assert ray_wrt_world.origins.shape == ray_wrt_world.directions.shape == (3,)

    # Get segment of ray that's within the bounding box.
    segment = _ray_segment_from_bounding_box(
        ray_wrt_world, aabb=aabb, min_segment_length=1e-3
    )
    step_size = (segment.t_max - segment.t_min) / samples_per_ray

    # Get sample points along ray.
    ts = jnp.arange(samples_per_ray)
    if prng_key is not None:
        # Jitter if a PRNG key is passed in.
        ts = ts + jax.random.uniform(
            key=prng_key,
            shape=ts.shape,
            dtype=step_size.dtype,
        )
    ts = ts * step_size
    ts = segment.t_min + ts

    # That's it!
    points = (
        ray_wrt_world.origins[:, None] + ray_wrt_world.directions[:, None] * ts[None, :]
    )
    assert points.shape == (3, samples_per_ray)
    assert ts.shape == (samples_per_ray,)
    assert step_size.shape == ()
    return points, ts, step_size


@jdc.pytree_dataclass
class _RaySegmentSpecification:
    t_min: jnp.ndarray
    t_max: jnp.ndarray


def _ray_segment_from_bounding_box(
    ray_wrt_world: cameras.Rays3D,
    aabb: jnp.ndarray,
    min_segment_length: float,
) -> _RaySegmentSpecification:
    """Given a ray and bounding box, compute the near and far t values that define a
    segment that lies fully in the box."""
    assert ray_wrt_world.origins.shape == ray_wrt_world.directions.shape == (3,)
    assert aabb.shape == (2, 3)

    # Find t for per-axis collision with the bounding box.
    #     origin + t * direction = bounding box
    #     t = (bounding box - origin) / direction
    offsets = aabb - ray_wrt_world.origins[None, :]
    t_intersections = offsets / (
        ray_wrt_world.directions + utils.eps_from_dtype(offsets.dtype)
    )

    # Compute near/far distances.
    t_min_per_axis = jnp.min(t_intersections, axis=0)
    t_max_per_axis = jnp.max(t_intersections, axis=0)
    assert t_min_per_axis.shape == t_max_per_axis.shape == (3,)

    # Clip.
    t_min = jnp.maximum(0.0, jnp.max(t_min_per_axis))
    t_max = jnp.min(t_max_per_axis)
    t_max_clipped = jnp.maximum(t_max, t_min + min_segment_length)

    # TODO: this should likely be returned as well, and used as a mask for supervision.
    # Currently our loss includes rays outside of the bounding box.
    valid_mask = t_min < t_max

    return _RaySegmentSpecification(
        t_min=jnp.where(valid_mask, t_min, 0.0),
        t_max=jnp.where(valid_mask, t_max_clipped, min_segment_length),
    )
