from __future__ import annotations

import dataclasses
import enum
import functools
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
    density_tensor: tensor_vm.TensorVM


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
    rays_wrt_world = jax.tree_map(lambda x: x.reshape((-1, 3)), rays_wrt_world)
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


@functools.partial(jax.jit, static_argnames=("appearance_mlp", "config", "dtype"))
def render_rays(
    appearance_mlp: networks.FeatureMlp,
    learnable_params: LearnableParams,
    aabb: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    prng_key: jax.random.KeyArray,
    config: RenderConfig,
    *,
    dtype: Any = jnp.float32,
) -> jnp.ndarray:
    """Render a set of rays.

    Output should have shape `(ray_count, 3)`."""

    # Cast everything to the desired dtype.
    learnable_params, aabb, rays_wrt_world = jax.tree_map(
        lambda x: x.astype(dtype), (learnable_params, aabb, rays_wrt_world)
    )

    (ray_count,) = rays_wrt_world.get_batch_axes()

    sample_prng_key, render_rgb_prng_key = jax.random.split(prng_key)

    points, ts, step_sizes = jax.vmap(
        lambda ray: sample_points_along_ray(
            ray_wrt_world=ray,
            aabb=aabb,
            samples_per_ray=config.density_samples_per_ray,
            prng_key=sample_prng_key,
        ),
        out_axes=(1, 0, 0),
    )(rays_wrt_world)
    assert points.shape == (3, ray_count, config.density_samples_per_ray)
    assert ts.shape == (ray_count, config.density_samples_per_ray)
    assert step_sizes.shape == (ray_count,)

    # Normalize points to [-1, 1].
    points = (
        (points - aabb[0][:, None, None]) / (aabb[1] - aabb[0])[:, None, None] - 0.5
    ) * 2.0
    assert points.shape == (3, ray_count, config.density_samples_per_ray)

    # Pull interpolated density features out of tensor decomposition.
    density_feat = learnable_params.density_tensor.interpolate(points)
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


@jdc.pytree_dataclass
class SegmentProbabilities(jdc.EnforcedAnnotationsMixin):
    p_exits: Annotated[jnp.ndarray, (), jnp.floating]
    """P(ray exits segment s).

    Note that this also implies that the ray has exited (and thus entered) all previous
    segments."""

    p_terminates: Annotated[jnp.ndarray, (), jnp.floating]
    """P(ray terminates at s, ray exits s - 1).

    For a ray to terminate in a segment, it must first pass through (and 'exit') all
    previous segments."""


def compute_segment_probabilities(
    sigmas: jnp.ndarray, step_size: jnp.ndarray
) -> SegmentProbabilities:
    r"""Compute some probabilities needed for rendering rays. Expects sigmas of shape
    (*, sample_count) and a per-ray step size of shape (*,).

    Each of the ray segments we're rendering is broken up into samples. We can treat the
    densities as piecewise constant and use an exponential distribution and compute:

      1. P(ray exits s) = exp(\sum_{i=1}^s -(sigma_i * l_i)
      2. P(ray terminates in s | ray exits s-1) = 1.0 - exp(-sigma_s * l_s)
      3. P(ray terminates in s, ray exits s-1)
         = P(ray terminates at s | ray exits s-1) * P(ray exits s-1)

    where l_i is the length of segment i.
    """

    # Support arbitrary leading batch axes.
    (*batch_axes, sample_count) = sigmas.shape
    assert step_size.shape == (*batch_axes,)

    # Equation 1.
    neg_scaled_sigmas = -sigmas * step_size[..., None]
    p_exits = jnp.exp(jnp.cumsum(neg_scaled_sigmas, axis=-1))
    assert p_exits.shape == (*batch_axes, sample_count)

    # Equation 2. Not used outside of this function, and not returned.
    p_terminates_given_exits_prev = 1.0 - jnp.exp(neg_scaled_sigmas)
    assert p_terminates_given_exits_prev.shape == (*batch_axes, sample_count)

    # Equation 3.
    p_terminates = jnp.multiply(
        p_terminates_given_exits_prev,
        # We prepend 1 because the ray is always alive initially.
        jnp.concatenate(
            [
                jnp.ones((*batch_axes, 1), dtype=neg_scaled_sigmas.dtype),
                p_exits[..., :-1],
            ],
            axis=-1,
        ),
    )
    assert p_terminates.shape == (*batch_axes, sample_count)

    return SegmentProbabilities(
        p_exits=p_exits,
        p_terminates=p_terminates,
    )


def sample_points_along_ray(
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
    segment = ray_segment_from_bounding_box(
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
    return points, ts, step_size


@jdc.pytree_dataclass
class RaySegmentSpecification:
    t_min: jnp.ndarray
    t_max: jnp.ndarray


def ray_segment_from_bounding_box(
    ray_wrt_world: cameras.Rays3D,
    aabb: jnp.ndarray,
    min_segment_length: float,
) -> RaySegmentSpecification:
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

    return RaySegmentSpecification(
        t_min=jnp.where(valid_mask, t_min, 0.0),
        t_max=jnp.where(valid_mask, t_max_clipped, min_segment_length),
    )


def _rgb_from_points(
    rays_wrt_world: cameras.Rays3D,
    probs: SegmentProbabilities,
    learnable_params: LearnableParams,
    points: jnp.ndarray,
    appearance_mlp: networks.FeatureMlp,
    config: RenderConfig,
    prng_key: jax.random.KeyArray,
    dtype: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Helper for rendering RGB values. Returns an RGB array of shape `(ray_count,
    config.samples_per_ray, 3)`, and an unbiasing coefficient array of shape
    `(ray_count,)`.

    The original PyTorch implementation speeds up training by only rendering RGB values
    for which the termination probability (sample weight) exceeds a provided threshold,
    but this requires some boolean masking and dynamic shapes which, alas, are quite
    difficult in JAX. To reduce the number of appearance computations needed, we instead
    resort to a weighted sampling approach.
    """
    ray_count = points.shape[1]
    assert points.shape == (3, ray_count, config.density_samples_per_ray)

    # Render the most visible points for each ray, with weighted random sampling.
    assert probs.p_terminates.shape == (ray_count, config.density_samples_per_ray)
    appearance_indices = jax.vmap(
        lambda p: jax.random.choice(
            key=prng_key,
            a=config.density_samples_per_ray,
            shape=(config.appearance_samples_per_ray,),
            replace=False,
            p=p,
        )
    )(probs.p_terminates)
    assert appearance_indices.shape == (ray_count, config.appearance_samples_per_ray)

    visible_points = points[:, jnp.arange(ray_count)[:, None], appearance_indices]
    assert visible_points.shape == (3, ray_count, config.appearance_samples_per_ray)

    appearance_tensor = learnable_params.appearance_tensor
    appearance_feat = appearance_tensor.interpolate(visible_points)
    assert appearance_feat.shape == (
        appearance_tensor.channel_dim(),
        ray_count,
        config.appearance_samples_per_ray,
    )

    total_sample_count = ray_count * config.appearance_samples_per_ray
    appearance_feat = jnp.moveaxis(appearance_feat, 0, -1).reshape(
        (total_sample_count, appearance_tensor.channel_dim())
    )
    viewdirs = jnp.tile(
        rays_wrt_world.directions[:, None, :],
        (1, config.appearance_samples_per_ray, 1),
    ).reshape((-1, 3))

    visible_rgb = cast(
        jnp.ndarray,
        appearance_mlp.apply(
            learnable_params.appearance_mlp_params,
            features=appearance_feat.reshape(
                (ray_count * config.appearance_samples_per_ray, -1)
            ),
            viewdirs=viewdirs,
            dtype=dtype,
        ),
    ).reshape((ray_count, config.appearance_samples_per_ray, 3))

    rgb = (
        jnp.zeros(
            (ray_count, config.density_samples_per_ray, 3),
            dtype=dtype,
        )
        .at[jnp.arange(ray_count)[:, None], appearance_indices, :]
        .set(visible_rgb)
    )
    assert rgb.shape == (ray_count, config.density_samples_per_ray, 3)

    # Coefficients for unbiasing the expected RGB values using the sampling
    # probabilities. This is helpful because RGB values for points that are not chosen
    # by our appearance sampler are zeroed out.
    #
    # As an example: if the weights* for all density samples is 0.95** but the sum of
    # weights for our appearance samples is only 0.7, we can correct the resulting
    # expected RGB value by scaling by (0.95/0.7).
    #
    # *weight at a segment = termination probability at that segment
    # **equivalently: p=0.05 of the ray exiting the last segment and rendering the
    # background.
    sampled_p_terminates = probs.p_terminates[
        jnp.arange(ray_count)[:, None], appearance_indices
    ]
    assert sampled_p_terminates.shape == (
        ray_count,
        config.appearance_samples_per_ray,
    )

    unbias_coeff = (
        # The 0.95 term in the example.
        1.0
        - probs.p_exits[:, -1]
        + utils.eps_from_dtype(dtype)
    ) / (
        # The 0.7 term in the example.
        jnp.sum(sampled_p_terminates, axis=1)
        + utils.eps_from_dtype(dtype)
    )
    assert unbias_coeff.shape == (ray_count,)

    return rgb, unbias_coeff
