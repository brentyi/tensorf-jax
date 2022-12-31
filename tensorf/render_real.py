from typing import Any, Optional, Tuple, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp

from . import cameras, networks, render_common, tensor_vm, utils


def render_rays(
    appearance_mlp: jdc.Static[networks.FeatureMlp],
    learnable_params: render_common.LearnableParams,
    aabb: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    prng_key: jax.random.KeyArray,
    config: jdc.Static[render_common.RenderConfig],
    *,
    proposal_anneal_factor: float = 1.0,
    dtype: jdc.Static[Any] = jnp.float32,
) -> jnp.ndarray:
    """Render a set of rays.

    Output should have shape `(ray_count, 3)`."""

    num_density_tensors = len(learnable_params.density_tensors)
    assert len(config.density_samples_per_ray) == num_density_tensors

    # Cast everything to the desired dtype.
    learnable_params, aabb, rays_wrt_world = jax.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(jnp.floating, dtype) else x,
        (learnable_params, aabb, rays_wrt_world),
    )
    (ray_count,) = rays_wrt_world.get_batch_axes()

    sample_prng_keys = jax.random.split(prng_key, num=len(num_density_tensors + 1))

    # Sample initial points in world space.
    ts, step_sizes = sample_initial_points_from_rays(
        ray_count,
        num_samples=config.density_samples_per_ray[0],
        near=config.near,
        far=config.far,
        sample_prng_key=sample_prng_keys[0],
    )
    render_points = rays_wrt_world.points_from_ts(ts)
    assert render_points.shape == (ray_count, config.density_samples_per_ray[0], 3)

    # Run proposal networks.
    # [ ] fix resampling.
    # [ ] Supervise proposal networks.
    # [ ] distortion loss?
    # that's it!
    probs = None
    for i in range(num_density_tensors):
        density_feat = interpolate_contracted(
            learnable_params.density_tensors[i], render_points
        )
        prerectified_sigmas = jnp.sum(density_feat, axis=0)
        probs = render_common.SegmentProbabilities.compute(
            prerectified_sigmas, step_sizes=step_sizes, ts=ts
        )
        del prerectified_sigmas
        del density_feat

        probs, indices = probs.stratified_resample(
            num_samples=(
                config.density_samples_per_ray[i + 1]
                if i < num_density_tensors - 1
                else config.appearance_samples_per_ray
            ),
            prng=sample_prng_keys[i + 1],
            anneal_factor=proposal_anneal_factor,
        )
        render_points = jnp.take_along_axis(
            arr=render_points, indices=indices[None, :, :], axis=-1
        )
    assert probs is not None

    appearance_mlp_out = render_common.get_appearance_mlp_out(
        appearance_mlp,
        learnable_params,
        render_points,
        rays_wrt_world,
    )
    return render_common.render_from_mlp_out(
        appearance_mlp_out,
        probs,
        config.mode,
    )


def interpolate_contracted(
    tensor: tensor_vm.TensorVM,
    points: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a feature vector from a VM decomposed tensor, with scene
    contraction."""
    assert points.shape[-1] == 3

    # Contract points to cube.
    norm = jnp.linalg.norm(points, ord=jnp.inf, axis=-1, keepdims=True)
    contracted_points = jnp.where(
        norm <= 1.0, points, (2.0 - 1.0 / norm) * points / norm
    )
    assert isinstance(contracted_points, jnp.ndarray)
    contracted_points = jnp.moveaxis(contracted_points, -1, 0)
    assert contracted_points.shape[0] == 3

    out = tensor.interpolate(contracted_points)
    assert out.shape == (tensor.channel_dim(), *contracted_points.shape[:-1])
    return out


def sample_initial_points_from_rays(
    ray_count: int,
    num_samples: int,
    near: float,
    far: float,
    sample_prng_key: jax.random.KeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Contracted scene: sample linearly for close samples, then start spacing
    # samples out.
    #
    # An occupancy grid or proposal network would really help us here!
    close_samples_per_ray = num_samples // 2
    far_samples_per_ray = num_samples - close_samples_per_ray

    close_ts = jnp.linspace(near, near + 1.0, close_samples_per_ray)

    # Some heuristics for sampling far points, which should be close to sampling
    # linearly in disparity when k=1. This is probably reasonable, but it'd be a
    # good idea to look at what real NeRF codebases do.
    far_start = near + 1.0 + 1.0 / close_samples_per_ray
    k = 10.0
    far_deltas = (
        1.0
        / (
            1.0
            - onp.linspace(  # onp here is important for float64.
                0.0,
                1.0 - 1 / ((far - far_start) / k + 1),
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
    sample_jitter = jax.random.uniform(sample_prng_key, shape=(ray_count, num_samples))
    ts = ts + step_sizes * sample_jitter

    assert ts.shape == (ray_count, num_samples)
    assert step_sizes.shape == (ray_count, num_samples)

    return ts, step_sizes
    #  # Compute points in world space.
    #  points = (
    #      rays_wrt_world.origins[:, None, :]
    #      + ts[:, :, None] * rays_wrt_world.directions[:, None, :]
    #  )
    #
    #  assert points.shape == (ray_count, num_samples, 3)
    #  return points
