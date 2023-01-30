from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from einops import rearrange
from jax import numpy as jnp

from . import cameras, interlevel, networks, render_common, tensor_vm, utils


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Render a set of rays.

    Output should have shape `(ray_count, 3)` and `()`, where the latter is an
    interlevel/proposal loss."""

    num_density_tensors = len(learnable_params.density_tensors)
    assert len(config.density_samples_per_ray) == num_density_tensors

    # Cast everything to the desired dtype.
    learnable_params, aabb, rays_wrt_world = jax.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(jnp.floating, dtype) else x,
        (learnable_params, aabb, rays_wrt_world),
    )
    (ray_count,) = rays_wrt_world.get_batch_axes()

    sample_prng_keys = jax.random.split(prng_key, num=num_density_tensors + 1)

    # Sample initial points in world space.
    bins = sample_initial_points_from_rays(
        ray_count,
        num_samples=config.density_samples_per_ray[0],
        near=config.near,
        far=config.far,
        sample_prng_key=sample_prng_keys[0],
    )
    render_points = rays_wrt_world.points_from_ts(bins.ts)
    render_points = jnp.moveaxis(render_points, -1, 0)

    #  def compute_sdist(boundaries):
    #      return (boundaries - config.near) / (config.far - config.near)

    probs = None
    ray_history = []

    assert (
        num_density_tensors == 1
    )  # Currently we assume only 1 density tensor. But could support more.
    for i in range(num_density_tensors):
        density_feat = interpolate_contracted(
            learnable_params.density_tensors[i],
            render_points,
        )
        prerectified_sigmas = jnp.sum(density_feat, axis=0)
        probs = render_common.SegmentProbabilities.compute(prerectified_sigmas, bins)
        #  ray_history.append(
        #      {"sdist": compute_sdist(bins.boundaries), "weights": probs.p_terminates}
        #  )
        del prerectified_sigmas
        del density_feat

        probs, render_indices = probs.resample_subset(
            num_samples=config.appearance_samples_per_ray,
            prng=sample_prng_keys[i + 1],
            anneal_factor=1.0,
        )
        bins = probs.bins

        render_points = jnp.take_along_axis(
            arr=points, indices=render_indices[None, :, :], axis=-1
        )
        assert render_points.shape == (3, ray_count, config.appearance_samples_per_ray)

        #  boundaries = bins.weighted_sample_stratified(
        #      weights=probs.p_terminates,
        #      prng=sample_prng_keys[i + 1],
        #      num_samples=(
        #          config.density_samples_per_ray[i + 1]
        #          if i < num_density_tensors - 1
        #          else config.appearance_samples_per_ray
        #      ),
        #      anneal_factor=proposal_anneal_factor,
        #  )
        #  boundaries = jax.lax.stop_gradient(boundaries)
        #  bins = render_common.Bins.from_boundaries(boundaries)
        #
        #  ts = probs.sample_stratified_ts(
        #      prng=sample_prng_keys[i + 1],
        #      num_samples=(
        #          config.density_samples_per_ray[i + 1]
        #          if i < num_density_tensors - 1
        #          else config.appearance_samples_per_ray
        #      ),
        #      anneal_factor=proposal_anneal_factor,
        #  )
        #
        #  render_points = rays_wrt_world.points_from_ts(bins.ts)
        #  render_points = jnp.moveaxis(render_points, -1, 0)
    assert probs is not None

    mlp_out = render_common.get_appearance_mlp_out(
        appearance_mlp,
        learnable_params,
        render_points,
        rays_wrt_world,
        interpolate_func=interpolate_contracted,
    )
    if mlp_out.shape[-1] == 3:
        # TODO: probs is broken in this case?
        assert False
        rgb_per_point = render_common.direct_rgb_from_mlp(mlp_out)
    else:
        #  rgb_per_point = mlp_out[..., :3]
        #  probs = resampled_probs
        assert mlp_out.shape[-1] == 4
        #  density_feat = interpolate_contracted(
        #      learnable_params.density_tensors[-1],
        #      render_points,
        #  )

        # Note that the MLP can only _decrease_ the observed density.
        #  prerectified_sigmas = jnp.sum(density_feat, axis=0) - jax.nn.softplus(
        #      mlp_out[..., -1]
        #  )
        #  prerectified_sigmas = mlp_out[..., -1]
        #  probs = render_common.SegmentProbabilities.compute(
        #      prerectified_sigmas=prerectified_sigmas, bins=bins
        #  )
        rgb_per_point = mlp_out[..., :3]
        # , probs = render_common.rgb_and_density_from_mlp(mlp_out, bins)

    rendered = render_common.render_from_mlp_out(
        rgb_per_point,
        probs,
        config.mode,
        background_mode="last_sample",
    )
    #  ray_history.append(
    #      {"sdist": compute_sdist(bins.boundaries), "weights": probs.p_terminates}
    #  )

    return rendered, 0.0  # interlevel.interlevel_loss(ray_history)


def interpolate_contracted(
    tensor: tensor_vm.TensorVM,
    points: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a feature vector from a VM decomposed tensor, with scene
    contraction."""

    # TODO: this logic should be consolidated with aabb stuff.
    assert points.shape[0] == 3

    # Contract points to cube.
    norm = jnp.linalg.norm(points, ord=jnp.inf, axis=0, keepdims=True)
    contracted_points = jnp.where(
        norm <= 1.0, points, (2.0 - 1.0 / norm) * points / norm
    )
    assert isinstance(contracted_points, jnp.ndarray)
    assert contracted_points.shape[0] == 3

    out = tensor.interpolate(contracted_points)
    assert out.shape == (tensor.channel_dim(), *points.shape[1:])
    return out


def sample_initial_points_from_rays(
    ray_count: int,
    num_samples: int,
    near: float,
    far: float,
    sample_prng_key: jax.random.KeyArray,
) -> render_common.Bins:
    # Contracted scene: sample linearly for close samples, then start spacing
    # samples out.
    #
    # Note that we add 1 because we are sampling bin boundaries here; each bin
    # corresponds to 1 point.
    close_samples_per_ray = (num_samples + 1) // 2
    far_samples_per_ray = (num_samples + 1) - close_samples_per_ray

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

    boundaries = jnp.tile(
        jnp.concatenate([close_ts, far_ts])[None, :], reps=(ray_count, 1)
    )

    bins = render_common.Bins.from_boundaries(boundaries)
    assert bins.starts.shape == (ray_count, num_samples)
    assert bins.ends.shape == (ray_count, num_samples)
    assert bins.ts.shape == (ray_count, num_samples)
    assert bins.step_sizes.shape == (ray_count, num_samples)
    return bins

    #  # Compute points in world space.
    #  points = (
    #      rays_wrt_world.origins[:, None, :]
    #      + ts[:, :, None] * rays_wrt_world.directions[:, None, :]
    #  )
    #
    #  assert points.shape == (ray_count, num_samples, 3)
    #  return points
