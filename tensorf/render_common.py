from __future__ import annotations

import dataclasses
import enum
from typing import Literal, NewType, Tuple, cast

import flax
import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import Annotated, assert_never

from . import cameras, networks, tensor_vm, utils


@jdc.pytree_dataclass
class LearnableParams:
    """Structure containing learnable parameters required for rendering."""

    appearance_mlp_params: flax.core.FrozenDict
    appearance_tensor: tensor_vm.TensorVM
    density_tensors: Tuple[tensor_vm.TensorVM, ...]
    bounded_scene: jdc.Static[bool]


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

    density_samples_per_ray: Tuple[int, ...]
    """Number of points to sample densities at."""

    appearance_samples_per_ray: int
    """Number of points to sample appearances at."""


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

    ts: Annotated[jnp.ndarray, (), jnp.floating]
    step_sizes: Annotated[jnp.ndarray, (), jnp.floating]

    def get_num_rays(self) -> int:
        # The batch axes should be (num_rays, num_samples).
        assert len(self.get_batch_axes()) == 2
        return self.get_batch_axes()[0]

    # def stratified_resample():
    #     pass

    def resample(
        self,
        num_samples: int,
        prng: jax.random.KeyArray,
        anneal_factor: float = 1.0,
    ) -> Tuple[SegmentProbabilities, jnp.ndarray]:
        """Hierarchical resampling, with anneal factor as per mipNeRF-360.

        Returns resampled probabilities and a selected index array.

        anneal_factor=0.0 => uniform distribution.
        anneal_factor=1.0 => samples weighted by termination probability.
        """
        probs = self.p_terminates**anneal_factor
        ray_count, orig_num_samples = probs.shape

        sampled_indices = jax.vmap(
            lambda p, prng: jax.random.choice(
                key=prng,
                a=orig_num_samples,
                shape=(num_samples,),
                replace=False,
                p=p,
            )
        )(
            probs,
            jax.random.split(prng, num=ray_count),
        )
        return self._sample_subset(sampled_indices), sampled_indices

    def _sample_subset(self, sample_indices: jnp.ndarray) -> SegmentProbabilities:
        """Pull out a subset of the sample probabilities from an index array of shape
        (num_rays, new_number_of_samples)."""
        num_rays, new_sample_count = sample_indices.shape
        assert num_rays == self.get_num_rays()

        # Extract subsets of probabilities.
        #
        # For more accurate background rendering, we match the exit probablity of the
        # last sample to the original.
        sub_p_exits = jnp.take_along_axis(
            self.p_exits, sample_indices.at[:, -1].set(-1), axis=-1
        )
        sub_p_terminates = jnp.take_along_axis(
            self.p_terminates, sample_indices, axis=-1
        )
        sub_ts = jnp.take_along_axis(self.ts, sample_indices, axis=-1)
        sub_step_sizes = jnp.take_along_axis(self.step_sizes, sample_indices, axis=-1)

        # Coefficients for unbiasing the expected RGB values using the sampling
        # probabilities. This is helpful because RGB values for points that are not chosen
        # by our appearance sampler are zeroed out.
        #
        # As an example: if the sum of weights* for all samples is 0.95** but the sum of
        # weights for our subset is only 0.7, we can correct the subset weights by
        # (0.95/0.7).
        #
        # *weight at a segment = termination probability at that segment
        # **equivalently: p=0.05 of the ray exiting the last segment and rendering the
        # background.
        eps = utils.eps_from_dtype(self.p_exits.dtype)
        unbias_coeff = (
            # The 0.95 term in the example.
            1.0
            - self.p_exits[:, -1]
            + eps
        ) / (
            # The 0.7 term in the example.
            jnp.sum(sub_p_terminates, axis=1)
            + eps
        )
        assert unbias_coeff.shape == (num_rays,)

        # TODO: a stop_gradient on the unbiasing term is currently needed to avoid NaNs
        # in mixed-precision training. Should be fixable via some scaling term to avoid
        # underflow.
        if sub_p_terminates.dtype == jnp.float16:
            unbias_coeff = jax.lax.stop_gradient(unbias_coeff)
        sub_p_terminates = sub_p_terminates * unbias_coeff[:, None]

        out = SegmentProbabilities(
            p_exits=sub_p_exits,
            p_terminates=sub_p_terminates,
            ts=sub_ts,
            step_sizes=sub_step_sizes,
        )
        assert out.get_batch_axes() == (num_rays, new_sample_count)
        return out

    def render_distance(
        self, mode: Literal[RenderMode.DIST_MEAN, RenderMode.DIST_MEDIAN]
    ) -> jnp.ndarray:
        """Render distances. Useful for depth maps, etc."""
        if mode is RenderMode.DIST_MEAN:
            # Compute distance via expected value.
            sample_distances = jnp.concatenate([self.ts, self.ts[:, -1:]], axis=-1)
            p_terminates_padded = jnp.concatenate(
                [self.p_terminates, self.p_exits[:, -1:]], axis=-1
            )
            assert sample_distances.shape == p_terminates_padded.shape
            return jnp.sum(p_terminates_padded * sample_distances, axis=-1)
        elif mode is RenderMode.DIST_MEDIAN:
            dtype = self.ts.dtype
            (*batch_axes, _num_samples) = self.get_batch_axes()

            # Compute distance via median.
            sample_distances = jnp.concatenate(
                [self.ts, jnp.full((*batch_axes, 1), jnp.inf, dtype=dtype)], axis=-1
            )
            p_not_alive_padded = jnp.concatenate(
                [1.0 - self.p_exits, jnp.ones((*batch_axes, 1), dtype=dtype)], axis=-1
            )
            assert sample_distances.shape == p_not_alive_padded.shape

            median_mask = p_not_alive_padded > 0.5
            median_mask = (
                jnp.zeros_like(median_mask)
                .at[..., 1:]
                .set(jnp.logical_xor(median_mask[..., :-1], median_mask[..., 1:]))
            )

            # Output is medians.
            dists = jnp.sum(median_mask * sample_distances, axis=-1)
            return dists

    @staticmethod
    def compute(
        prerectified_sigmas: jnp.ndarray, step_sizes: jnp.ndarray, ts: jnp.ndarray
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

        sigmas = jax.nn.softplus(prerectified_sigmas + 10.0)

        # Support arbitrary leading batch axes.
        (*batch_axes, sample_count) = sigmas.shape
        assert step_sizes.shape == (*batch_axes, sample_count)

        # Equation 1.
        neg_scaled_sigmas = -sigmas * step_sizes
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
            ts=ts,
            step_sizes=step_sizes,
        )


MlpOutArray = NewType("MlpOutArray", jnp.ndarray)


def get_appearance_mlp_out(
    appearance_mlp: networks.FeatureMlp,
    learnable_params: LearnableParams,
    points: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
) -> MlpOutArray:
    assert points.shape[0] == 3
    _, ray_count, samples_per_ray = points.shape

    appearance_tensor = learnable_params.appearance_tensor
    appearance_feat = appearance_tensor.interpolate(points)
    assert appearance_feat.shape == (
        appearance_tensor.channel_dim(),
        ray_count,
        samples_per_ray,
    )

    total_sample_count = ray_count * samples_per_ray
    appearance_feat = jnp.moveaxis(appearance_feat, 0, -1).reshape(
        (total_sample_count, appearance_tensor.channel_dim())
    )

    assert rays_wrt_world.directions.shape == (ray_count, 3)
    viewdirs = jnp.repeat(rays_wrt_world.directions, repeats=samples_per_ray, axis=0)
    assert viewdirs.shape == (ray_count * samples_per_ray, 3)

    camera_indices = rays_wrt_world.camera_indices
    assert camera_indices.shape == (ray_count,)
    camera_indices = jnp.repeat(camera_indices, repeats=samples_per_ray, axis=0)
    assert camera_indices.shape == (ray_count * samples_per_ray,)

    dtype = points.dtype
    mlp_out = cast(
        jnp.ndarray,
        appearance_mlp.apply(
            learnable_params.appearance_mlp_params,
            features=appearance_feat.reshape((ray_count * samples_per_ray, -1)),
            viewdirs=viewdirs,
            camera_indices=camera_indices,
            dtype=dtype,
        ),
    ).reshape((ray_count, samples_per_ray, -1))

    return MlpOutArray(mlp_out)


def render_from_mlp_out(
    mlp_out: MlpOutArray,
    probs: SegmentProbabilities,
    mode: RenderMode,
) -> jnp.ndarray:

    (ray_count, samples_per_ray, out_dim) = mlp_out.shape
    dtype = mlp_out.dtype

    if mlp_out.shape[-1] == 3:
        rgb_per_point = jax.nn.sigmoid(mlp_out)
    elif mlp_out.shape[-1] == 4:
        rgb_per_point = jax.nn.sigmoid(mlp_out[..., :3])
        probs = SegmentProbabilities.compute(
            prerectified_sigmas=mlp_out[..., 3],
            step_sizes=probs.step_sizes,
            ts=probs.ts,
        )
    else:
        assert False, f"Unsupported shape {mlp_out.shape}"

    expected_rgb = jnp.sum(rgb_per_point * probs.p_terminates[:, :, None], axis=-2)
    assert expected_rgb.shape == (ray_count, 3)

    # Add white background.
    background_color = jnp.ones(3, dtype=dtype)
    expected_rgb_with_background = (
        expected_rgb + probs.p_exits[:, -1:] * background_color
    )
    assert expected_rgb_with_background.shape == (ray_count, 3)
    return expected_rgb_with_background
