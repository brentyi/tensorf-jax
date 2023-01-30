from __future__ import annotations

import dataclasses
import enum
from typing import Callable, Literal, NewType, Optional, Tuple, cast

import flax
import jax
import jax_dataclasses as jdc
import numpy as onp
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

    bins: Bins

    def get_num_rays(self) -> int:
        # The batch axes should be (num_rays, num_samples).
        assert len(self.get_batch_axes()) == 2
        return self.get_batch_axes()[0]

    def resample_subset(
        self,
        num_samples: int,
        prng: jax.random.KeyArray,
        anneal_factor: float = 1.0,
    ) -> Tuple[SegmentProbabilities, jnp.ndarray]:
        """Hierarchical resampling.

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
        sub_bins = Bins(
            ts=jnp.take_along_axis(self.bins.ts, sample_indices, axis=-1),
            step_sizes=jnp.take_along_axis(
                self.bins.step_sizes, sample_indices, axis=-1
            ),
            starts=jnp.take_along_axis(self.bins.starts, sample_indices, axis=-1),
            ends=jnp.take_along_axis(self.bins.ends, sample_indices, axis=-1),
            contiguous=False,
        )
        assert sub_bins.ts.shape == sub_p_exits.shape

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
            bins=sub_bins,
        )
        assert out.get_batch_axes() == (num_rays, new_sample_count)
        return out

    def render_distance(
        self, mode: Literal[RenderMode.DIST_MEAN, RenderMode.DIST_MEDIAN]
    ) -> jnp.ndarray:
        """Render distances. Useful for depth maps, etc."""
        if mode is RenderMode.DIST_MEAN:
            # Compute distance via expected value.
            sample_distances = jnp.concatenate(
                [self.bins.ts, self.bins.ts[:, -1:]], axis=-1
            )
            p_terminates_padded = jnp.concatenate(
                [self.p_terminates, self.p_exits[:, -1:]], axis=-1
            )
            assert sample_distances.shape == p_terminates_padded.shape
            return jnp.sum(p_terminates_padded * sample_distances, axis=-1)
        elif mode is RenderMode.DIST_MEDIAN:
            dtype = self.bins.ts.dtype
            (*batch_axes, _num_samples) = self.get_batch_axes()

            # Compute distance via median.
            sample_distances = jnp.concatenate(
                [self.bins.ts, jnp.full((*batch_axes, 1), jnp.inf, dtype=dtype)],
                axis=-1,
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
        prerectified_sigmas: jnp.ndarray,
        bins: Bins,
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

        # Initial training tends to be faster with a truncated exponential, but
        # converged PSNRs are higher with a softplus.
        #
        # The trunc_exp may still be nice for half-precision training.
        sigmas = jax.nn.softplus(prerectified_sigmas + 10.0)
        # sigmas = trunc_exp(prerectified_sigmas)

        # Support arbitrary leading batch axes.
        (*batch_axes, sample_count) = sigmas.shape
        assert bins.step_sizes.shape == (*batch_axes, sample_count)

        # Equation 1.
        neg_scaled_sigmas = -sigmas * bins.step_sizes
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
            bins=bins,
        )


MlpOutArray = NewType("MlpOutArray", jnp.ndarray)
RgbPerPointArray = NewType("RgbPerPointArray", jnp.ndarray)


def get_appearance_mlp_out(
    appearance_mlp: networks.FeatureMlp,
    learnable_params: LearnableParams,
    points: jnp.ndarray,
    rays_wrt_world: cameras.Rays3D,
    interpolate_func: Callable[[tensor_vm.TensorVM, jnp.ndarray], jnp.ndarray],
) -> MlpOutArray:
    assert points.shape[0] == 3
    _, ray_count, samples_per_ray = points.shape

    appearance_tensor = learnable_params.appearance_tensor
    appearance_feat = interpolate_func(appearance_tensor, points)
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


def direct_rgb_from_mlp(mlp_out: MlpOutArray) -> RgbPerPointArray:
    """No-op."""
    assert mlp_out.shape[-1] == 3
    return RgbPerPointArray(mlp_out)


#  def rgb_and_density_from_mlp(
#      mlp_out: MlpOutArray,
#      bins: Bins,
#  ) -> Tuple[RgbPerPointArray, SegmentProbabilities]:
#      assert mlp_out.shape[-1] == 4
#
#      # Extract RGB values.
#      rgb_per_point = RgbPerPointArray(mlp_out[..., :3])
#
#      probs = SegmentProbabilities.compute(
#          prerectified_sigmas=mlp_out[..., -1] + 10.0,
#          step_sizes=bins.step_sizes,
#          ts=bins.ts,
#      )
#      return rgb_per_point, probs


def render_from_mlp_out(
    rgb_per_point: RgbPerPointArray,
    probs: SegmentProbabilities,
    mode: RenderMode,
    background_mode: Literal["white", "last_sample"] = "white",
) -> jnp.ndarray:
    """Returns rendered colors, and updated probabilities if original ones were
    overwritten via a density a channel from the MLP."""

    (ray_count, samples_per_ray, out_dim) = rgb_per_point.shape
    dtype = rgb_per_point.dtype

    overridden_probs = False
    assert rgb_per_point.shape[-1] == 3
    rgb_per_point = jax.nn.sigmoid(rgb_per_point)
    # if mlp_out.shape[-1] == 3:
    #     rgb_per_point = jax.nn.sigmoid(mlp_out)
    # elif mlp_out.shape[-1] == 4:
    #     rgb_per_point = jax.nn.sigmoid(mlp_out[..., :3])
    #     probs = SegmentProbabilities.compute(
    #         prerectified_sigmas=mlp_out[..., 3],
    #         step_sizes=probs.step_sizes,
    #         ts=probs.ts,
    #         near=probs.near,
    #         far=probs.far,
    #     )
    #     overridden_probs = True
    # else:
    #     assert False, f"Unsupported shape {mlp_out.shape}"

    expected_rgb = jnp.sum(rgb_per_point * probs.p_terminates[:, :, None], axis=-2)
    assert expected_rgb.shape == (ray_count, 3)

    # Add white background.
    if background_mode == "white":
        background_color = jnp.ones(3, dtype=dtype)
    elif background_mode == "last_sample":
        background_color = rgb_per_point[..., -1, :]
    else:
        assert_never(background_mode)

    expected_rgb_with_background = (
        expected_rgb + probs.p_exits[:, -1:] * background_color
    )
    assert expected_rgb_with_background.shape == (ray_count, 3)

    return expected_rgb_with_background


@jdc.pytree_dataclass
class Bins:
    ts: jnp.ndarray
    step_sizes: jnp.ndarray

    starts: jnp.ndarray
    ends: jnp.ndarray

    contiguous: jdc.Static[bool]
    """True if there are no gaps between bins."""

    @staticmethod
    def from_boundaries(boundaries: jnp.ndarray) -> Bins:
        starts = boundaries[..., :-1]
        ends = boundaries[..., 1:]
        step_sizes = ends - starts
        ts = starts + step_sizes / 2.0

        return Bins(
            ts=ts,
            step_sizes=step_sizes,
            starts=starts,
            ends=ends,
            contiguous=True,
        )

    def weighted_sample_stratified(
        self,
        weights: jnp.ndarray,
        prng: jax.random.KeyArray,
        num_samples: int,
        anneal_factor: float,
    ) -> jnp.ndarray:
        *batch_dims, old_num_samples = weights.shape
        batch_dims = tuple(batch_dims)

        assert self.contiguous, "Currently only contiguous bins are supported."
        boundaries = jnp.concatenate(
            [self.starts, self.ends[..., -1:]],
            axis=-1,
        )

        # Accumulate weights, and scale from 0 to 1.
        accumulated_weights = jnp.cumsum(weights, axis=-1)
        accumulated_weights = jnp.concatenate(
            [jnp.zeros(batch_dims + (1,)), accumulated_weights],
            axis=-1,
        )
        accumulated_weights = accumulated_weights / (
            accumulated_weights[..., -1:] + 1e-4
        )
        assert accumulated_weights.shape == batch_dims + (old_num_samples + 1,)

        batch_dim_flattened = int(onp.prod(batch_dims))

        x = _sample_quasi_uniform_ordered(
            prng,
            min_bound=0.0,
            max_bound=1.0,
            bins=num_samples,
            batch_dims=(batch_dim_flattened,),
        )
        samples = jax.vmap(jnp.interp)(
            x=x,
            xp=accumulated_weights.reshape((batch_dim_flattened, -1)),
            fp=boundaries.reshape((batch_dim_flattened, -1)),
        ).reshape(batch_dims + (num_samples,))

        return samples


def _sample_quasi_uniform_ordered(
    prng: jax.random.KeyArray,
    min_bound: float,
    max_bound: float,
    bins: int,
    batch_dims: Tuple[int, ...],
) -> jnp.ndarray:
    """Quasi-uniform sampling. Separates the sampling range into a specified number of
    bins, and selects one sample from each bin.
    Output is in ascending order."""
    sampling_bin_size = (max_bound - min_bound) / bins
    sampling_bin_starts = jnp.arange(0, bins) * sampling_bin_size

    # Add some batch axes; these two lines are totally unnecessary.
    for i in range(len(batch_dims)):
        sampling_bin_starts = sampling_bin_starts[None, ...]

    samples = sampling_bin_starts + jax.random.uniform(
        key=prng,
        shape=batch_dims + (bins,),
        minval=0.0,
        maxval=sampling_bin_size,
    )
    assert samples.shape == batch_dims + (bins,)
    return samples


@jax.custom_jvp
def trunc_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Exponential with a clipped gradients."""
    return jnp.exp(x)


@trunc_exp.defjvp
def trunc_exp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = trunc_exp(x)
    tangent_out = x_dot * jnp.exp(jnp.clip(x, -15, 15))
    return primal_out, tangent_out
