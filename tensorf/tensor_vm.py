from __future__ import annotations

import functools
from typing import Any, Callable, Tuple, Union

import jax
import jax.scipy
import jax_dataclasses as jdc
from jax import numpy as jnp

Shape = Tuple[int, ...]
Dtype = Any

Scalar = Union[float, jnp.ndarray]


@jdc.pytree_dataclass
class TensorVM:
    """A tensor decomposition consisted of three vector-matrix pairs."""

    stacked_single_vm: TensorVMSingle
    """Three vector-matrix pairs, stacked along axis 0."""

    @staticmethod
    def initialize(
        grid_dim: int,
        per_axis_channel_dim: int,
        init: Callable[[jax.random.KeyArray, Shape, Dtype], jnp.ndarray],
        prng_key: jax.random.KeyArray,
        dtype: Dtype,
    ) -> TensorVM:
        prng_keys = jax.random.split(prng_key, 3)
        return TensorVM(
            stacked_single_vm=jax.vmap(
                lambda prng_key: TensorVMSingle.initialize(
                    grid_dim,
                    per_axis_channel_dim,
                    init,
                    prng_key=prng_key,
                    dtype=dtype,
                )
            )(prng_keys)
        )

    def interpolate(self, ijk: jnp.ndarray) -> jnp.ndarray:
        """Look up a coordinate in our VM decomposition.

        Input should have shape (3, *) and be in the range [-1.0, 1.0].
        Output should have shape (channel_dim * 3, *)."""

        batch_axes = ijk.shape[1:]
        assert ijk.shape == (3, *batch_axes)
        kij = ijk[jnp.array([2, 0, 1]), ...]
        jki = ijk[jnp.array([1, 2, 0]), ...]
        indices = jnp.stack([ijk, kij, jki], axis=0)
        assert indices.shape == (3, 3, *batch_axes)

        interpolate_func = TensorVMSingle.interpolate
        if len(batch_axes) >= 2:
            # TODO: this magic vmap is unnecessary and doesn't impact numerical results,
            # but enables a massive performance increase. This is 3~4x better training
            # throughput for single-precision, ~1.5x in mixed-precision.
            #
            # I'm not exactly sure why, but it appears to:
            # - Shuffle the memory layout and improve access patterns.
            # - Reduce the length of the HLO generated during tracing by ~150 lines.
            #
            # Some plots/discussion: https://github.com/google/jax/discussions/10332
            #
            # Setting the axis to -1 also produces a speedup, albeit a slightly smaller
            # one. Numerical results are identical in either case.
            interpolate_func = jax.vmap(
                interpolate_func,
                in_axes=(None, -2),
                out_axes=-2,
            )

        # Vectorize over axis=0, which will be of size 3. (one for each vector-matrix
        # pair)
        #
        # Empirically, applying this after the magic vmap above is slightly
        # faster than applying it before.
        interpolate_func = jax.vmap(interpolate_func)

        feature = interpolate_func(self.stacked_single_vm, indices)
        assert feature.shape == (3, self.stacked_single_vm.channel_dim(), *batch_axes)

        # Note the original implementation also has a basis matrix that left-multiplies
        # here; we fold this into the appearance network.
        feature = feature.reshape(
            (3 * self.stacked_single_vm.channel_dim(), *batch_axes)
        )
        return feature

    @functools.partial(jax.jit, static_argnums=1)
    def resize(self, grid_dim: int) -> TensorVM:
        """Resize our tensor decomposition."""

        d: TensorVMSingle
        return TensorVM(
            stacked_single_vm=jax.vmap(
                lambda inner: TensorVMSingle.resize(inner, grid_dim=grid_dim)
            )(self.stacked_single_vm)
        )

    def grid_dim(self) -> int:
        return self.stacked_single_vm.grid_dim()

    def channel_dim(self) -> int:
        return self.stacked_single_vm.channel_dim() * 3


@jdc.pytree_dataclass
class TensorVMSingle:
    """Helper for 4D tensors decomposed into a vector-matrix pair."""

    vector: jnp.ndarray
    matrix: jnp.ndarray

    @staticmethod
    def initialize(
        grid_dim: int,
        channel_dim: int,
        init: Callable[[jax.random.KeyArray, Shape, Dtype], jnp.ndarray],
        prng_key: jax.random.KeyArray,
        dtype: Dtype,
    ) -> TensorVMSingle:
        """ "Initialize a VM-decomposed 4D tensor (depth, width, height, channel).

        For now, we assume that the depth/width/height dimensions are equal."""
        key0, key1 = jax.random.split(prng_key)

        # Note that putting channel dimension first is *much* faster.
        # vector_shape = (grid_dim, channel_dim)
        # matrix_shape = (grid_dim, grid_dim, channel_dim)
        vector_shape = (channel_dim, grid_dim)
        matrix_shape = (channel_dim, grid_dim, grid_dim)

        return TensorVMSingle(
            vector=init(key0, vector_shape, dtype),
            matrix=init(key1, matrix_shape, dtype),
        )

    def interpolate(self, ijk: jnp.ndarray) -> jnp.ndarray:
        """Grid lookup with interpolation.

        ijk should be of shape (3, *) all be within [-1, 1].
        Output will have shape (channel_dim, *)."""
        batch_axes = ijk.shape[1:]
        assert ijk.shape == (3, *batch_axes)
        assert jnp.issubdtype(ijk.dtype, jnp.floating)

        # [-1.0, 1.0] => [0.0, 1.0]
        ijk = (ijk + 1.0) / 2.0

        # [0.0, 1.0] => [0.0, grid_dim - 1.0]
        ijk = ijk * (self.grid_dim() - 1.0)

        vector_coeffs = linear_interpolation_with_channel_axis(
            self.vector, coordinates=ijk[0:1, ...]
        )
        matrix_coeffs = linear_interpolation_with_channel_axis(
            self.matrix, coordinates=ijk[1:3, ...]
        )

        assert (
            vector_coeffs.shape
            == matrix_coeffs.shape
            == (self.channel_dim(), *batch_axes)
        )
        return vector_coeffs * matrix_coeffs

    def grid_dim(self) -> int:
        """Returns the grid dimension."""
        r0, r1 = self.matrix.shape[-2:]
        r2 = self.vector.shape[-1]
        assert r0 == r1 == r2
        return r0

    def channel_dim(self) -> int:
        """Returns the channel dimension."""
        c0 = self.matrix.shape[-3]
        c1 = self.vector.shape[-2]
        assert c0 == c1
        return c0

    @functools.partial(jax.jit, static_argnums=1)
    def resize(self, grid_dim: int) -> TensorVMSingle:
        """Resize our decomposition, while interpolating linearly."""

        channel_dim = self.channel_dim()
        matrix_shape = (channel_dim, grid_dim, grid_dim)
        vector_shape = (channel_dim, grid_dim)

        return TensorVMSingle(
            matrix=jax.image.resize(self.matrix, matrix_shape, "linear"),
            vector=jax.image.resize(self.vector, vector_shape, "linear"),
        )


def linear_interpolation_with_channel_axis(
    grid: jnp.ndarray, coordinates: jnp.ndarray
) -> jnp.ndarray:
    """Thin wrapper around `jax.scipy.ndimage.map_coordinates()` for linear
    interpolation.

    Standard set of shapes might look like:
        grid (C, 128, 128, 128)
        coordinates (3, *)

    Which would return:
        (C, *)
    """
    assert len(grid.shape[1:]) == coordinates.shape[0]
    # vmap to add a channel axis.
    output = jax.vmap(
        lambda g: jax.scipy.ndimage.map_coordinates(
            g,
            coordinates=coordinates,  # type: ignore
            order=1,
            mode="nearest",
        )
    )(grid)
    assert output.shape == grid.shape[:1] + coordinates.shape[1:]
    return output
