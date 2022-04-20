"""Neural networks for volumetric rendering."""
from __future__ import annotations

from typing import Any

from flax import linen as nn
from jax import numpy as jnp

relu_layer_init = nn.initializers.kaiming_normal()  # variance = 2.0 / fan_in
linear_layer_init = nn.initializers.lecun_normal()  # variance = 1.0 / fan_in

Dtype = Any


def _fourier_encode(coords: jnp.ndarray, n_freqs: int) -> jnp.ndarray:
    """Fourier feature helper.

    Args:
        coords (jnp.ndarray): Coordinates of shape (*, D).
        n_freqs (int): Number of fourier frequencies.

    Returns:
        jnp.ndarray: Shape (*, n_freqs * 2).
    """
    *batch_axes, D = coords.shape
    coeffs = 2 ** jnp.arange(n_freqs, dtype=jnp.float32)
    inputs = coords[..., None] * coeffs
    assert inputs.shape == (*batch_axes, D, n_freqs)

    out = jnp.sin(
        jnp.concatenate(
            [inputs, inputs + 0.5 * jnp.pi],
            axis=-1,
        )
    )
    assert out.shape == (*batch_axes, D, 2 * n_freqs)
    return out.reshape((*batch_axes, D * 2 * n_freqs))


class FeatureMlp(nn.Module):
    feature_squash_dim: int = 27
    units: int = 128
    feature_n_freqs: int = 6
    viewdir_n_freqs: int = 6

    @nn.compact
    def __call__(  # type: ignore
        self,
        features: jnp.ndarray,
        viewdirs: jnp.ndarray,
        # Computation dtype. Main parameters will always be float32.
        dtype: Any = jnp.float32,
    ) -> jnp.ndarray:
        *batch_axes, feat_dim = features.shape
        assert viewdirs.shape == (*batch_axes, 3)

        # Layer 0. This is `basis_mat` in the original implementation, and reduces the
        # computational requirements of the fourier encoding.
        features = nn.Dense(
            features=self.feature_squash_dim,
            kernel_init=linear_layer_init,
            use_bias=False,
            dtype=dtype,
        )(features)

        # Compute fourier features.
        #
        # This computes both sines and cosines to match other implementations, but since
        # cos(x) == sin(x + pi/2) we could also consider just adding a bias term to the
        # dense layer above and only picking one.
        x = jnp.concatenate(
            [
                features,
                viewdirs,
                _fourier_encode(features, self.feature_n_freqs),
                _fourier_encode(viewdirs, self.viewdir_n_freqs),
            ],
            axis=-1,
        )
        expected_encoded_dim = (
            self.feature_squash_dim
            + 3
            + 2 * self.feature_n_freqs * self.feature_squash_dim
            + 2 * self.viewdir_n_freqs * 3
        )
        assert x.shape == (*batch_axes, expected_encoded_dim)

        # Layer 1.
        x = nn.Dense(
            features=self.units,
            kernel_init=relu_layer_init,
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        assert x.shape == (*batch_axes, self.units)

        # Layer 2.
        x = nn.Dense(
            features=self.units,
            kernel_init=relu_layer_init,
            dtype=dtype,
        )(x)
        x = nn.relu(x)
        assert x.shape == (*batch_axes, self.units)

        # Layer 3.
        x = nn.Dense(
            features=3,
            kernel_init=linear_layer_init,
            dtype=dtype,
        )(x)
        assert x.shape == (*batch_axes, 3)

        rgb = nn.sigmoid(x)
        return rgb
