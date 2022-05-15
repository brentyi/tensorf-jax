from typing import Any

from jax import numpy as jnp


def psnr_from_mse(mse: jnp.ndarray) -> jnp.ndarray:
    # Threshold to avoid NaNs.
    mse = jnp.maximum(
        mse,
        eps_from_dtype(
            mse.dtype,
            eps_f16=1e-7,
            eps_f32=1e-10,
        ),
    )

    psnr = -10.0 * jnp.log10(mse)
    return psnr.astype(jnp.float32)


def eps_from_dtype(dtype: Any, eps_f16: float = 1e-4, eps_f32: float = 1e-8) -> float:
    """Get precision constants from data-type."""
    if jnp.issubdtype(dtype, jnp.float16):
        return eps_f16
    elif jnp.issubdtype(dtype, jnp.float32):
        return eps_f32
    else:
        assert False
