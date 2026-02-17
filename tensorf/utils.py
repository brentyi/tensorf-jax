from jax import numpy as jnp


def psnr_from_mse(mse: jnp.ndarray) -> jnp.ndarray:
    # Threshold to avoid NaNs.
    mse = jnp.maximum(mse, 1e-10)

    psnr = -10.0 * jnp.log10(mse)
    return psnr.astype(jnp.float32)
