import dataclasses
import functools
import pathlib

import dcargs
import fifteen
import jax
from jax import numpy as jnp

from tensorf.tensor_vm import TensorVM


@dataclasses.dataclass
class Args:
    use_magic_vmap: bool
    hlo_target_path: pathlib.Path


args = dcargs.parse(Args)

tensor = TensorVM.initialize(
    grid_dim=128,
    channel_dim=48,
    init=jax.nn.initializers.normal(stddev=0.1),
    prng_key=jax.random.PRNGKey(0),
    dtype=jnp.float32,
)
ijk = jax.random.uniform(
    key=jax.random.PRNGKey(94709),
    shape=(3, 4096, 443),
    minval=-1.0,
    maxval=1.0,
)

with fifteen.utils.stopwatch("JIT compiling"):
    jax.block_until_ready(tensor.interpolate(ijk))
with fifteen.utils.stopwatch("Forward"):
    for i in range(100):
        jax.block_until_ready(tensor.interpolate(ijk))


@jax.jit
def interpolate_backward(tensor: TensorVM, ijk: jnp.ndarray) -> jnp.ndarray:
    def inner(tensor: TensorVM) -> jnp.ndarray:
        return jnp.log(
            jnp.sum(
                jnp.exp(tensor.interpolate(ijk, use_magic_vmap=args.use_magic_vmap))
            )
        )

    return jax.grad(inner)(tensor)


with fifteen.utils.stopwatch("JIT compiling"):
    jax.block_until_ready(interpolate_backward(tensor, ijk))


# 10.3 seconds
with fifteen.utils.stopwatch("Backward"):
    for i in range(100):
        jax.block_until_ready(interpolate_backward(tensor, ijk))


with fifteen.utils.stopwatch("Writing HLO"):
    args.hlo_target_path.write_text(
        interpolate_backward.lower(tensor, ijk).compile().compiler_ir()[0].to_string()
    )
