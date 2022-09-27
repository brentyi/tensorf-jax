from __future__ import annotations

import functools
import math
from typing import Tuple

import fifteen
import flax
import jax
import jax_dataclasses as jdc
import optax
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated

from . import data, networks, render, tensor_vm, train_config, utils


@jdc.pytree_dataclass
class TrainState(jdc.EnforcedAnnotationsMixin):
    config: jdc.Static[train_config.TensorfConfig]

    # Representation/parameters.
    appearance_mlp: jdc.Static[networks.FeatureMlp]
    learnable_params: render.LearnableParams

    # Optimizer.
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState

    # Current axis-aligned bounding box.
    aabb: Annotated[jnp.ndarray, (2, 3)]

    # Misc.
    prng_key: jax.random.KeyArray
    step: int

    @staticmethod
    def initialize(
        config: train_config.TensorfConfig,
        grid_dim: int,
        prng_key: jax.random.KeyArray,
    ) -> TrainState:
        prng_keys = jax.random.split(prng_key, 4)
        normal_init = jax.nn.initializers.normal(stddev=0.1)

        def make_mlp() -> Tuple[networks.FeatureMlp, flax.core.FrozenDict]:
            dummy_features = jnp.zeros((1, config.appearance_feat_dim * 3))
            dummy_viewdirs = jnp.zeros((1, 3))
            appearance_mlp = networks.FeatureMlp(
                feature_n_freqs=config.feature_n_freqs,
                viewdir_n_freqs=config.viewdir_n_freqs,
            )
            appearance_mlp_params = appearance_mlp.init(
                prng_keys[0],
                features=dummy_features,
                viewdirs=dummy_viewdirs,
            )
            return appearance_mlp, appearance_mlp_params

        appearance_mlp, appearance_mlp_params = make_mlp()

        learnable_params = render.LearnableParams(
            appearance_mlp_params=appearance_mlp_params,
            appearance_tensor=tensor_vm.TensorVM.initialize(
                grid_dim=grid_dim,
                per_axis_channel_dim=config.appearance_feat_dim,
                init=normal_init,
                prng_key=prng_keys[1],
                dtype=jnp.float32,  # Main copy of parameters are always float32.
            ),
            density_tensor=tensor_vm.TensorVM.initialize(
                grid_dim=grid_dim,
                per_axis_channel_dim=config.density_feat_dim,
                init=normal_init,
                prng_key=prng_keys[2],
                dtype=jnp.float32,
            ),
        )
        optimizer = TrainState._make_optimizer(config.optimizer)
        optimizer_state = optimizer.init(learnable_params)

        return TrainState(
            config=config,
            appearance_mlp=appearance_mlp,
            learnable_params=learnable_params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            aabb=jnp.array([config.initial_aabb_min, config.initial_aabb_max]),
            prng_key=prng_keys[3],
            step=0,
        )

    @functools.partial(jax.jit, donate_argnums=0)
    def training_step(
        self, minibatch: data.RenderedRays
    ) -> Tuple[TrainState, fifteen.experiments.TensorboardLogData]:
        """Single training step."""
        render_prng_key, new_prng_key = jax.random.split(self.prng_key)

        # If in mixed-precision mode, we render and backprop in float16.
        if self.config.mixed_precision:
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.float32

        def compute_loss(
            learnable_params: render.LearnableParams,
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            # Compute sample counts from grid dimensionality.
            # TODO: move heuristics into config?
            grid_dim = self.learnable_params.appearance_tensor.grid_dim()
            assert grid_dim == self.learnable_params.density_tensor.grid_dim()
            density_samples_per_ray = int(math.sqrt(3 * grid_dim**2))
            appearance_samples_per_ray = int(0.15 * density_samples_per_ray)

            # Render and compute loss.
            rendered = render.render_rays(
                appearance_mlp=self.appearance_mlp,
                learnable_params=learnable_params,
                aabb=self.aabb,
                rays_wrt_world=minibatch.rays_wrt_world,
                prng_key=render_prng_key,
                config=render.RenderConfig(
                    # TODO: the near/far thresholds don't matter much since we fit rays
                    # into the bounding box, but probably shouldn't be hardcoded.
                    near=0.1,
                    far=10.0,
                    mode=render.RenderMode.RGB,
                    density_samples_per_ray=density_samples_per_ray,
                    appearance_samples_per_ray=appearance_samples_per_ray,
                ),
                dtype=compute_dtype,
            )
            assert (
                rendered.shape
                == minibatch.colors.shape
                == minibatch.get_batch_axes() + (3,)
            )
            label_colors = minibatch.colors
            assert jnp.issubdtype(rendered.dtype, compute_dtype)
            assert jnp.issubdtype(label_colors.dtype, jnp.float32)

            mse = jnp.mean((rendered - label_colors) ** 2)
            loss = mse  # TODO: add regularization terms.

            log_data = fifteen.experiments.TensorboardLogData(
                scalars={
                    "mse": mse,
                    "psnr": utils.psnr_from_mse(mse),
                }
            )
            return loss * self.config.loss_scale, log_data

        # Compute gradients.
        log_data: fifteen.experiments.TensorboardLogData
        grads: render.LearnableParams
        learnable_params = jax.tree_map(
            # Cast parameters to desired precision.
            lambda x: x.astype(compute_dtype),
            self.learnable_params,
        )
        (loss, log_data), grads = jax.value_and_grad(
            compute_loss,
            has_aux=True,
        )(learnable_params)

        # To prevent NaNs from momentum computations in mixed-precision mode, it's
        # important that gradients are float32 before being passed to the optimizer.
        grads_unscaled = jax.tree_map(
            lambda x: x.astype(jnp.float32) / self.config.loss_scale,
            grads,
        )
        assert jnp.issubdtype(
            jax.tree_leaves(grads_unscaled)[0].dtype, jnp.float32
        ), "Gradients should always be float32."

        # Compute learning rate decay.
        # We could put this in the optax chain as well, but explicitly computing here
        # makes logging & reset handling easier.
        if self.config.optimizer.lr_upsample_reset:
            # For resetting after upsampling, we find the smallest non-negative value of
            # (current step - upsampling iteration #).
            step_deltas = self.step - jnp.array((0,) + self.config.upsamp_iters)
            step_deltas = jnp.where(
                step_deltas >= 0, step_deltas, jnp.iinfo(step_deltas.dtype).max
            )
            resetted_step = jnp.min(step_deltas)
        else:
            resetted_step = self.step

        decay_iters = self.config.optimizer.lr_decay_iters
        if decay_iters is None:
            decay_iters = self.config.n_iters

        lr_decay_coeff = optax.exponential_decay(
            init_value=1.0,
            transition_steps=decay_iters,
            decay_rate=self.config.optimizer.lr_decay_target_ratio,
            end_value=self.config.optimizer.lr_decay_target_ratio,
        )(resetted_step)

        # Propagate gradients through ADAM, learning rate scheduler, etc.
        updates, new_optimizer_state = self.optimizer.update(
            grads_unscaled, self.optimizer_state, self.learnable_params
        )
        updates = jax.tree_map(lambda x: lr_decay_coeff * x, updates)

        # Add learning rates to Tensorboard logs.
        log_data = log_data.merge_scalars(
            {
                "lr_tensor": lr_decay_coeff * self.config.optimizer.lr_init_tensor,
                "lr_mlp": lr_decay_coeff * self.config.optimizer.lr_init_mlp,
                "grad_norm": optax.global_norm(grads),
            }
        )

        with jdc.copy_and_mutate(self, validate=True) as new_state:
            new_state.optimizer_state = new_optimizer_state
            new_state.learnable_params = optax.apply_updates(
                self.learnable_params, updates
            )
            new_state.prng_key = new_prng_key
            new_state.step = new_state.step + 1
        return new_state, log_data.prefix("train/")

    @staticmethod
    def _make_optimizer(
        config: train_config.OptimizerConfig,
    ) -> optax.GradientTransformation:
        """Set up Adam optimizer."""
        return optax.chain(
            # First, we rescale gradients with ADAM. Note that eps=1e-8 is OK because
            # gradients are always converted to float32 before being passed to the
            # optimizer.
            optax.scale_by_adam(
                b1=0.9,
                b2=0.99,
                eps=1e-8,
                eps_root=0.0,
            ),
            # Apply MLP parameter learning rate. Note the negative sign needed for
            # gradient descent.
            optax.masked(
                optax.scale(-config.lr_init_mlp),
                mask=render.LearnableParams(
                    appearance_mlp_params=True,  # type: ignore
                    appearance_tensor=False,  # type: ignore
                    density_tensor=False,  # type: ignore
                ),
            ),
            # Apply tensor decomposition learning rate. Note the negative sign needed
            # for gradient descent.
            optax.masked(
                optax.scale(-config.lr_init_tensor),
                mask=render.LearnableParams(
                    appearance_mlp_params=False,  # type: ignore
                    appearance_tensor=True,  # type: ignore
                    density_tensor=True,  # type: ignore
                ),
            ),
        )

    def resize_grid(self, new_grid_dim: int) -> TrainState:
        """Resize the grid underlying a training state by linearly interpolating grid
        parameters."""
        with jdc.copy_and_mutate(self, validate=False) as resized:
            # Resample the feature grids, with linear interpolation.
            resized.learnable_params.density_tensor = (
                resized.learnable_params.density_tensor.resize(new_grid_dim)
            )
            resized.learnable_params.appearance_tensor = (
                resized.learnable_params.appearance_tensor.resize(new_grid_dim)
            )

            # Perform some nasty surgery to resample the momentum parameters as well.
            adam_state = resized.optimizer_state[0]
            assert isinstance(adam_state, optax.ScaleByAdamState)
            nu: render.LearnableParams = adam_state.nu
            mu: render.LearnableParams = adam_state.mu
            resized.optimizer_state = (
                adam_state._replace(  # NamedTuple `_replace()`.
                    nu=jdc.replace(
                        nu,
                        density_tensor=nu.density_tensor.resize(new_grid_dim),
                        appearance_tensor=nu.appearance_tensor.resize(new_grid_dim),
                    ),
                    mu=jdc.replace(
                        mu,
                        density_tensor=mu.density_tensor.resize(new_grid_dim),
                        appearance_tensor=mu.appearance_tensor.resize(new_grid_dim),
                    ),
                ),
            ) + resized.optimizer_state[1:]
        return resized


def run_training_loop(config: train_config.TensorfConfig) -> None:
    """Full training loop implementation."""

    # Set up our experiment: for checkpoints, logs, metadata, etc.
    experiment = fifteen.experiments.Experiment(data_dir=config.run_dir)
    experiment.clear()
    experiment.write_metadata("config", config)

    # Initialize training state.
    train_state: TrainState
    train_state = TrainState.initialize(
        config,
        grid_dim=config.grid_dim_init,
        prng_key=jax.random.PRNGKey(94709),
    )

    # Load dataset.
    assert config.dataset_type == "blender"
    dataloader = fifteen.data.InMemoryDataLoader(
        dataset=data.rendered_rays_from_views(
            views=data.load_blender_dataset(
                config.dataset_path,
                split="train",
                progress_bar=True,
            )
        ),
        minibatch_size=config.minibatch_size,
    )
    minibatches = fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0)
    minibatches = iter(minibatches)

    # Totally optional: frontload JIT compile time. This just prevents training from
    # pausing in the middle to re-JIT for different grid dimensions.
    frontload_jit_compile_time = False
    if frontload_jit_compile_time:
        for upsamp_index in tqdm(
            range(-1, len(config.upsamp_iters)), desc="JIT training step"
        ):
            grid_dim = int(
                config.grid_dim_init
                + (config.grid_dim_final - config.grid_dim_init)
                * ((upsamp_index + 1) / len(config.upsamp_iters))
            )
            TrainState.initialize(
                config,
                grid_dim=grid_dim,
                prng_key=jax.random.PRNGKey(94709),
            ).training_step(next(iter(minibatches)))
    del frontload_jit_compile_time

    # Run!
    print("Training with config:", config)
    loop_metrics: fifteen.utils.LoopMetrics
    for loop_metrics in tqdm(
        fifteen.utils.range_with_metrics(config.n_iters - train_state.step),
        desc="Training",
    ):
        # Load minibatch.
        minibatch = next(minibatches)
        assert minibatch.get_batch_axes() == (config.minibatch_size,)
        assert minibatch.colors.shape == (4096, 3)

        # Training step.
        log_data: fifteen.experiments.TensorboardLogData
        train_state, log_data = train_state.training_step(minibatch)

        # Log & checkpoint.
        train_step = int(train_state.step)
        experiment.log(
            log_data.merge_scalars(
                {"train/iterations_per_sec": loop_metrics.iterations_per_sec}
            ),
            step=train_step,
            log_scalars_every_n=5,
            log_histograms_every_n=100,
        )
        if train_step % 1000 == 0:
            experiment.save_checkpoint(
                train_state,
                step=train_state.step,
                keep_every_n_steps=2000,
            )

        # Grid upsampling. We linearly interpolate between the initial and final grid
        # dimensionalities.
        if train_step in config.upsamp_iters:
            upsamp_index = config.upsamp_iters.index(train_step)
            train_state = train_state.resize_grid(
                new_grid_dim=int(
                    config.grid_dim_init
                    + (config.grid_dim_final - config.grid_dim_init)
                    * ((upsamp_index + 1) / len(config.upsamp_iters))
                )
            )
