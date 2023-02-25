from __future__ import annotations

import math
import random
from typing import Tuple

import fifteen
import flax
import jax
import jax_dataclasses as jdc
import optax
from jax import numpy as jnp
from jax._src.tree_util import Leaf
from tqdm.auto import tqdm
from typing_extensions import Annotated, assert_never

from . import (
    data,
    networks,
    render_bounded,
    render_common,
    render_real,
    tensor_vm,
    train_config,
    utils,
)


@jdc.pytree_dataclass
class TrainState(jdc.EnforcedAnnotationsMixin):
    config: jdc.Static[train_config.TensorfConfig]

    # Representation/parameters.
    appearance_mlp: jdc.Static[networks.FeatureMlp]
    learnable_params: render_common.LearnableParams

    # Optimizer.
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState

    # Current axis-aligned bounding box.
    aabb: Annotated[jnp.ndarray, jnp.floating, (2, 3)]

    # Misc.
    prng_key: jax.random.KeyArray
    step: Annotated[jnp.ndarray, jnp.integer, ()]

    @staticmethod
    @jdc.jit
    def initialize(
        config: jdc.Static[train_config.TensorfConfig],
        density_grid_dim: jdc.Static[int],
        app_grid_dim: jdc.Static[int],
        prng_key: jax.random.KeyArray,
        num_cameras: jdc.Static[int],
    ) -> TrainState:
        prng_keys = jax.random.split(prng_key, 5)
        normal_init = jax.nn.initializers.normal(stddev=0.1)

        def make_mlp() -> Tuple[networks.FeatureMlp, flax.core.FrozenDict]:
            dummy_features = jnp.zeros((1, config.appearance_feat_dim * 3))
            dummy_viewdirs = jnp.zeros((1, 3))
            appearance_mlp = networks.FeatureMlp(
                feature_n_freqs=config.feature_n_freqs,
                viewdir_n_freqs=config.viewdir_n_freqs,
                # If num_cameras is set, camera embeddings are enabled.
                num_cameras=num_cameras if config.camera_embeddings else None,
                output_dim=4 if config.density_from_appearance_mlp else 3,
            )
            dummy_camera_indices = jnp.zeros((1,), dtype=jnp.uint32)

            appearance_mlp_params = appearance_mlp.init(
                prng_keys[0],
                features=dummy_features,
                viewdirs=dummy_viewdirs,
                camera_indices=dummy_camera_indices,
            )
            return appearance_mlp, appearance_mlp_params

        appearance_mlp, appearance_mlp_params = make_mlp()

        learnable_params = render_common.LearnableParams(
            appearance_mlp_params=appearance_mlp_params,
            appearance_tensor=tensor_vm.TensorVM.initialize(
                grid_dim=app_grid_dim,
                per_axis_channel_dim=config.appearance_feat_dim,
                init=normal_init,
                prng_key=prng_keys[1],
                dtype=jnp.float32,  # Main copy of parameters are always float32.
            ),
            density_tensors=(
                tensor_vm.TensorVM.initialize(
                    grid_dim=density_grid_dim,
                    per_axis_channel_dim=config.density_feat_dim,
                    init=normal_init,
                    prng_key=prng_keys[2],
                    dtype=jnp.float32,
                ),
            ),
            bounded_scene=config.bounded_scene,
        )
        optimizer = TrainState._make_optimizer(config.optimizer, config.bounded_scene)
        optimizer_state = optimizer.init(learnable_params)

        return TrainState(
            config=config,
            appearance_mlp=appearance_mlp,
            learnable_params=learnable_params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            aabb=jnp.array([config.initial_aabb_min, config.initial_aabb_max]),
            prng_key=prng_keys[4],
            step=jnp.array(0),
        )

    @jdc.jit(donate_argnums=0)
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
            learnable_params: render_common.LearnableParams,
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            # Compute sample counts from grid dimensionality.
            # TODO: move heuristics into config?
            density_grid_dim = self.learnable_params.density_tensors[0].grid_dim()
            density_samples_per_ray = int(
                math.sqrt(3 * density_grid_dim**2)
                * self.config.train_ray_sample_multiplier
            )

            app_grid_dim = self.learnable_params.appearance_tensor.grid_dim()
            appearance_samples_per_ray = 64  # int(
            #      math.sqrt(3 * app_grid_dim**2)
            #      * self.config.train_ray_sample_multiplier
            #      * 0.2
            #  )

            # Render and compute loss.
            if self.config.bounded_scene:
                rendered = render_bounded.render_rays(
                    appearance_mlp=self.appearance_mlp,
                    learnable_params=learnable_params,
                    aabb=self.aabb,
                    rays_wrt_world=minibatch.rays_wrt_world,
                    prng_key=render_prng_key,
                    config=render_common.RenderConfig(
                        near=self.config.render_near,
                        far=self.config.render_far,
                        mode=render_common.RenderMode.RGB,
                        density_samples_per_ray=(density_samples_per_ray,),
                        appearance_samples_per_ray=appearance_samples_per_ray,
                    ),
                    dtype=compute_dtype,
                )
                interlevel_loss = 0.0
            else:
                rendered, interlevel_loss = render_real.render_rays(
                    appearance_mlp=self.appearance_mlp,
                    learnable_params=learnable_params,
                    aabb=self.aabb,
                    rays_wrt_world=minibatch.rays_wrt_world,
                    prng_key=render_prng_key,
                    config=render_common.RenderConfig(
                        near=self.config.render_near,
                        far=self.config.render_far,
                        mode=render_common.RenderMode.RGB,
                        density_samples_per_ray=(density_samples_per_ray,),
                        appearance_samples_per_ray=appearance_samples_per_ray,
                    ),
                    proposal_anneal_factor=1.0,
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
            loss = mse + interlevel_loss  # TODO: add regularization terms.

            log_data = fifteen.experiments.TensorboardLogData(
                scalars={
                    "mse": mse,
                    "psnr": utils.psnr_from_mse(mse),
                    "interlevel_loss": interlevel_loss,
                }
            )
            return loss * self.config.loss_scale, log_data

        # Compute gradients.
        log_data: fifteen.experiments.TensorboardLogData
        grads: render_common.LearnableParams
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
            jax.tree_util.tree_leaves(grads_unscaled)[0].dtype, jnp.float32
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
        bounded_scene: bool,
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
                mask=render_common.LearnableParams(
                    appearance_mlp_params=True,  # type: ignore
                    appearance_tensor=False,  # type: ignore
                    density_tensors=False,  # type: ignore
                    bounded_scene=bounded_scene,
                ),
            ),
            # Apply tensor decomposition learning rate. Note the negative sign needed
            # for gradient descent.
            optax.masked(
                optax.scale(-config.lr_init_tensor),
                mask=render_common.LearnableParams(
                    appearance_mlp_params=False,  # type: ignore
                    appearance_tensor=True,  # type: ignore
                    density_tensors=True,  # type: ignore
                    bounded_scene=bounded_scene,
                ),
            ),
        )

    def resize_grid(
        self,
        new_density_grid_dim: int,
        new_app_grid_dim: int,
    ) -> TrainState:
        """Resize the grid underlying a training state by linearly interpolating grid
        parameters."""
        with jdc.copy_and_mutate(self, validate=False) as resized:
            num_density_tensors = len(resized.learnable_params.density_tensors)

            # Resample the feature grids, with linear interpolation.
            resized.learnable_params.density_tensors = tuple(
                resized.learnable_params.density_tensors[i].resize(new_density_grid_dim)
                for i in range(num_density_tensors)
            )
            resized.learnable_params.appearance_tensor = (
                resized.learnable_params.appearance_tensor.resize(new_app_grid_dim)
            )

            # Perform some nasty surgery to resample the momentum parameters as well.
            adam_state = resized.optimizer_state[0]
            assert isinstance(adam_state, optax.ScaleByAdamState)
            nu: render_common.LearnableParams = adam_state.nu
            mu: render_common.LearnableParams = adam_state.mu
            resized.optimizer_state = (
                adam_state._replace(  # NamedTuple `_replace()`.
                    nu=jdc.replace(
                        nu,
                        density_tensors=tuple(
                            nu.density_tensors[i].resize(new_density_grid_dim)
                            for i in range(num_density_tensors)
                        ),
                        appearance_tensor=nu.appearance_tensor.resize(new_app_grid_dim),
                    ),
                    mu=jdc.replace(
                        mu,
                        density_tensors=tuple(
                            mu.density_tensors[i].resize(new_density_grid_dim)
                            for i in range(num_density_tensors)
                        ),
                        appearance_tensor=mu.appearance_tensor.resize(new_app_grid_dim),
                    ),
                ),
            ) + resized.optimizer_state[1:]
        return resized


def run_training_loop(
    config: train_config.TensorfConfig,
    restore_checkpoint: bool = False,
    clear_existing: bool = False,
) -> None:
    """Full training loop implementation."""

    # Set up our experiment: for checkpoints, logs, metadata, etc.
    experiment = fifteen.experiments.Experiment(data_dir=config.run_dir)
    if restore_checkpoint:
        experiment.assert_exists()
        config = experiment.read_metadata("config", train_config.TensorfConfig)
    else:
        if clear_existing:
            experiment.clear()
        else:
            experiment.assert_new()
        experiment.write_metadata("config", config)

    # Load dataset.
    dataset = data.make_dataset(
        config.dataset_type,
        config.dataset_path,
        config.scene_scale,
    )
    num_cameras = len(dataset.get_cameras())
    experiment.write_metadata("num_cameras", num_cameras)

    # Initialize training state.
    train_state: TrainState
    train_state = TrainState.initialize(
        config,
        density_grid_dim=config.density_grid_dim_init,
        app_grid_dim=config.app_grid_dim_init,
        prng_key=jax.random.PRNGKey(94709),
        num_cameras=num_cameras,
    )
    if restore_checkpoint:
        train_state = experiment.restore_checkpoint(train_state)

    dataloader = data.CachedNerfDataloader(
        dataset=dataset, minibatch_size=config.minibatch_size
    )
    minibatches = fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0)
    minibatches = iter(minibatches)

    # Run!
    print("Training with config:", config)
    loop_metrics: fifteen.utils.LoopMetrics
    for loop_metrics in tqdm(
        fifteen.utils.range_with_metrics(config.n_iters - int(train_state.step)),
        desc="Training",
    ):
        # Load minibatch.
        minibatch = next(minibatches)
        assert minibatch.get_batch_axes() == (config.minibatch_size,)
        assert minibatch.colors.shape == (config.minibatch_size, 3)

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
                step=int(train_state.step),
                keep_every_n_steps=2000,
            )

        # Grid upsampling. We linearly interpolate between the initial and final grid
        # dimensionalities.
        if train_step in config.upsamp_iters:
            upsamp_index = config.upsamp_iters.index(train_step)
            train_state = train_state.resize_grid(
                new_density_grid_dim=int(
                    config.density_grid_dim_init
                    + (config.density_grid_dim_final - config.density_grid_dim_init)
                    * ((upsamp_index + 1) / len(config.upsamp_iters))
                ),
                new_app_grid_dim=int(
                    config.app_grid_dim_init
                    + (config.app_grid_dim_final - config.app_grid_dim_init)
                    * ((upsamp_index + 1) / len(config.upsamp_iters))
                ),
            )
