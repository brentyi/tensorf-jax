import dataclasses
import pathlib

import dcargs
import fifteen
import jax

import tensorf.data
import tensorf.train_config
import tensorf.training


@dataclasses.dataclass
class Args:
    use_magic_vmap: bool
    hlo_target_path: pathlib.Path


args = dcargs.parse(Args)
# Default configuration for lego dataset.
config = tensorf.train_config.TensorfConfig(
    run_dir=pathlib.Path(f"./runs/lego-{fifteen.utils.timestamp()}"),
    dataset_path=pathlib.Path("./data/nerf_synthetic/lego"),
    dataset_type="blender",
    n_iters=30000,
    optimizer=tensorf.train_config.OptimizerConfig(),
    initial_aabb_min=(-0.6585, -1.1833, -0.4651),
    initial_aabb_max=(0.6636, 1.1929, 1.0512),
    appearance_feat_dim=48,
    density_feat_dim=16,
    feature_n_freqs=2,
    viewdir_n_freqs=2,
    grid_dim_init=128,
    grid_dim_final=300,
    upsamp_iters=(2000, 3000, 4000, 5500, 7000),
)

# Initialize training state.
train_state: tensorf.training.TrainState
train_state = tensorf.training.TrainState.initialize(
    config,
    grid_dim=config.grid_dim_init,
    prng_key=jax.random.PRNGKey(94709),
)

# Load dataset.
assert config.dataset_type == "blender"
dataloader = fifteen.data.InMemoryDataLoader(
    dataset=tensorf.data.rendered_rays_from_views(
        views=tensorf.data.load_blender_dataset(
            config.dataset_path,
            split="train",
            progress_bar=True,
        )
    ),
    minibatch_size=config.minibatch_size,
)
minibatches = fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0)
minibatch = next(iter(minibatches))

with fifteen.utils.stopwatch("JIT compile"):
    train_state = jax.block_until_ready(
        tensorf.training.TrainState.training_step(
            train_state, minibatch, use_magic_vmap=args.use_magic_vmap
        )
    )[0]
with fifteen.utils.stopwatch("Step 100x"):
    for i in range(100):
        train_state = jax.block_until_ready(
            tensorf.training.TrainState.training_step(
                train_state, minibatch, use_magic_vmap=args.use_magic_vmap
            )
        )[0]
with fifteen.utils.stopwatch("Writing HLO"):
    args.hlo_target_path.write_text(
        tensorf.training.TrainState.training_step.lower(
            train_state,
            minibatch,
            use_magic_vmap=args.use_magic_vmap,
        )
        .compile()
        .compiler_ir()[0]
        .to_string()
    )
