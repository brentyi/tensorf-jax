# tensorf-jax

JAX implementation of
[Tensorial Radiance Fields](https://apchenstu.github.io/TensoRF/), written as an
exercise.

```
@misc{TensoRF,
      title={TensoRF: Tensorial Radiance Fields},
      author={Anpei Chen and Zexiang Xu and Andreas Geiger and and Jingyi Yu and Hao Su},
      year={2022},
      eprint={2203.09517},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We don't attempt to reproduce the original paper exactly, but can achieve decent
results after 5~10 minutes of training:

![Lego rendering GIF](./lego_render.gif)

## Instructions

1. Download `nerf_synthetic` dataset:
   [Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
   With the default training script arguments, we expect this to be extracted to
   `./data`, eg `./data/nerf_synthetic/lego`.

2. Install dependencies. Probably you want the GPU version of JAX; see the
   [official instructions](https://github.com/google/jax#Installation). Then:

   ```bash
   pip install -r requirements.txt
   ```

3. To print training options:

   ```bash
   python ./train_lego.py --help
   ```

4. To monitor training, we use Tensorboard:

   ```bash
   tensorboard --logdir=./runs/
   ```

5. To generate some renders, visit `./render.ipynb` in Jupyter. Or:

   ```bash
   python ./render_360.py --help
   ```

## Differences from the PyTorch implementation

Things aren't totally matched to the official implementation:

- The official implementation relies heavily on masking operations to improve
  runtime (for example, by using a weight threshold for sampled points). These
  require dynamic shapes and are currently difficult to implement in JAX, so we
  replace them with workarounds like weighted sampling.
- Several training details that would likely improve performance are not yet
  implemented: bounding box refinement, ray filtering, regularization, etc.
- We include mixed-precision training, which can speed training throughput up by
  a significant factor. (is this actually faster in terms of wall-clock time?
  unclear)

## References

Implementation details are based loosely on the original PyTorch implementation
[apchsenstu/TensoRF](https://github.com/apchenstu/TensoRF).

[unixpickle/learn-nerf](https://github.com/unixpickle/learn-nerf) and
[google-research/jaxnerf](https://github.com/google-research/google-research/tree/master/jaxnerf)
were also really helpful for understanding core NeRF concepts + connecting them
to JAX!

## To-do

- [x] Blender dataloading
- [x] Main implementation
  - [x] Point sampling
  - [x] Feature MLP
  - [x] Rendering
  - [x] VM decomposition
    - [x] Basic implementation
    - [x] Vectorized
- [x] Training
  - [x] Learning rate scheduler
    - [x] ADAM + grouped LR
    - [x] Exponential decay
    - [x] Reset decay after upsampling
  - [x] Running
  - [x] Checkpointing
  - [x] Logging
    - [x] Loss
    - [x] PSNR
    - [ ] Test metrics
    - [ ] Test images
  - [ ] Ray filtering
  - [ ] Bounding box refinement
  - [x] Incremental upsampling
  - [ ] Regularization terms
- [x] Performance
  - [x] Weight thresholding for computing appearance features
    - [x] per ray top-k
    - [x] global top-k (bad & deleted)
  - [x] Mixed-precision
    - [x] implemented
    - [x] stable
  - [ ] Multi-GPU (should be quick)
- [x] Rendering
  - [x] RGB
  - [x] Depth (median)
  - [x] Depth (mean)
  - [x] Batching
  - [x] Generate some GIFs
- [ ] Misc engineering
  - [x] Actions
  - [ ] Understand vmap performance differences
        ([details](https://github.com/google/jax/discussions/10332))
