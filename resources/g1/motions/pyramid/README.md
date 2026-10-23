# G1 terrain motion examples

These examples show the results of terrain motion optimization. Replay them
directly to inspect the optimized motions without running the optimization
pipeline.

Three motion excerpts are provided for flat ground and the four formal pyramid
terrain variants.

| Motion | Movement | Duration | Frames at 50 FPS |
| --- | --- | ---: | ---: |
| `walk` | Walking | 10.00 s | 500 |
| `crawl_and_back_handspring` | Crawling followed by a back handspring | 11.52 s | 576 |
| `backward_roll` | Backward roll followed by standing up | 10.00 s | 500 |

Each group contains the same three relative motion paths:

```text
<group>/
├── manifest.json
├── walk/motion.npz
├── crawl_and_back_handspring/motion.npz
└── backward_roll/motion.npz
```

The groups are `flat`, `hf_pyramid_slope_lv0`, `hf_pyramid_slope_inv_lv0`,
`pyramid_stairs_lv0` and `pyramid_stairs_inv_lv0`. Each NPZ stores float32 `qpos`
and scalar `fps`; the group manifest names the 29 joints in MuJoCo order.
Body states and velocities are reconstructed by runtime FK.

## Replay

From the repository root:

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir resources/g1/motions/pyramid \
  --motion crawl_and_back_handspring/motion.npz \
  --viz-ee-contact
```

Select a terrain and motion in the viewer. Pause, slow playback and frame
scrubbing help inspect the adapted references and estimated contacts.
