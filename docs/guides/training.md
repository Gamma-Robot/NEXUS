# Training

Train a privileged teacher, then distill its actions into a depth-conditioned
student. Use the [full installation](installation.md#install-the-full-pipeline)
and [packed data](data-preparation.md#pack-the-training-dataset), and run commands
from the repository root.

Examples use local checkpoints and TensorBoard on **four GPUs**
(`--gpu-ids 0 1 2 3`), with **16,384 environments per GPU** (65,536 total).
For a single GPU, use `--gpu-ids 0`. Adjust `--env.scene.num-envs` to fit
your available GPU memory.

## Train the privileged teacher

```bash
uv run python -m perceptive_bfm.scripts.train \
  Perceptive-PyramidTerrain-Unitree-G1-Teacher-Symmetry \
  --env.scene.num-envs 16384 \
  --env.commands.motion.motion-file dataset/g1/pyramid \
  --agent.run-name teacher \
  --agent.logger tensorboard \
  --gpu-ids 0 1 2 3
```

The actor uses privileged motion/state observations and terrain elevation. Runs
are saved under `logs/rsl_rl/g1_perceptive/<timestamp>_teacher/`, with
`model_<iteration>.pt` checkpoints and resolved configurations in `params/`.

Inspect learning curves with:

```bash
uv run tensorboard --logdir logs/rsl_rl/g1_perceptive
```

## Distill the perceptive student

Choose a trained teacher checkpoint. Replace `<teacher_run>` with its complete
directory name, including the timestamp, and `<iteration>` with a saved iteration:

```bash
uv run python -m perceptive_bfm.scripts.train \
  Perceptive-PyramidTerrain-Unitree-G1-BC-Symmetry \
  --env.scene.num-envs 16384 \
  --env.commands.motion.motion-file dataset/g1/pyramid \
  --agent.load-run "<teacher_run>" \
  --agent.load-checkpoint "model_<iteration>.pt" \
  --agent.run-name student \
  --agent.logger tensorboard \
  --gpu-ids 0 1 2 3
```

The teacher stays frozen. The student learns its action targets from flat-ground
motion commands, proprioceptive history and depth images. BC tasks load the
selected checkpoint automatically; initializing from a teacher does not require
`--agent.resume True`.

Run and checkpoint selectors are regular expressions inside the configured
experiment directory. Select an unambiguous run/checkpoint pair. If changing
`--agent.experiment-name`, use the same experiment directory for teacher and
student so local checkpoint lookup finds the teacher.

## Resume a run

Select the matching task and a checkpoint from the run you want to continue:

```bash
uv run python -m perceptive_bfm.scripts.train \
  Perceptive-PyramidTerrain-Unitree-G1-BC-Symmetry \
  --env.scene.num-envs 16384 \
  --env.commands.motion.motion-file dataset/g1/pyramid \
  --agent.resume True \
  --agent.load-run "<student_run>" \
  --agent.load-checkpoint "model_<iteration>.pt" \
  --agent.run-name student_resume \
  --agent.logger tensorboard \
  --gpu-ids 0 1 2 3
```

Use a teacher or one-stage PPO task for the corresponding continuation. Each
continuation creates a new log directory. `--agent.max-iterations` specifies
the number of learning iterations passed to that run.

## One-stage perceptive PPO

To train the depth-conditioned actor directly with PPO:

```bash
uv run python -m perceptive_bfm.scripts.train \
  Perceptive-PyramidTerrain-Unitree-G1-Symmetry \
  --env.scene.num-envs 16384 \
  --env.commands.motion.motion-file dataset/g1/pyramid \
  --agent.run-name ppo \
  --agent.logger tensorboard \
  --gpu-ids 0 1 2 3
```

## Symmetry variants

Remove `-Symmetry` to select the corresponding task without symmetry. PPO uses
a mirror-consistency loss; distillation adds teacher supervision on augmented
observations. The [illustrated symmetry reference](../reference/symmetry.md)
shows how depth, elevation, state and actions transform together.

## GPU and run configuration

| Override | Purpose |
| --- | --- |
| `--gpu-ids 0 1 2 3` | Use four GPUs; `--gpu-ids 0` selects one GPU |
| `--env.scene.num-envs` | Set parallel environments per process |
| `--agent.max-iterations` | Set learning iterations |
| `--agent.save-interval` | Set checkpoint interval |
| `--agent.experiment-name` | Choose the directory under `logs/rsl_rl/` |

Multi-GPU runs shard motions across ranks and synchronize updates. Each rank
needs a nonempty assigned subset of the paired motions. Keep the local packed
dataset path explicit, including when loading a W&B checkpoint; automatic
motion-artifact lookup does not describe this grouped layout.

Inspect defaults with the task before `--help`:

```bash
uv run python -m perceptive_bfm.scripts.train \
  Perceptive-PyramidTerrain-Unitree-G1-Teacher-Symmetry --help
```

See [registered tasks](../reference/configuration.md#registered-tasks) for all six
IDs. Continue with [policy playback](playback.md#policy-playback).
