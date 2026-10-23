# Installation

Choose the full environment for training, or the standalone package for offline
motion adaptation. Full-pipeline commands run from the repository root;
standalone commands run from `terrain_motion_adaptation/`.

## Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal so `uv` is available on `PATH`.

## Install the full pipeline

After cloning the repository, change to the repository root and run:

```bash
uv sync --locked
```

This creates one `.venv` containing the training project, its RL module and the
local motion-adaptation package. Python is selected from `.python-version`,
currently 3.13.12. `uv.lock` supplies the dependency versions; use `uv run` to
execute commands in that environment.

GPU training uses the configured CUDA 12.8 PyTorch wheels. Linux with a compatible
NVIDIA driver is the environment used for the training checks. Verify GPU access:

```bash
uv run --locked python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Continue with [data preparation](data-preparation.md).

## Install motion adaptation only

For offline adaptation without PyTorch, mjlab or RSL-RL:

```bash
cd terrain_motion_adaptation
uv sync --locked
```

For browser-based motion replay, include the visualization extra:

```bash
uv sync --locked --extra visualization
```

Keep `--extra visualization` on standalone viewer commands. Headless replay only
needs the base installation. Full-pipeline users can use their root environment
for adaptation as well; a second installation is unnecessary.

Continue with [standalone motion adaptation](motion-adaptation.md).

For environment updates, see uv's
[locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/).
