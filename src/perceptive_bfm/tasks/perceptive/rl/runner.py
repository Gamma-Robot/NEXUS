from pathlib import Path
from typing import cast

import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.models import CNNModel
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

from perceptive_bfm.rl.exporter_utils import get_base_metadata
from perceptive_bfm.rl.runner import MjlabDistillationRunner
from perceptive_bfm.tasks.perceptive.mdp import MotionCommand


class MotionTrackingSaveMixin:
    """Cooperative class/mixin to add ONNX export and metadata attachment functionality to runners."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = Path(path).absolute().parent
        filename = policy_path.name + ".onnx"
        writer = self.logger.writer
        wandb_run = wandb.run if isinstance(writer, WandbLogWriter) else None
        try:
            self.export_policy_to_onnx(str(policy_path), filename)
            policy = self.alg.get_policy()
            obs_groups = list(policy.obs_groups)
            if isinstance(policy, CNNModel):
                obs_groups.extend(policy.obs_groups_2d)
            run_name = wandb_run.name if wandb_run is not None else "local"
            metadata = get_base_metadata(self.env.unwrapped, run_name, obs_groups)
            motion_term = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
            metadata.update(
                {
                    "anchor_body_name": motion_term.cfg.anchor_body_name,
                    "body_names": list(motion_term.cfg.body_names),
                }
            )
            attach_metadata_to_onnx(str(policy_path / filename), metadata)
            if wandb_run is not None and self.cfg["upload_model"]:
                writer.save_file(str(policy_path / filename))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
        if wandb_run is not None and self.registry_name is not None:
            try:
                wandb_run.use_artifact(self.registry_name)
                self.registry_name = None
            except Exception as e:
                print(f"[WARN] W&B motion artifact association failed (training continues): {e}")


class MotionTrackingOnPolicyRunner(MotionTrackingSaveMixin, MjlabOnPolicyRunner):  # Caution to MRO!
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

        # Inject the true total training env-steps so the reward curriculum
        # (mdp.modify_reward_scale) computes progress against the resolved
        # max_iterations, honoring a CLI --agent.max_iterations override.
        if "num_steps_per_env" in train_cfg and "max_iterations" in train_cfg:
            self.env.unwrapped.curriculum_total_steps = int(train_cfg["num_steps_per_env"]) * int(
                train_cfg["max_iterations"]
            )


class MotionTrackingDistillationRunner(
    MotionTrackingSaveMixin, MjlabDistillationRunner  # Caution to MRO!
):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
