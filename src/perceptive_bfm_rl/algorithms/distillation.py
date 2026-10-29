# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    student: MLPModel
    """The student model."""

    teacher: MLPModel
    """The teacher model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        num_mini_batches: int = 4,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        symmetry_bc_coef: float = 1.0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,  # handle unused config parameters
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.student.parameters(), lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        # Recurrent students backpropagate through fixed-length windows.
        # Non-recurrent students update once per pooled minibatch.
        self.gradient_length = gradient_length if self.student.is_recurrent else 1

        # Optional symmetry supervision for non-recurrent students.
        if symmetry_cfg is not None:
            # Resolve the data augmentation function (supports string names or direct callables)
            symmetry_cfg["data_augmentation_func"] = resolve_callable(
                symmetry_cfg["data_augmentation_func"]
            )
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the student policy is compatible with symmetry. The extra
            # forward pass in the symmetric branch would advance/corrupt hidden
            # state that the main training loop tracks separately. Deliberately
            # not checking self.teacher.is_recurrent too (unlike PPO's
            # actor.is_recurrent or critic.is_recurrent): _symmetric_teacher_targets
            # re-queries the teacher on mirrored obs, which would hit the same
            # failure mode for a recurrent teacher, but no teacher model in this
            # repo is ever recurrent (MLPModel.is_recurrent is always False) --
            # revisit if that changes.
            if self.student.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None
        self.symmetry_bc_coef = symmetry_bc_coef

        # Initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}"
            )

        self.num_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Compute the actions
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.privileged_actions = self.teacher(obs).detach()
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.student.update_normalization(obs)
        # Reset sequence memory at training boundaries without changing logger dones.
        if "segment_truncations" in extras:
            segments = extras["segment_truncations"].to(device=self.device, dtype=torch.bool)
            dones = (dones.bool() | segments).to(dtype=dones.dtype)
        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def _symmetric_teacher_targets(self, obs: TensorDict) -> tuple[TensorDict, torch.Tensor] | None:
        """Mirror obs via the configured symmetry transform, then re-query the
        frozen teacher on the mirrored obs for a fresh mirrored target action.
        Re-queries rather than algebraically mirroring an already-computed
        teacher action -- does not assume the teacher is itself exactly
        mirror-equivariant. Returns None when symmetry is disabled."""
        if self.symmetry is None:
            return None
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        mirrored_obs, _ = data_augmentation_func(obs=obs, actions=None, env=self.symmetry["env"])
        with torch.no_grad():
            mirrored_teacher_actions = self.teacher(mirrored_obs)
        return mirrored_obs, mirrored_teacher_actions

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op since distillation does not use return targets."""
        # Not needed for distillation
        pass

    def _generate_batches(self):
        """Pool rollout observations, targets and dones across time and environments,
        then yield shuffled minibatches for a non-recurrent student."""
        observations = self.storage.observations.flatten(0, 1)
        privileged_actions = self.storage.privileged_actions.flatten(0, 1)
        dones = self.storage.dones.flatten(0, 1)
        batch_size = observations.shape[0]
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(mini_batch_size * self.num_mini_batches, device=self.device)
        for i in range(self.num_mini_batches):
            batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            yield RolloutStorage.Batch(
                observations=observations[batch_idx],
                privileged_actions=privileged_actions[batch_idx],
                dones=dones[batch_idx],
            )

    def _generate_recurrent_chunks(self, seq_len: int, num_mini_batches: int):
        """Yield shuffled fixed-length recurrent windows and their validity masks.

        Incomplete trailing windows are dropped. Each window starts with a blank
        hidden state, so its first step is masked out, as is each step immediately
        following an environment reset. Steps within each window retain their order."""
        num_envs = self.storage.num_envs
        num_steps = self.storage.num_transitions_per_env
        num_windows_per_env = num_steps // seq_len
        if num_windows_per_env == 0:
            raise ValueError(
                f"gradient_length ({seq_len}) exceeds num_steps_per_env ({num_steps}); "
                "no complete window fits in one rollout."
            )
        num_chunks = num_windows_per_env * num_envs
        window_of = torch.arange(num_chunks, device=self.device) // num_envs
        env_of = torch.arange(num_chunks, device=self.device) % num_envs

        mini_batch_size = num_chunks // num_mini_batches
        perm = torch.randperm(mini_batch_size * num_mini_batches, device=self.device)

        for i in range(num_mini_batches):
            chunk_idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
            window_idx = window_of[chunk_idx]
            env_idx = env_of[chunk_idx]

            steps = []
            valid = torch.zeros(seq_len, mini_batch_size, dtype=torch.bool, device=self.device)
            # Step 0 of every window is always invalid (blank hidden state).
            prev_done = torch.ones(mini_batch_size, dtype=torch.bool, device=self.device)
            for s in range(seq_len):
                t = window_idx * seq_len + s
                steps.append(
                    RolloutStorage.Batch(
                        observations=self.storage.observations[t, env_idx],
                        privileged_actions=self.storage.privileged_actions[t, env_idx],
                        dones=self.storage.dones[t, env_idx],
                    )
                )
                valid[s] = ~prev_done
                prev_done = steps[-1].dones.view(-1).bool()
            yield steps, valid

    def _update_recurrent(self) -> dict[str, float]:
        """Update a recurrent student with shuffled fixed-length windows.

        Reset hidden states at window boundaries and environment resets. Each
        minibatch uses one masked mean loss and one optimizer step; the loss
        denominator includes both valid and masked elements. Symmetry is not
        supported on this path."""
        self.num_updates += 1
        rollout_hidden_state = self.student.get_hidden_state()
        mean_behavior_loss = 0.0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            for steps, valid in self._generate_recurrent_chunks(
                self.gradient_length, self.num_mini_batches
            ):
                self.student.reset()  # blank hidden state at the start of every window
                total_loss = 0.0
                total_elems = 0
                for s, batch in enumerate(steps):
                    actions = self.student(batch.observations)
                    step_loss = self.loss_fn(actions, batch.privileged_actions, reduction="none")
                    mask = valid[s].unsqueeze(-1).float()
                    total_loss = total_loss + (step_loss * mask).sum()
                    total_elems += step_loss.numel()
                    self.student.reset(batch.dones.view(-1))
                # Mask invalid elements while retaining the total element count.
                loss = total_loss / total_elems
                mean_behavior_loss += loss.item()
                cnt += 1

                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.student.detach_hidden_state()

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.student.reset(hidden_state=rollout_hidden_state)
        loss_dict = {"behavior": mean_behavior_loss}
        return loss_dict

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        if self.student.is_recurrent:
            return self._update_recurrent()

        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            for batch in self._generate_batches():
                # Inference of the student for gradient computation
                actions = self.student(batch.observations)

                # Behavior cloning loss
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Symmetric BC loss: direct supervision on the mirrored pair,
                # not a self-consistency regularizer -- same weight as the
                # ordinary BC loss by default (symmetry_bc_coef=1.0).
                symmetric = self._symmetric_teacher_targets(batch.observations)
                if symmetric is not None:
                    mirrored_obs, mirrored_teacher_actions = symmetric
                    mirrored_student_actions = self.student(mirrored_obs)
                    symmetry_loss = self.loss_fn(mirrored_student_actions, mirrored_teacher_actions)
                    loss = loss + self.symmetry_bc_coef * symmetry_loss

                # Gradient step. gradient_length is forced to 1 for a
                # non-recurrent student (see __init__), so this fires every
                # iteration -- one immediate step per pooled+shuffled
                # minibatch, no cross-minibatch accumulation.
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0

        mean_behavior_loss /= cnt
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for the student and keep the teacher in eval mode."""
        self.student.train()
        # Teacher is always in eval mode
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for student and teacher models."""
        self.student.eval()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, determine what to load automatically
        if load_cfg is None and "actor_state_dict" in loaded_dict:  # Load from RL training
            load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
        elif load_cfg is None:  # Load from distillation training
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("student"):
            self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"],
                strict=strict,
            )
            self.teacher_loaded = True
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.student

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> Distillation:
        """Construct the distillation algorithm."""
        # Resolve class callables
        alg_class: type[Distillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["student", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Distillation is not compatible with RND.
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        cfg["algorithm"]["rnd_cfg"] = None
        # Symmetry: resolve the same way PPO does. None (default) is a no-op.
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        student: MLPModel = student_class(
            obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]
        ).to(device)
        print(f"Student Model: {student}")
        teacher: MLPModel = teacher_class(
            obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]
        ).to(device)
        print(f"Teacher Model: {teacher}")

        # Initialize the storage
        storage = RolloutStorage(
            "distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # Initialize the algorithm
        alg: Distillation = alg_class(
            student,
            teacher,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.student.state_dict(), self.teacher.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.student.load_state_dict(model_params[0])
        self.teacher.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [
            param.grad.view(-1) for param in self.student.parameters() if param.grad is not None
        ]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.student.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
