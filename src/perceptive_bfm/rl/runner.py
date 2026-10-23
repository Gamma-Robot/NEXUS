import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env.vec_env import VecEnv

from perceptive_bfm_rl.algorithms import Distillation


class MjlabDistillationRunner(MjlabOnPolicyRunner):
    alg: Distillation

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if "model_state_dict" in loaded_dict:
            print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
            model_state_dict = loaded_dict.pop("model_state_dict")
            student_state_dict = {}
            teacher_state_dict = {}
            actor_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith("student."):
                    new_key = key.replace("student.", "mlp.")
                    student_state_dict[new_key] = value
                elif key.startswith("student_obs_normalizer."):
                    new_key = key.replace("student_obs_normalizer.", "obs_normalizer.")
                    student_state_dict[new_key] = value
                elif key in ["std", "log_std"]:
                    student_state_dict[key] = value
                if key.startswith("teacher."):
                    new_key = key.replace("teacher.", "mlp.")
                    teacher_state_dict[new_key] = value
                elif key.startswith("teacher_obs_normalizer."):
                    new_key = key.replace("teacher_obs_normalizer.", "obs_normalizer.")
                    teacher_state_dict[new_key] = value
                if key.startswith("actor."):
                    new_key = key.replace("actor.", "mlp.")
                    actor_state_dict[new_key] = value
                elif key.startswith("actor_obs_normalizer."):
                    new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
                    actor_state_dict[new_key] = value
            loaded_dict["student_state_dict"] = student_state_dict
            loaded_dict["teacher_state_dict"] = teacher_state_dict
            if actor_state_dict:
                loaded_dict["actor_state_dict"] = actor_state_dict
        student_sd = loaded_dict.get("student_state_dict", {})
        if "std" in student_sd:
            student_sd["distribution.std_param"] = student_sd.pop("std")
        if "log_std" in student_sd:
            student_sd["distribution.log_std_param"] = student_sd.pop("log_std")
        actor_sd = loaded_dict.get("actor_state_dict", {})
        actor_sd.pop("distribution.std_param", None)
        actor_sd.pop("distribution.log_std_param", None)
        actor_sd.pop("std", None)
        actor_sd.pop("log_std", None)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.alg.teacher_loaded:
            raise ValueError(
                "Teacher model parameters not loaded. Please load a teacher model to distill."
            )
        super().learn(num_learning_iterations, init_at_random_ep_len)
