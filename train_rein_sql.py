#!/usr/bin/env python3
"""Hydra/Ray launcher for the project-specific REIN-SQL trainer."""

import os
import socket
import sys

import ray
from omegaconf import OmegaConf

from rein_sql_trainer import RayREINSQLTrainer
from verl.trainer import main_ppo


@ray.remote(num_cpus=1)
class ReinSQLTaskRunner:
    """verl TaskRunner equivalent that instantiates RayREINSQLTrainer."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def _add_workers(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker
            actor_cls = ActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            actor_cls = AsyncActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
            actor_cls = AsyncActorRolloutRefWorker
        else:
            raise NotImplementedError("Unsupported actor strategy")

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_cls)
        self.mapping[Role.ActorRollout] = "global_pool"

        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError("Unsupported critic strategy")
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

        if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(actor_cls)
            self.mapping[Role.RefPolicy] = "global_pool"
        return actor_cls, RayWorkerGroup

    def run(self, config):
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"REIN-SQL TaskRunner hostname={socket.gethostname()} pid={os.getpid()}", flush=True)
        OmegaConf.resolve(config)
        actor_cls, ray_worker_group_cls = self._add_workers(config)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        pool = ResourcePoolManager(
            resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping=self.mapping,
        )
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        trainer = RayREINSQLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=pool,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer.init_workers()
        trainer.fit()


_original_run_ppo = main_ppo.run_ppo


def _run_ppo_with_rein_trainer(config):
    return _original_run_ppo(config, task_runner_class=ReinSQLTaskRunner)


main_ppo.run_ppo = _run_ppo_with_rein_trainer


def _normalize_config_path() -> None:
    if "--config-path" in sys.argv:
        idx = sys.argv.index("--config-path")
        if idx + 1 < len(sys.argv) and not os.path.isabs(sys.argv[idx + 1]):
            sys.argv[idx + 1] = os.path.join(os.path.dirname(__file__), sys.argv[idx + 1])


if __name__ == "__main__":
    _normalize_config_path()
    main_ppo.main()
