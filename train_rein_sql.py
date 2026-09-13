#!/usr/bin/env python3
"""verl 0.9 launcher for the project-specific fixed-rollout REIN-SQL trainer."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


class ReinSQLTaskRunner(BaseTaskRunner):
    """Standard verl 0.9 task runner that installs ``RayREINSQLTrainer``."""

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"REIN-SQL TaskRunner hostname={socket.gethostname()} pid={os.getpid()}", flush=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_data_config = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
        with open_dict(val_data_config):
            val_data_config.filter_overlong_prompts = config.data.get("val_filter_overlong_prompts", False)
            val_data_config.max_prompt_length = config.data.get(
                "val_max_prompt_length", config.data.max_prompt_length
            )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            val_data_config,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )

        from rein_sql_trainer import RayREINSQLTrainer

        trainer = RayREINSQLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="configs", config_name="rein_sql_v3_h100_1gpu", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ReinSQLTaskRunner))


if __name__ == "__main__":
    main()
