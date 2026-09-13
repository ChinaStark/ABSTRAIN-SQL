"""REIN-style capability-aware abstention trainer for Text-to-SQL RL.

This trainer uses fixed-size on-policy groups:

1. Every prompt receives exactly ``actor_rollout_ref.rollout.n`` trajectories.
2. If any trajectory is correct the prompt is ``WITHIN`` capability; if all
   are wrong it is ``BEYOND`` capability.
3. ``WITHIN`` groups reward correct SQL and penalize ``ABSTAIN``. ``BEYOND``
   groups reward ``ABSTAIN`` while wrong SQL receives zero reward.
4. Every generated trajectory stays in its original GRPO group. There is no
   additional sampling, online filtering, or correct/wrong quota selection.

The SQL execution function stays external and is reused both for online capability
probing and ordinary validation. Boundary-aware rewards are cached directly into
``token_level_scores`` before GRPO advantage computation.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.skip import SkipManager

from boundary import BEYOND, WITHIN, RewardValues, trajectory_reward
from output_protocol import parse_response
from sql_reward import score_sql


class RayREINSQLTrainer(RayPPOTrainer):
    """PPO/GRPO trainer with REIN-style capability-aware SQL abstention."""

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _rein_cfg(self):
        """Return fixed-rollout REIN-SQL reward configuration."""
        cfg = self.config.algorithm.get("rein_sql", None)
        defaults = dict(correct_threshold=1.0)
        if cfg is None:
            merged = defaults
            reward_cfg = {}
        else:
            merged = dict(defaults)
            for k in defaults:
                v = cfg.get(k, None)
                if v is not None:
                    merged[k] = v
            reward_cfg = cfg.get("reward_values", {}) or {}

        merged["group_size"] = int(self.config.actor_rollout_ref.rollout.n)
        # v3 RewardValues defaults match the final reward ladder; only apply
        # overrides for fields that actually exist in the dataclass.
        _rv_fields = RewardValues.__dataclass_fields__
        rv_kwargs = {
            name: float(reward_cfg[name])
            for name in _rv_fields
            if name in reward_cfg
        }
        merged["reward_values"] = RewardValues(**rv_kwargs)
        return merged

    def _load_scoring_fn(self):
        """Load the external SQL execution correctness function."""
        if getattr(self, "_scoring_fn", None) is not None:
            return self._scoring_fn
        fn = get_custom_reward_fn(self.config)
        assert fn is not None, (
            "REIN-SQL requires reward.custom_reward_function.path/name to be set "
            "to a SQL execution verifier."
        )
        self._scoring_fn = fn
        # Extract reward_kwargs for direct score_sql calls in _score_rollouts
        reward_cfg = self.config.reward.get("custom_reward_function", {}) or {}
        self._reward_kwargs = dict(reward_cfg.get("reward_kwargs", {}) or {})
        return fn

    def _decode_responses(self, gen_out: DataProto) -> list[str]:
        tokenizer = self.tokenizer
        prompts = gen_out.batch["prompts"]
        responses = gen_out.batch["responses"]
        attention_mask = gen_out.batch["attention_mask"]
        prompt_len = prompts.shape[-1]
        texts = []
        for i in range(len(gen_out)):
            valid_resp_len = int(attention_mask[i, prompt_len:].sum().item())
            resp_ids = responses[i, :valid_resp_len]
            texts.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
        return texts

    def _score_rollouts(self, gen_out: DataProto) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
        """Score final <answer> SQL and <draft_sql> with the same execution verifier."""
        self._load_scoring_fn()
        texts = self._decode_responses(gen_out)

        scoring_inputs = []
        parsed_meta = []
        for i, response_str in enumerate(texts):
            parsed = parse_response(response_str)
            gt = gen_out.non_tensor_batch["reward_model"][i].get("ground_truth", None)
            data_source = gen_out.non_tensor_batch.get("data_source", ["sql"] * len(gen_out))[i]
            extra_info = gen_out.non_tensor_batch.get("extra_info", [{}] * len(gen_out))[i]
            scoring_inputs.append((data_source, parsed.final_sql, parsed.draft_sql, gt, extra_info))
            parsed_meta.append({
                "abstained": parsed.abstained,
                "reflection_present": parsed.reflection_present,
                "format_valid": parsed.format_valid,
                "has_draft": parsed.has_draft,
                "has_answer": parsed.has_answer,
            })

        reward_kwargs = getattr(self, "_reward_kwargs", {})
        def score_one(item):
            data_source, final_sql, draft_sql, gt, extra_info = item
            def run(sql):
                if not sql:
                    return 0.0
                if sql.strip().upper() == "ABSTAIN":
                    return 0.0
                try:
                    score = score_sql(
                        data_source=data_source,
                        sql=sql,
                        ground_truth=gt,
                        extra_info=extra_info,
                        **reward_kwargs,
                    )
                    return float(score)
                except Exception as e:
                    print(f"[rein_sql] execution scoring error (treated as wrong): {e}")
                    return 0.0
            return run(final_sql), run(draft_sql)

        if not scoring_inputs:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, texts, parsed_meta

        max_workers = min(16, len(scoring_inputs))
        print(
            f"[rein_sql_reward] executing final+draft SQL for {len(scoring_inputs)} trajectories "
            f"with concurrency={max_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rein-sql-reward") as executor:
            pairs = list(executor.map(score_one, scoring_inputs))
        final_scores = np.asarray([x[0] for x in pairs], dtype=np.float32)
        draft_scores = np.asarray([x[1] for x in pairs], dtype=np.float32)
        # Log all rollouts to a best-effort side file next to trainer.logger_path.
        import time as _time

        logger_path = self.config.trainer.get("logger_path", None)
        rollout_log_path = os.path.join(os.path.dirname(logger_path), "rollouts.log") if logger_path else None
        try:
            if rollout_log_path:
                with open(rollout_log_path, "a") as _rf:
                    _rf.write("\n=== batch at %s ===\n" % _time.strftime("%H:%M:%S"))
                    for _i, (_t, _m, _fs, _ds) in enumerate(zip(texts, parsed_meta, final_scores, draft_scores)):
                        _rf.write("\n--- rollout %d ---\n" % _i)
                        _rf.write("format_valid: %s\n" % _m.get("format_valid", False))
                        _rf.write("abstained: %s\n" % _m.get("abstained", False))
                        _rf.write("final_score: %.2f\n" % float(_fs))
                        _rf.write("draft_score: %.2f\n" % float(_ds))
                        _rf.write("response (first 500): %s\n" % str(_t)[:500])
        except (OSError, IOError):
            pass  # log file not writable, skip
        return final_scores, draft_scores, texts, parsed_meta

    def _attach_meta(self, gen_out: DataProto, gen_in: DataProto) -> DataProto:
        for k in ("data_source", "reward_model", "extra_info", "uid"):
            if k in gen_in.non_tensor_batch and k not in gen_out.non_tensor_batch:
                gen_out.non_tensor_batch[k] = gen_in.non_tensor_batch[k]
        return gen_out

    def _fixed_rollout_one_gen_batch(self, gen_batch: DataProto, timing_raw: dict):
        """Fixed rollout: all prompts get exactly group_size rollouts, no adaptive collection.
        Still classifies WITHIN/BEYOND based on whether any rollout is correct."""
        acfg = self._rein_cfg()
        group_size = int(acfg["group_size"])
        reward_values = acfg["reward_values"]
        thr = float(acfg["correct_threshold"])

        n_prompts = len(gen_batch)
        gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)
        gen_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        gen_batch.meta_info["global_steps"] = self.global_steps

        size_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        padded_gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        with marked_timer("gen", timing_raw, "red"):
            gen_out = self.async_rollout_manager.generate_sequences(padded_gen_batch)
            gen_out = unpad_dataproto(gen_out, pad_size)
            gen_out.meta_info.pop("timing", None)
        gen_out = self._attach_meta(gen_out, gen_batch)

        with marked_timer("fixed_score", timing_raw, "yellow"):
            exec_scores, draft_scores, texts, parsed_meta = self._score_rollouts(gen_out)

        # Classify WITHIN/BEYOND per prompt: if any rollout correct → WITHIN, else BEYOND
        boundaries = []
        n_within = 0
        n_beyond = 0
        for p_idx in range(n_prompts):
            group_exec = exec_scores[p_idx * group_size : (p_idx + 1) * group_size]
            if any(s >= thr for s in group_exec):
                boundaries.append(WITHIN)
                n_within += 1
            else:
                boundaries.append(BEYOND)
                n_beyond += 1

        kept_rows = []
        kept_scores = []
        for i in range(len(gen_out)):
            p_idx = i // group_size
            row = gen_out.slice(i, i + 1)
            parsed = parsed_meta[i]
            abstained = bool(parsed["abstained"])
            final_exec = float(exec_scores[i])
            draft_exec = float(draft_scores[i])
            reflection_present = bool(parsed["reflection_present"])
            r = trajectory_reward(
                final_exec=final_exec, draft_exec=draft_exec,
                abstained=abstained, boundary=boundaries[p_idx],
                reflection_present=reflection_present,
                has_draft=bool(parsed["has_draft"]),
                has_answer=bool(parsed["has_answer"]),
                correct_threshold=thr, values=reward_values,
            )
            kept_rows.append(row)
            kept_scores.append(r)

        stats = {
            "rein/num_prompts": n_prompts,
            "rein/num_within": n_within,
            "rein/num_beyond": n_beyond,
            "rein/selected_abstain": sum(1 for m in parsed_meta if m["abstained"]),
            "rein/rollouts_per_prompt": int(group_size),
        }

        if not kept_rows:
            return None, [], stats

        batch = DataProto.concat(kept_rows)
        return batch, kept_scores, stats

    def _set_cached_token_level_scores(self, batch: DataProto, scores: list):
        """Write the cached per-trajectory SQL reward into a token-level score tensor.

        This avoids re-executing every SQL a second time in the reward stage. The scalar
        reward is placed on the last valid response token (outcome reward), matching the
        NaiveRewardManager convention that GRPO consumes.
        """
        prompt_len = batch.batch["prompts"].shape[-1]
        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        attention_mask = batch.batch["attention_mask"]
        for i, sc in enumerate(scores):
            valid_resp_len = int(attention_mask[i, prompt_len:].sum().item())
            if valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] = float(sc)
        batch.batch["token_level_scores"] = reward_tensor
        # seq-level reward for logging / GRPO metric compatibility
        batch.non_tensor_batch["seq_reward"] = reward_tensor.sum(dim=-1).numpy()
        return batch

    # ------------------------------------------------------------------ #
    # KL / ref logprob helper (same as DAPO recipe)
    # ------------------------------------------------------------------ #
    def _compute_logprob_related(self, batch: DataProto, metrics: dict, timing_raw: dict):
        from verl.trainer.ppo.core_algos import agg_loss

        # Match RayPPOTrainer.fit(): the FSDP inference path materializes this
        # meta field into the no-padding TensorDict used for log-prob forward.
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            metrics.update(
                {
                    "actor/entropy": entropy_agg.detach().item(),
                    "perf/mfu/actor_infer": old_log_prob_mfu,
                }
            )
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        return batch

    # ------------------------------------------------------------------ #
    # main training loop
    # ------------------------------------------------------------------ #
    def fit(self):
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.max_steps_duration = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)
        SkipManager.init(self.config)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        SkipManager.set_step(self.global_steps)

        group_size = int(self.config.actor_rollout_ref.rollout.n)
        print(
            f"REIN SQL source-data epoch: start={current_epoch}, "
            f"total={self.config.trainer.total_epochs}, resumed_global_steps={self.global_steps - 1}"
        )

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(new_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # One fixed generation call: every prompt gets exactly rollout.n samples.
                    batch, cached_scores, stats = self._fixed_rollout_one_gen_batch(gen_batch, timing_raw)
                    if batch is None:
                        raise RuntimeError("fixed rollout unexpectedly returned an empty batch")
                    expected_trajectories = len(gen_batch) * group_size
                    if len(batch) != expected_trajectories:
                        raise RuntimeError(
                            f"fixed rollout returned {len(batch)} trajectories; expected {expected_trajectories}"
                        )
                    metrics.update(stats)

                    self.checkpoint_manager.sleep_replicas()

                    # ---- attach cached SQL reward (avoid re-executing every SQL) ----
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    batch = self._set_cached_token_level_scores(batch, cached_scores)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ---- old/ref log prob ----
                    batch = self._compute_logprob_related(batch, metrics, timing_raw)

                    # ---- token_level_rewards (+ optional KL in reward) ----
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # ---- values (only if critic; GRPO has none) ----
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # ---- advantage (GRPO groups by uid; each selected group has 8 rows) ----
                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=group_size,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # ---- critic update ----
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # ---- actor update ----
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                # ---- validation ----
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # ---- metrics ----
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._shutdown_dump_executor()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                SkipManager.set_step(self.global_steps)

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = {}
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
        self._shutdown_dump_executor()
