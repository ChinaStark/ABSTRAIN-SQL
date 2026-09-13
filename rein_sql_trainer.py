"""REIN-style capability-aware abstention trainer for Text-to-SQL RL.

This trainer reuses the supplied AdaptiveSQLTrainer integration pattern but changes
the online routing rule:

1. The first ``probe_n`` on-policy trajectories form a fixed empirical capability
   probe. SQL execution correctness is used as the verifier.
2. If any probe trajectory is correct the prompt is ``WITHIN`` capability; if all
   are wrong it is ``BEYOND`` capability.
3. ``WITHIN`` groups reward correct SQL and penalize ``ABSTAIN``. ``BEYOND``
   groups reward ``ABSTAIN`` while wrong SQL receives zero reward.
4. The trainer keeps the original adaptive generation budget so it can collect a
   mixed-reward GRPO group instead of discarding all-wrong prompts.

The SQL execution function stays external and is reused both for online capability
probing and ordinary validation. Boundary-aware rewards are cached directly into
``token_level_scores`` before GRPO advantage computation.
"""

import json
import os
import uuid
from collections import defaultdict
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
from verl.utils.rollout_skip import RolloutSkip

from boundary import BEYOND, WITHIN, RewardValues, trajectory_reward
from output_protocol import parse_response
from sql_reward import score_sql


class RayREINSQLTrainer(RayPPOTrainer):
    """PPO/GRPO trainer with REIN-style capability-aware SQL abstention."""

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _adaptive_cfg(self):
        """Return REIN-style adaptive-rollout config with defaults."""
        cfg = self.config.algorithm.get("adaptive_rollout", None)
        defaults = dict(
            enable=True,
            n_correct=4,
            n_wrong=4,
            probe_n=8,
            b_max=32,
            micro_rollout_n=8,
            correct_threshold=1.0,
            max_num_gen_batches=0,
            stage2_dump_path=None,
            resume_data_epoch=None,
            abstain_text="ABSTAIN",
            beyond_n_abstain=2,
            within_abstain_cap=2,
        )
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

        merged["group_size"] = int(merged["n_correct"]) + int(merged["n_wrong"])
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
            "adaptive_sql_rein requires reward.custom_reward_function.path/name to be set "
            "to a SQL execution verifier."
        )
        self._scoring_fn = fn
        # Extract reward_kwargs for direct score_sql calls in _score_rollouts
        reward_cfg = self.config.get("custom_reward_function", {}) or {}
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
        scoring_fn = self._load_scoring_fn()
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
        # Log all rollouts to separate file
        import time as _time
        try:
            with open("/home/ma-user/rollouts.log", "a") as _rf:
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

    @staticmethod
    def _pick_within_negatives(wrong_rows, abstain_rows, n_negative: int, abstain_cap: int):
        """Return ``[(row, abstained), ...]`` for a WITHIN negative set."""
        chosen = []
        take_a = min(len(abstain_rows), max(0, abstain_cap), n_negative)
        chosen.extend((r, True) for r in abstain_rows[:take_a])
        remain = n_negative - len(chosen)
        take_w = min(len(wrong_rows), remain)
        chosen.extend((r, False) for r in wrong_rows[:take_w])
        remain = n_negative - len(chosen)
        if remain > 0:
            chosen.extend((r, True) for r in abstain_rows[take_a : take_a + remain])
        return chosen

    def _fixed_rollout_one_gen_batch(self, gen_batch: DataProto, timing_raw: dict):
        """Fixed rollout: all prompts get exactly group_size rollouts, no adaptive collection.
        Still classifies WITHIN/BEYOND based on whether any rollout is correct."""
        acfg = self._adaptive_cfg()
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
            format_valid = bool(parsed["format_valid"])

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
            "adaptive/num_prompts": n_prompts,
            "adaptive/num_kept": n_prompts,
            "adaptive/num_within": n_within,
            "adaptive/num_beyond": n_beyond,
            "adaptive/num_too_easy": 0,
            "adaptive/num_within_unfilled": 0,
            "adaptive/num_beyond_no_signal": 0,
            "adaptive/num_no_signal": 0,
            "adaptive/selected_abstain": sum(1 for m in parsed_meta if m["abstained"]),
            "adaptive/avg_rollouts_per_prompt": float(group_size),
            "adaptive/max_rollouts_per_prompt": int(group_size),
        }

        if not kept_rows:
            return None, [], stats

        batch = DataProto.concat(kept_rows)
        return batch, kept_scores, stats

    def _adaptive_rollout_one_gen_batch(self, gen_batch: DataProto, timing_raw: dict):
        """Fixed-K capability probe + adaptive collection of learnable GRPO groups.

        REIN-style boundary:
            WITHIN: at least one of the first ``probe_n`` trajectories is execution-correct.
            BEYOND: all of the first ``probe_n`` trajectories are execution-wrong.

        Safety refinement used here: if a prompt initially classified BEYOND later
        produces a correct SQL while we are collecting an abstention learning signal,
        it is promoted to WITHIN. This prevents rewarding refusal after the current
        policy has directly demonstrated solvability.
        """
        acfg = self._adaptive_cfg()
        n_correct = int(acfg["n_correct"])
        n_wrong = int(acfg["n_wrong"])
        probe_n = int(acfg["probe_n"])
        b_max = int(acfg["b_max"])
        micro_n = int(acfg["micro_rollout_n"])
        thr = float(acfg["correct_threshold"])
        group_size = int(acfg["group_size"])
        abstain_text = str(acfg["abstain_text"])
        beyond_n_abstain = int(acfg["beyond_n_abstain"])
        within_abstain_cap = int(acfg["within_abstain_cap"])
        reward_values: RewardValues = acfg["reward_values"]

        if n_correct <= 0 or n_wrong <= 0:
            raise ValueError("n_correct and n_wrong must both be positive")
        if group_size != int(self.config.actor_rollout_ref.rollout.n):
            raise ValueError(
                "rollout.n must equal n_correct+n_wrong for GRPO grouping; "
                f"got rollout.n={self.config.actor_rollout_ref.rollout.n}, group_size={group_size}"
            )
        if probe_n <= 0 or probe_n > b_max:
            raise ValueError(f"probe_n must be in [1,b_max], got probe_n={probe_n}, b_max={b_max}")
        if micro_n <= 0:
            raise ValueError("micro_rollout_n must be positive")
        if not (1 <= beyond_n_abstain < group_size):
            raise ValueError("beyond_n_abstain must be in [1, group_size-1]")

        n_prompts = len(gen_batch)
        correct_rows = [[] for _ in range(n_prompts)]
        wrong_rows = [[] for _ in range(n_prompts)]
        abstain_rows = [[] for _ in range(n_prompts)]
        # Scores of the FIRST probe_n trajectories only.
        probe_exec_scores = [[] for _ in range(n_prompts)]
        num_gen = [0 for _ in range(n_prompts)]
        status = ["active" for _ in range(n_prompts)]
        boundary = [None for _ in range(n_prompts)]

        active = list(range(n_prompts))
        round_idx = 0
        while active:
            round_idx += 1
            round_in = gen_batch.select_idxs(active)
            round_in = round_in.repeat(repeat_times=micro_n, interleave=True)
            round_in.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            round_in.meta_info["global_steps"] = self.global_steps

            size_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
            padded_round_in, pad_size = pad_dataproto_to_divisor(round_in, size_divisor)
            with marked_timer("gen", timing_raw, "red"):
                round_out = self.async_rollout_manager.generate_sequences(padded_round_in)
                round_out = unpad_dataproto(round_out, pad_size)
                round_out.meta_info.pop("timing", None)
            round_out = self._attach_meta(round_out, round_in)

            with marked_timer("adaptive_score", timing_raw, "yellow"):
                round_exec_scores, round_draft_scores, round_texts, round_parsed = self._score_rollouts(round_out)

            new_active = []
            for a_pos, p_idx in enumerate(active):
                base = a_pos * micro_n
                extra_info = gen_batch.non_tensor_batch.get("extra_info", [{}] * n_prompts)[p_idx]
                db_id = extra_info.get("db_id") if isinstance(extra_info, dict) else None

                round_correct = 0
                round_wrong = 0
                round_abstain = 0
                for j in range(micro_n):
                    idx = base + j
                    row = round_out.slice(idx, idx + 1)
                    sc = float(round_exec_scores[idx])
                    text = round_texts[idx]
                    meta = round_parsed[idx]
                    abstained = bool(meta["abstained"])
                    record = {
                        "row": row,
                        "final_exec": sc,
                        "draft_exec": float(round_draft_scores[idx]),
                        "abstained": abstained,
                        "reflection_present": bool(meta["reflection_present"]),
                        "format_valid": bool(meta["format_valid"]),
                        "has_draft": bool(meta["has_draft"]),
                        "has_answer": bool(meta["has_answer"]),
                    }
                    if len(probe_exec_scores[p_idx]) < probe_n:
                        probe_exec_scores[p_idx].append(sc)

                    if sc >= thr:
                        correct_rows[p_idx].append(record)
                        round_correct += 1
                    elif abstained:
                        abstain_rows[p_idx].append(record)
                        round_abstain += 1
                    else:
                        wrong_rows[p_idx].append(record)
                        round_wrong += 1

                num_gen[p_idx] += micro_n

                # Classify boundary exactly once after fixed probe_n scores exist.
                if boundary[p_idx] is None and len(probe_exec_scores[p_idx]) >= probe_n:
                    first_k = probe_exec_scores[p_idx][:probe_n]
                    boundary[p_idx] = WITHIN if any(s >= thr for s in first_k) else BEYOND

                # If a later rollout demonstrates correctness, never continue to
                # reward abstention for this prompt.
                if boundary[p_idx] == BEYOND and len(correct_rows[p_idx]) > 0:
                    boundary[p_idx] = WITHIN

                c = len(correct_rows[p_idx])
                w = len(wrong_rows[p_idx])
                a = len(abstain_rows[p_idx])

                reward_summary = {
                    "event": "rein_sql_rollout",
                    "prompt_index": p_idx,
                    "db_id": db_id,
                    "round": round_idx,
                    "boundary": boundary[p_idx],
                    "round_correct": round_correct,
                    "round_wrong_sql": round_wrong,
                    "round_abstain": round_abstain,
                    "generated_total": num_gen[p_idx],
                    "cumulative_correct": c,
                    "cumulative_wrong_sql": w,
                    "cumulative_abstain": a,
                    "probe_n": probe_n,
                    "b_max": b_max,
                }
                print(f"[rein_sql_reward] {json.dumps(reward_summary, ensure_ascii=False)}", flush=True)

                if boundary[p_idx] is None:
                    if num_gen[p_idx] >= b_max:
                        status[p_idx] = "no_signal"
                    else:
                        new_active.append(p_idx)
                    continue

                if boundary[p_idx] == WITHIN:
                    negative_count = w + a
                    if c >= n_correct and negative_count >= n_wrong:
                        status[p_idx] = "done"
                    elif num_gen[p_idx] >= b_max:
                        if c >= n_correct and negative_count < n_wrong:
                            status[p_idx] = "too_easy"
                        else:
                            status[p_idx] = "within_unfilled"
                    else:
                        new_active.append(p_idx)
                else:
                    # BEYOND group must contain both abstentions (+reward) and
                    # wrong SQLs (0 reward), otherwise GRPO has no useful variance.
                    need_wrong = group_size - beyond_n_abstain
                    if a >= beyond_n_abstain and w >= need_wrong:
                        status[p_idx] = "done"
                    elif num_gen[p_idx] >= b_max:
                        status[p_idx] = "beyond_no_signal"
                    else:
                        new_active.append(p_idx)

            active = new_active

        kept_group_rows = []
        kept_scores = []
        n_done = 0
        n_within = 0
        n_beyond = 0
        n_selected_abstain = 0

        for p_idx in range(n_prompts):
            selected_rows = []
            selected_rewards = []
            bnd = boundary[p_idx]

            if status[p_idx] == "done" and bnd == WITHIN:
                n_done += 1
                n_within += 1
                chosen_correct = correct_rows[p_idx][:n_correct]
                chosen_negative = self._pick_within_negatives(
                    wrong_rows[p_idx], abstain_rows[p_idx], n_wrong, within_abstain_cap
                )
                if len(chosen_correct) != n_correct or len(chosen_negative) != n_wrong:
                    raise RuntimeError("WITHIN selection invariant violated")
                for rec in chosen_correct:
                    selected_rows.append(rec["row"])
                    selected_rewards.append(trajectory_reward(
                        final_exec=rec["final_exec"], draft_exec=rec["draft_exec"],
                        abstained=rec["abstained"], boundary=WITHIN,
                        reflection_present=rec["reflection_present"],
                        has_draft=rec["has_draft"], has_answer=rec["has_answer"],
                        correct_threshold=thr,
                        values=reward_values,
                    ))
                for rec, abstained in chosen_negative:
                    selected_rows.append(rec["row"])
                    selected_rewards.append(trajectory_reward(
                        final_exec=rec["final_exec"], draft_exec=rec["draft_exec"],
                        abstained=abstained, boundary=WITHIN,
                        reflection_present=rec["reflection_present"],
                        has_draft=rec["has_draft"], has_answer=rec["has_answer"],
                        correct_threshold=thr,
                        values=reward_values,
                    ))
                    if abstained:
                        n_selected_abstain += 1

            elif status[p_idx] == "done" and bnd == BEYOND:
                n_done += 1
                n_beyond += 1
                need_wrong = group_size - beyond_n_abstain
                chosen_a = abstain_rows[p_idx][:beyond_n_abstain]
                chosen_w = wrong_rows[p_idx][:need_wrong]
                if len(chosen_a) != beyond_n_abstain or len(chosen_w) != need_wrong:
                    raise RuntimeError("BEYOND selection invariant violated")
                chosen = chosen_a + chosen_w
                selected_rows = [rec["row"] for rec in chosen]
                selected_rewards = [
                    trajectory_reward(
                        final_exec=rec["final_exec"], draft_exec=rec["draft_exec"],
                        abstained=rec["abstained"], boundary=BEYOND,
                        reflection_present=rec["reflection_present"],
                        has_draft=rec["has_draft"], has_answer=rec["has_answer"],
                        correct_threshold=thr,
                        values=reward_values,
                    )
                    for rec in chosen
                ]
                n_selected_abstain += beyond_n_abstain

            if selected_rows:
                if len(selected_rows) != group_size or len(selected_rewards) != group_size:
                    raise RuntimeError("selected GRPO group has wrong size")
                kept_group_rows.extend(selected_rows)
                kept_scores.extend(selected_rewards)

            extra_info = gen_batch.non_tensor_batch.get("extra_info", [{}] * n_prompts)[p_idx]
            db_id = extra_info.get("db_id") if isinstance(extra_info, dict) else None
            selection_summary = {
                "event": "rein_sql_selection",
                "prompt_index": p_idx,
                "db_id": db_id,
                "status": status[p_idx],
                "boundary": bnd,
                "generated_total": num_gen[p_idx],
                "generated_correct": len(correct_rows[p_idx]),
                "generated_wrong_sql": len(wrong_rows[p_idx]),
                "generated_abstain": len(abstain_rows[p_idx]),
                "selected_total": len(selected_rows),
            }
            print(f"[rein_sql_reward] {json.dumps(selection_summary, ensure_ascii=False)}", flush=True)

        stage2_path = acfg.get("stage2_dump_path", None)
        if stage2_path:
            self._dump_stage2(gen_batch, status, boundary, stage2_path)

        stats = {
            "adaptive/num_prompts": n_prompts,
            "adaptive/num_kept": n_done,
            "adaptive/num_within": n_within,
            "adaptive/num_beyond": n_beyond,
            "adaptive/num_too_easy": sum(1 for s in status if s == "too_easy"),
            "adaptive/num_within_unfilled": sum(1 for s in status if s == "within_unfilled"),
            "adaptive/num_beyond_no_signal": sum(1 for s in status if s == "beyond_no_signal"),
            "adaptive/num_no_signal": sum(1 for s in status if s == "no_signal"),
            "adaptive/selected_abstain": n_selected_abstain,
            "adaptive/avg_rollouts_per_prompt": float(np.mean(num_gen)) if num_gen else 0.0,
            "adaptive/max_rollouts_per_prompt": int(np.max(num_gen)) if num_gen else 0,
        }

        if not kept_group_rows:
            return None, [], stats

        batch = DataProto.concat(kept_group_rows)
        return batch, kept_scores, stats

    def _dump_stage2(self, gen_batch: DataProto, status: list, boundary: list, path: str):
        """Dump hard/no-signal prompts for optional later analysis."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for i, st in enumerate(status):
                if st not in {"beyond_no_signal", "within_unfilled"}:
                    continue
                rec = {
                    "uid": str(gen_batch.non_tensor_batch["uid"][i]),
                    "global_steps": self.global_steps,
                    "status": st,
                    "boundary": boundary[i],
                    "ground_truth": gen_batch.non_tensor_batch["reward_model"][i].get("ground_truth", None),
                    "extra_info": gen_batch.non_tensor_batch.get("extra_info", [{}] * len(gen_batch))[i],
                }
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

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
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        self._load_checkpoint()
        resumed_global_steps = self.global_steps
        self.checkpoint_manager.update_weights()

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        acfg = self._adaptive_cfg()
        group_size = int(acfg["group_size"])
        max_num_gen_batches = int(acfg["max_num_gen_batches"])

        timing_raw = defaultdict(float)
        batch = None
        cached_scores: list = []
        num_prompt_in_batch = 0
        num_gen_batches = 0
        agg_stats = defaultdict(float)
        resume_data_epoch = acfg.get("resume_data_epoch")
        if resumed_global_steps > 0 and resume_data_epoch is not None:
            current_epoch = int(resume_data_epoch)
        else:
            current_epoch = self.global_steps // len(self.train_dataloader)
        print(
            f"REIN SQL source-data epoch: start={current_epoch}, "
            f"total={self.config.trainer.total_epochs}, resumed_global_steps={resumed_global_steps}"
        )

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )


                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ===== REIN-style capability probe + abstention-aware GRPO =====
                    if acfg.get("enable", True):
                        kept_batch, kept_scores, stats = self._adaptive_rollout_one_gen_batch(gen_batch, timing_raw)
                    else:
                        kept_batch, kept_scores, stats = self._fixed_rollout_one_gen_batch(gen_batch, timing_raw)
                    for k, v in stats.items():
                        agg_stats[k] += v

                    if kept_batch is not None:
                        num_prompt_in_batch += stats["adaptive/num_kept"]
                        if batch is None:
                            batch = kept_batch
                            cached_scores = list(kept_scores)
                        else:
                            batch = DataProto.concat([batch, kept_batch])
                            cached_scores = cached_scores + list(kept_scores)

                    prompt_bsz = self.config.data.train_batch_size
                    if num_prompt_in_batch < prompt_bsz:
                        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating more prompts...")
                            self.gen_steps += 1
                            continue
                        else:
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}. Could not collect enough "
                                "learnable prompts. Consider raising b_max, lowering train_batch_size, "
                                "or checking data difficulty. Set max_num_gen_batches=0 for endless trials."
                            )

                    # ---- trim to exactly train_batch_size prompts (group-aligned) ----
                    traj_bsz = prompt_bsz * group_size
                    batch = batch[:traj_bsz]
                    cached_scores = cached_scores[:traj_bsz]

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
                            self.checkpoint_manager.update_weights()

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
                metrics["train/num_gen_batches"] = num_gen_batches
                for k, v in agg_stats.items():
                    metrics[k] = v
                timing_raw = defaultdict(float)

                # reset per-step accumulators
                batch = None
                cached_scores = []
                num_prompt_in_batch = 0
                num_gen_batches = 0
                agg_stats = defaultdict(float)

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
