"""
train.py — GRPO Fine-Tuning Entry Point
========================================
Runs the full multi-agent training pipeline:
  1. Collect episodes with current agents (data collection)
  2. Build GRPO dataset from trajectories
  3. Fine-tune both agents with GRPO
  4. Evaluate on held-out tasks
  5. Save adapter checkpoints

Usage:
  # Quick demo (rule-based agents, no GPU needed)
  python train.py --backend rule --episodes 20 --epochs 1 --dry-run

  # Local Qwen model training (GPU recommended)
  python train.py --backend local --model Qwen/Qwen2.5-1.5B-Instruct --episodes 100 --epochs 3

  # API-based (Qwen via Together.ai / Alibaba Cloud)
  python train.py --backend api --api-base https://api.together.xyz/v1 --model Qwen/Qwen2.5-7B-Instruct

  # Load from checkpoint and continue
  python train.py --backend local --resume checkpoints/epoch_2
"""

import argparse
import json
import os
import sys
import warnings

# Suppress repetitive transformers deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*generation_config.*deprecated.*")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="GRPO fine-tuning for Qwen Defense Multi-Agent System")
    p.add_argument("--backend",    default="rule",
                   choices=["rule","local","api"],
                   help="Agent backend: 'rule' (no model), 'local' (HuggingFace), 'api' (OpenAI-compat)")
    p.add_argument("--model",      default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID or API model name")
    p.add_argument("--api-base",   default="https://api.openai.com/v1",
                   help="API base URL (for --backend api)")
    p.add_argument("--api-key",    default=os.environ.get("HF_TOKEN", "no-key"),
                   help="API key (for --backend api)")
    p.add_argument("--device",     default="auto",
                   help="Device: 'auto', 'cuda', 'cpu'")
    p.add_argument("--episodes",   type=int, default=50,
                   help="Total episodes for data collection")
    p.add_argument("--epochs",     type=int, default=3,
                   help="GRPO training epochs")
    p.add_argument("--group-size", type=int, default=4,
                   help="GRPO group size G (completions per prompt)")
    p.add_argument("--lr",         type=float, default=2e-5,
                   help="Learning rate")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size")
    p.add_argument("--lora-rank",  type=int, default=16,
                   help="LoRA rank r")
    p.add_argument("--tasks",      nargs="+",
                   default=["task_easy","task_medium","task_hard"],
                   help="Task IDs to train on")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--resume",     default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--dry-run",    action="store_true",
                   help="Collect data and print stats, skip GRPO update (CPU-safe)")
    p.add_argument("--verbose",    action="store_true", default=False,
                   help="Show step-by-step episode logs (default: off for cleaner output)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("DEFENSE AI — GRPO MULTI-AGENT TRAINING")
    print("="*60)
    print(f"  Backend:    {args.backend}")
    print(f"  Model:      {args.model}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Dry run:    {args.dry_run}")
    print(f"  Tasks:      {args.tasks}")
    print("="*60)

    # ── 1. Build agents ─────────────────────────────────────────────────────
    from agents.radar_agent import QwenRadarAgent
    from agents.actor_agent  import QwenActorAgent
    from agents.orchestrator import MultiAgentOrchestrator
    from defense_env.environment import DefenseEnvironment

    common_kwargs = dict(
        backend    = args.backend,
        model_name = args.model,
        api_base   = args.api_base,
        api_key    = args.api_key,
        device     = args.device,
    )

    print("\n[Setup] Initialising agents...")
    radar_agent = QwenRadarAgent(**common_kwargs)
    actor_agent = QwenActorAgent(**common_kwargs)

    # ── 2. Collect training trajectories ────────────────────────────────────
    print(f"\n[Collect] Running {args.episodes} episodes across tasks: {args.tasks}")
    all_trajectories = []
    episode_results  = []   # track per-episode score/reward/solved

    eps_per_task = max(1, args.episodes // len(args.tasks))
    for task_id in args.tasks:
        env = DefenseEnvironment(task_id=task_id)
        orch = MultiAgentOrchestrator(radar_agent, actor_agent, env, verbose=args.verbose)
        for ep in range(eps_per_task):
            result = orch.run_episode(task_id=task_id)
            all_trajectories.extend(result["trajectory"])
            episode_results.append(result)
            status = "SOLVED" if result["solved"] else "failed"
            print(f"  [Ep {ep+1:02d}/{eps_per_task} | {task_id}] "
                  f"steps={result['steps']:2d} | reward={result['total_reward']:+.3f} | "
                  f"score={result['score']:.4f} | {status}")

    print(f"\n[Collect] Total trajectory steps collected: {len(all_trajectories)}")

    # ── 3. Build GRPO datasets ───────────────────────────────────────────────
    from training.dataset import build_grpo_dataset, save_dataset

    os.makedirs("data", exist_ok=True)

    radar_train, radar_eval = build_grpo_dataset(all_trajectories, agent="radar", seed=args.seed)
    actor_train, actor_eval = build_grpo_dataset(all_trajectories, agent="actor", seed=args.seed)

    save_dataset(radar_train, "data/radar_train.jsonl")
    save_dataset(radar_eval,  "data/radar_eval.jsonl")
    save_dataset(actor_train, "data/actor_train.jsonl")
    save_dataset(actor_eval,  "data/actor_eval.jsonl")

    print(f"\n[Dataset] Radar: {len(radar_train)} train / {len(radar_eval)} eval")
    print(f"[Dataset] Actor: {len(actor_train)} train / {len(actor_eval)} eval")

    # Classification accuracy baseline
    correct = sum(1 for s in all_trajectories
                  if s.get("classification",{}).get("contact_class") == s.get("true_class"))
    total   = len(all_trajectories)
    print(f"\n[Baseline] Agent 1 classification accuracy: {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    if args.dry_run or args.backend == "rule":
        print("\n[Dry run / rule backend] Skipping GRPO fine-tuning.")
        print("  To run full training:  python train.py --backend local --epochs 3")
        _print_summary(all_trajectories, episode_results, args.tasks)
        return

    # ── 4. GRPO training ─────────────────────────────────────────────────────
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

    print("\n[Train] Starting GRPO fine-tuning for Agent 1 (Radar)...")
    cfg = GRPOConfig(
        model_name      = args.model,
        lora_rank       = args.lora_rank,
        group_size      = args.group_size,
        lr              = args.lr,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        checkpoint_dir  = os.path.join(args.checkpoint_dir, "radar"),
        device          = args.device,
        seed            = args.seed,
    )
    trainer_radar = GRPOTrainer(cfg)
    if args.resume:
        trainer_radar.load_checkpoint(os.path.join(args.resume, "radar"))
    log_radar = trainer_radar.train(radar_train, radar_eval)

    print("\n[Train] Starting GRPO fine-tuning for Agent 2 (Actor)...")
    cfg2 = GRPOConfig(
        model_name      = args.model,
        lora_rank       = args.lora_rank,
        group_size      = args.group_size,
        lr              = args.lr,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        checkpoint_dir  = os.path.join(args.checkpoint_dir, "actor"),
        device          = args.device,
        seed            = args.seed,
    )
    trainer_actor = GRPOTrainer(cfg2)
    if args.resume:
        trainer_actor.load_checkpoint(os.path.join(args.resume, "actor"))
    log_actor = trainer_actor.train(actor_train, actor_eval)

    # ── 5. Save training logs ────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "training_log.json"), "w") as f:
        json.dump({"radar": log_radar, "actor": log_actor}, f, indent=2)
    print(f"\n[Done] Training logs saved → {args.checkpoint_dir}/training_log.json")

    _print_summary(all_trajectories, episode_results, args.tasks)


def _print_summary(trajectories: list, episode_results: list, tasks: list) -> None:
    """
    Full two-agent evaluation report:
      - Episode performance table (per task)
      - Agent 1: overall accuracy, per-class precision & recall
      - Agent 2: correctness rate, per-action breakdown
      - Reward components breakdown
      - Confusion matrix (predicted vs true)
    """
    from collections import Counter, defaultdict

    SEP  = "=" * 70
    SEP2 = "-" * 70

    # ── Raw data extraction ────────────────────────────────────────────────
    true_classes  = [s.get("true_class", "?")                         for s in trajectories]
    pred_classes  = [s.get("classification", {}).get("contact_class", "?") for s in trajectories]
    confidences   = [s.get("classification", {}).get("confidence", 0.0)    for s in trajectories]
    chosen_acts   = [s.get("action", {}).get("action", "?")               for s in trajectories]
    priorities    = [s.get("action", {}).get("priority", "?")              for s in trajectories]
    r_class       = [s.get("classification_reward", 0.0)                   for s in trajectories]
    r_action      = [s.get("action_reward", 0.0)                           for s in trajectories]
    r_env         = [s.get("env_reward", 0.0)                              for s in trajectories]
    r_total       = [s.get("total_reward", 0.0)                            for s in trajectories]
    N = max(len(trajectories), 1)

    # Agent 2 correctness: did it pick the optimal action given the TRUE class?
    from defense_env.reward import CORRECT_ACTIONS, ACCEPTABLE_ACTIONS
    act2_correct    = sum(1 for tc, ca in zip(true_classes, chosen_acts) if ca in CORRECT_ACTIONS.get(tc, set()))
    act2_acceptable = sum(1 for tc, ca in zip(true_classes, chosen_acts) if ca in ACCEPTABLE_ACTIONS.get(tc, set()))
    act2_wrong      = N - act2_correct - act2_acceptable

    # Agent 1 per-class stats
    all_classes = sorted(set(true_classes) | set(pred_classes) - {"?"})

    # confusion counts: true → predicted
    confusion = defaultdict(Counter)
    for t, p in zip(true_classes, pred_classes):
        confusion[t][p] += 1

    # Per-class TP/FP/FN
    tp = {c: confusion[c][c] for c in all_classes}
    fp = {c: sum(confusion[t][c] for t in all_classes if t != c) for c in all_classes}
    fn = {c: sum(confusion[c][p] for p in all_classes if p != c) for c in all_classes}
    precision = {c: tp[c] / max(tp[c] + fp[c], 1) for c in all_classes}
    recall    = {c: tp[c] / max(tp[c] + fn[c], 1) for c in all_classes}
    f1        = {c: 2 * precision[c] * recall[c] / max(precision[c] + recall[c], 1e-9) for c in all_classes}

    total_correct = sum(tp.values())
    avg_conf      = sum(confidences) / N

    # Per-task episode stats
    task_stats = defaultdict(lambda: {"episodes":0,"solved":0,"scores":[],"rewards":[]})
    for ep in episode_results:
        tid = ep.get("task_id", "?")
        task_stats[tid]["episodes"] += 1
        task_stats[tid]["solved"]   += int(ep.get("solved", False))
        task_stats[tid]["scores"].append(ep.get("score", 0.0))
        task_stats[tid]["rewards"].append(ep.get("total_reward", 0.0))

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n")
    print(SEP)
    print("  MULTI-AGENT DEFENSE SYSTEM — EVALUATION REPORT")
    print(SEP)

    # ── 1. Episode Performance Table ───────────────────────────────────────
    print("\n  [1] EPISODE PERFORMANCE BY TASK")
    print(SEP2)
    print(f"  {'Task':<18} {'Episodes':>8} {'Solved':>7} {'Solve%':>7} {'Avg Score':>10} {'Avg Reward':>11}")
    print(SEP2)
    for tid in tasks:
        st = task_stats.get(tid)
        if not st or st["episodes"] == 0:
            continue
        ep_n     = st["episodes"]
        solved_n = st["solved"]
        solve_pct = solved_n / ep_n * 100
        avg_sc   = sum(st["scores"])  / ep_n
        avg_rew  = sum(st["rewards"]) / ep_n
        bar      = "#" * int(solve_pct / 10)  # 0-10 char bar
        print(f"  {tid:<18} {ep_n:>8} {solved_n:>7} {solve_pct:>6.1f}% {avg_sc:>10.4f} {avg_rew:>11.3f}  [{bar:<10}]")

    all_scores  = [ep.get("score", 0.0)         for ep in episode_results]
    all_rewards = [ep.get("total_reward", 0.0)  for ep in episode_results]
    all_solved  = [ep.get("solved", False)       for ep in episode_results]
    print(SEP2)
    print(f"  {'OVERALL':<18} {len(episode_results):>8} {sum(all_solved):>7} "
          f"{sum(all_solved)/max(len(all_solved),1)*100:>6.1f}% "
          f"{sum(all_scores)/max(len(all_scores),1):>10.4f} "
          f"{sum(all_rewards)/max(len(all_rewards),1):>11.3f}")

    # ── 2. Agent 1 — Radar Classification ──────────────────────────────────
    print("\n  [2] AGENT 1 — RADAR CLASSIFICATION PERFORMANCE")
    print(SEP2)
    print(f"  Overall accuracy : {total_correct}/{N} = {total_correct/N*100:.1f}%")
    print(f"  Avg confidence   : {avg_conf:.3f}")
    print(SEP2)
    print(f"  {'Class':<26} {'Count':>6} {'TP':>5} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(SEP2)
    true_cnt = Counter(true_classes)
    for c in sorted(all_classes):
        if c == "?":
            continue
        cnt  = true_cnt.get(c, 0)
        prec = precision[c]
        rec  = recall[c]
        f1s  = f1[c]
        bar  = "#" * int(rec * 10)
        print(f"  {c:<26} {cnt:>6} {tp[c]:>5} {prec:>9.1%} {rec:>7.1%} {f1s:>7.1%}  [{bar:<10}]")

    # ── 3. Agent 1 — Confusion Matrix ──────────────────────────────────────
    print("\n  [3] AGENT 1 — CONFUSION MATRIX (rows=True, cols=Predicted)")
    print(SEP2)
    active = sorted(set(true_classes) - {"?"})
    pred_active = sorted(set(pred_classes) - {"?"})
    # Column headers (abbreviated)
    abbrev = {
        "MISSILE_INBOUND":      "MSSL",
        "ENEMY_AIRCRAFT":       "ENMY",
        "FRIENDLY_AIRCRAFT":    "FRND",
        "DOMESTIC_FLIGHT":      "DOMF",
        "FOREIGN_PERMITTED":    "FRP ",
        "FOREIGN_UNPERMITTED":  "FRNP",
        "OWN_ASSET":            "OWN ",
        "UNKNOWN":              "UNK ",
    }
    col_order = [c for c in [
        "MISSILE_INBOUND","ENEMY_AIRCRAFT","FRIENDLY_AIRCRAFT",
        "DOMESTIC_FLIGHT","FOREIGN_PERMITTED","FOREIGN_UNPERMITTED",
        "OWN_ASSET","UNKNOWN"
    ] if c in pred_active]
    _col_label = "True / Pred"
    header = f"  {_col_label:<26}" + "".join(f" {abbrev.get(c,c[:4]):>5}" for c in col_order)
    print(header)
    print("  " + "-" * (26 + 6 * len(col_order)))
    for tr in active:
        row = f"  {tr:<26}"
        for pr in col_order:
            cnt = confusion[tr][pr]
            marker = f"[{cnt:>3}]" if tr == pr and cnt > 0 else f" {cnt:>3} " if cnt > 0 else "  .  "
            row += f" {marker}"
        print(row)

    # ── 4. Agent 2 — Action Performance ────────────────────────────────────
    print("\n  [4] AGENT 2 — TACTICAL ACTION PERFORMANCE")
    print(SEP2)
    print(f"  Optimal action chosen   : {act2_correct}/{N} = {act2_correct/N*100:.1f}%")
    print(f"  Acceptable action chosen: {act2_acceptable}/{N} = {act2_acceptable/N*100:.1f}%")
    print(f"  Suboptimal action chosen: {act2_wrong}/{N} = {act2_wrong/N*100:.1f}%")
    print(f"  Optimal + Acceptable    : {act2_correct+act2_acceptable}/{N} = {(act2_correct+act2_acceptable)/N*100:.1f}%")
    print(SEP2)
    print(f"  {'Action':<26} {'Count':>6} {'%':>7}  Priority Breakdown")
    print(SEP2)
    act_counter  = Counter(chosen_acts)
    act_priority = defaultdict(Counter)
    for a, p in zip(chosen_acts, priorities):
        act_priority[a][p] += 1
    for act, cnt in act_counter.most_common():
        pct     = cnt / N * 100
        pri_str = "  ".join(f"{p}:{c}" for p, c in act_priority[act].most_common())
        bar     = "#" * int(pct / 5)
        print(f"  {act:<26} {cnt:>6} {pct:>6.1f}%  [{bar:<16}]  {pri_str}")

    # ── 5. Per-class Agent 2 breakdown ─────────────────────────────────────
    print("\n  [5] AGENT 2 — ACTION vs TRUE CLASS BREAKDOWN")
    print(SEP2)
    class_action = defaultdict(Counter)
    for tc, ca in zip(true_classes, chosen_acts):
        class_action[tc][ca] += 1
    for tc in sorted(set(true_classes) - {"?"}):
        opt  = CORRECT_ACTIONS.get(tc, set())
        cnts = class_action[tc]
        total_tc = sum(cnts.values())
        correct_tc = sum(cnts[a] for a in opt)
        print(f"  {tc:<26} (n={total_tc:3d}) | correct={correct_tc:3d} ({correct_tc/max(total_tc,1)*100:.0f}%)")
        for act, cnt in cnts.most_common():
            tag = "[OPTIMAL] " if act in opt else "[OK]      " if act in ACCEPTABLE_ACTIONS.get(tc, set()) else "[MISS]    "
            print(f"    {tag} {act:<26}: {cnt}")

    # ── 6. Reward component breakdown ──────────────────────────────────────
    print("\n  [6] REWARD COMPONENT BREAKDOWN")
    print(SEP2)
    def _stats(vals):
        mn  = min(vals) if vals else 0.0
        mx  = max(vals) if vals else 0.0
        avg = sum(vals) / max(len(vals), 1)
        return avg, mn, mx
    avg_rc, min_rc, max_rc = _stats(r_class)
    avg_ra, min_ra, max_ra = _stats(r_action)
    avg_re, min_re, max_re = _stats(r_env)
    avg_rt, min_rt, max_rt = _stats(r_total)
    print(f"  {'Component':<25} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print(SEP2)
    print(f"  {'Classification (Agent 1)':<25} {avg_rc:>+8.4f} {min_rc:>+8.4f} {max_rc:>+8.4f}")
    print(f"  {'Action (Agent 2)':<25} {avg_ra:>+8.4f} {min_ra:>+8.4f} {max_ra:>+8.4f}")
    print(f"  {'Environment':<25} {avg_re:>+8.4f} {min_re:>+8.4f} {max_re:>+8.4f}")
    print(f"  {'Total (weighted)':<25} {avg_rt:>+8.4f} {min_rt:>+8.4f} {max_rt:>+8.4f}")
    print(SEP2)
    print(f"  Cumulative total reward  : {sum(r_total):+.3f}")
    print(f"  Total trajectory steps   : {N}")
    print(f"  Total episodes           : {len(episode_results)}")

    print("\n" + SEP)
    print("  END OF REPORT")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
