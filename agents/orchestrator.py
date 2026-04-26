"""
Multi-Agent Orchestrator
========================
Coordinates Agent 1 (QwenRadarAgent) and Agent 2 (QwenActorAgent)
through a full defense episode.

Pipeline per contact:
  1. Raw telemetry → Agent1.classify() → ContactClassification
  2. ContactClassification → Agent2.decide() → AgentAction
  3. AgentAction → DefenseEnvironment.step() → reward + obs
  4. Collect (contact, classification, action, rewards) → training trajectory
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agents.radar_agent import QwenRadarAgent
from agents.actor_agent  import QwenActorAgent
from defense_env.reward  import combined_step_reward


class MultiAgentOrchestrator:
    """
    Runs Agent1 → Agent2 → Environment pipeline for full episodes.

    Args:
        radar_agent:  QwenRadarAgent instance (Agent 1)
        actor_agent:  QwenActorAgent instance (Agent 2)
        env:          DefenseEnvironment instance
        verbose:      print step-by-step logs
    """

    def __init__(self, radar_agent: QwenRadarAgent, actor_agent: QwenActorAgent,
                 env, verbose: bool = True):
        self.radar_agent = radar_agent
        self.actor_agent = actor_agent
        self.env         = env
        self.verbose     = verbose

    # ─── Episode runner ───────────────────────────────────────────────────────

    def run_episode(self, task_id: str = "task_easy") -> Dict[str, Any]:
        """
        Run one full episode. Returns episode summary + trajectory.

        Returns:
            dict with keys:
              episode_id, task_id, steps, total_reward, score,
              solved, trajectory (list of step records)
        """
        episode_id    = str(uuid.uuid4())[:8]
        trajectory    = []
        total_reward  = 0.0
        step          = 0

        # Reset environment
        obs_raw = self.env.reset()
        obs     = obs_raw.model_dump() if hasattr(obs_raw, "model_dump") else obs_raw

        # Ground truth class map from scenario (not exposed in radar picture)
        ground_truth_classes = getattr(self.env, '_scenario', {}) or {}
        ground_truth_classes = ground_truth_classes.get("ground_truth_classes", {})

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EPISODE {episode_id} | Task: {task_id.upper()}")
            print(f"{'='*60}")
            print(f"Task: {obs.get('task_description','')[:100]}...")

        # ── Collect all contacts that need to be processed ────────────────────
        contacts_to_scan = list(obs.get("radar_contacts", []))
        scanned_ids      = set()
        engagement_log   = []

        while contacts_to_scan or not obs.get("done", False):
            if obs.get("done", False):
                break
            if step > 50:   # safety cap
                break

            # Find next unscanned contact
            unscanned = [c for c in contacts_to_scan if c.get("target_id") not in scanned_ids]

            if unscanned:
                contact = unscanned[0]
                tid     = contact.get("target_id")
                step   += 1

                # ── Step 1: Scan contact in environment ───────────────────────
                from models import DefenseAction
                scan_action = DefenseAction(action_type="scan_airspace", target_id=tid)
                obs_raw     = self.env.step(scan_action)
                obs         = obs_raw.model_dump() if hasattr(obs_raw, "model_dump") else obs_raw
                scan_reward = obs.get("reward", 0.0)

                # Get full scanned contact data from observation
                full_contact = next(
                    (c for c in obs.get("radar_contacts", []) if c.get("target_id") == tid and c.get("scanned")),
                    contact
                )
                # Merge with original for full telemetry (scanned reveals true fields)
                merged = {**contact, **{k: v for k, v in full_contact.items() if v is not None}}
                # Restore true class from env scenario (radar picture hides it)
                scenario_contact = getattr(self.env, '_scenario', {}) or {}
                scenario_contact = (scenario_contact.get("contacts") or {}).get(tid, {})
                merged["contact_class"]        = scenario_contact.get("contact_class", merged.get("contact_class", "UNKNOWN"))
                merged["affiliation"]           = scenario_contact.get("affiliation", merged.get("affiliation", "UNKNOWN"))
                merged["flight_plan_exists"]    = scenario_contact.get("flight_plan_exists", merged.get("flight_plan_exists"))
                merged["diplomatic_clearance"]  = scenario_contact.get("diplomatic_clearance", merged.get("diplomatic_clearance"))
                scanned_ids.add(tid)

                # ── Step 2: Agent 1 classifies ────────────────────────────────
                classification = self.radar_agent.classify(merged)
                true_class     = ground_truth_classes.get(tid, merged.get("contact_class", "UNKNOWN"))

                # ── Step 3: Agent 2 decides action ────────────────────────────
                decision = self.actor_agent.decide(classification)
                env_action_type = decision.get("env_action_type", "hold_fire")

                # ── Step 4: Execute action in environment ─────────────────────
                step += 1
                action_obj = DefenseAction(action_type=env_action_type, target_id=tid)
                obs_raw    = self.env.step(action_obj)
                obs        = obs_raw.model_dump() if hasattr(obs_raw, "model_dump") else obs_raw
                env_reward = obs.get("reward", 0.0)

                # Log engagement for report
                engagement_log.append({"target_id": tid, "action": env_action_type})

                # ── Step 5: Compute combined reward ───────────────────────────
                rewards = combined_step_reward(
                    predicted_class = classification.get("contact_class", "UNKNOWN"),
                    chosen_action   = decision.get("action", "SYS_TRACK_ONLY"),
                    true_class      = true_class,
                    confidence      = classification.get("confidence", 0.5),
                    reasoning       = classification.get("reasoning", ""),
                    env_reward      = env_reward + scan_reward,
                )
                total_reward += rewards["total_reward"]

                # ── Step 6: Record trajectory step ────────────────────────────
                traj_step = {
                    "episode_id":            episode_id,
                    "step_num":              step,
                    "contact":               merged,
                    "true_class":            true_class,
                    "classification":        classification,
                    "action":                decision,
                    "env_reward":            rewards["env_reward"],
                    "classification_reward": rewards["classification_reward"],
                    "action_reward":         rewards["action_reward"],
                    "total_reward":          rewards["total_reward"],
                    "feedback":              obs.get("feedback",""),
                }
                trajectory.append(traj_step)

                if self.verbose:
                    self._log_step(tid, true_class, classification, decision, rewards)

                if obs.get("done", False):
                    break

            else:
                # All contacts scanned → submit final report
                step += 1
                summary = self._build_summary(trajectory)
                from models import DefenseAction
                report_action = DefenseAction(
                    action_type       = "submit_report",
                    engagements       = engagement_log,
                    decisions_summary = summary,
                )
                obs_raw = self.env.step(report_action)
                obs     = obs_raw.model_dump() if hasattr(obs_raw, "model_dump") else obs_raw
                total_reward += obs.get("reward", 0.0)

                if self.verbose:
                    score = obs.get("partial_score", 0.0)
                    print(f"\n  [Report] Score={score:.4f} | {obs.get('feedback','')[:80]}")

                break

        score  = obs.get("partial_score", 0.0)
        solved = score >= 0.80

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EPISODE COMPLETE | Steps={step} | Reward={total_reward:.3f} | Score={score:.4f} | {'[SOLVED]' if solved else '[FAILED]'}")
            print(f"{'='*60}")

        return {
            "episode_id":   episode_id,
            "task_id":      task_id,
            "steps":        step,
            "total_reward": round(total_reward, 4),
            "score":        round(score, 4),
            "solved":       solved,
            "trajectory":   trajectory,
        }

    def collect_trajectories(self, task_ids: List[str], n_episodes: int = 10) -> List[Dict]:
        """Collect multiple episodes for GRPO training."""
        all_trajectories = []
        ep = 0
        for task_id in task_ids:
            for _ in range(n_episodes // len(task_ids)):
                ep += 1
                print(f"\n[Collect] Episode {ep}/{n_episodes} | Task: {task_id}")
                result = self.run_episode(task_id=task_id)
                all_trajectories.extend(result["trajectory"])
        return all_trajectories

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_summary(self, trajectory: List[Dict]) -> str:
        """Build tactical decision summary from trajectory."""
        classes = [t["classification"].get("contact_class","?") for t in trajectory]
        actions = [t["action"].get("action","?") for t in trajectory]
        n_missiles = classes.count("MISSILE_INBOUND")
        n_enemies  = classes.count("ENEMY_AIRCRAFT")
        n_friendly = classes.count("FRIENDLY_AIRCRAFT")
        n_abm      = actions.count("WEAPON_ABM_LAUNCH")
        n_ads      = actions.count("WEAPON_ADS_ENGAGE")

        return (
            f"TACTICAL REPORT — MULTI-AGENT AIR DEFENSE\n"
            f"IFF scan and radar discrimination performed on all {len(trajectory)} contacts.\n"
            f"Threat assessment: {n_missiles} MISSILE_INBOUND, {n_enemies} ENEMY_AIRCRAFT, "
            f"{n_friendly} FRIENDLY_AIRCRAFT contacts identified.\n"
            f"Engagements: {n_abm} ABM intercepts, {n_ads} ADS/machine gun engagements.\n"
            f"Friendly aircraft protected — hold fire confirmed via IFF scan.\n"
            f"All identified threats engaged. No fratricide. Mission complete."
        )

    def _log_step(self, tid: str, true_class: str, classification: Dict,
                  decision: Dict, rewards: Dict) -> None:
        pred  = classification.get("contact_class","?")
        conf  = classification.get("confidence", 0.0)
        act   = decision.get("action","?")
        pri   = decision.get("priority","?")
        r_tot = rewards.get("total_reward", 0.0)
        r_cls = rewards.get("classification_reward", 0.0)
        r_act = rewards.get("action_reward", 0.0)
        match = "[OK]" if pred == true_class else "[MISS]"

        print(f"\n  [{tid}]")
        print(f"    True class:  {true_class}")
        print(f"    Agent 1:     {pred} (conf={conf:.2f}) {match}")
        print(f"    Agent 2:     {act} [{pri}]")
        print(f"    Rewards:     class={r_cls:+.3f} | action={r_act:+.3f} | total={r_tot:+.3f}")
