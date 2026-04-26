"""
Core environment logic for Defense AI.

Implements the three OpenEnv methods:
  reset()  → DefenseObservation   (start new mission)
  step()   → DefenseObservation   (take one action)
  state    → DefenseState         (episode metadata)
"""

import random
import uuid
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:
    class Environment:
        def __init__(self, *args, **kwargs):
            pass

from models import DefenseAction, DefenseObservation, DefenseState
from defense_env.data_generator import generate_scenario
from defense_env.reward import calculate_step_reward, calculate_report_reward
from defense_env.tasks import TASKS, GRADERS, TASK_ORDER

SOLVE_THRESHOLD = 0.80


class DefenseEnvironment(Environment):
    """
    Defense AI — Air Defense Decision Environment.

    The agent acts as an air defense commander:
      - Scans radar contacts for IFF (friend or foe)
      - Engages enemy jets with machine gun fire
      - Intercepts inbound missiles with anti-ballistic missiles (ABM)
      - Holds fire on friendly aircraft
      - Submits a tactical engagement report
    """

    def __init__(self, task_id=None):
        try:
            super().__init__()
        except Exception:
            pass

        self._task_id_override = task_id
        self._episode_index    = 0

        self._scenario:         Optional[Dict] = None
        self._task_id:          str = ""
        self._episode_id:       str = ""
        self._step_count:       int = 0
        self._scanned:          Dict[str, bool] = {}
        self._engaged:          Dict[str, str] = {}   # target_id → action used
        self._held:             List[str] = []
        self._solved:           bool = False
        self._cumulative_reward: float = 0.0
        self._best_score:        float = 0.0001
        self._engagement_log:    List[str] = []
        self._seed:             int = 0

    # ─── reset() ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> DefenseObservation:
        """Start a fresh mission with a new scenario."""

        if self._task_id_override:
            self._task_id = self._task_id_override
        else:
            self._task_id = TASK_ORDER[self._episode_index % len(TASK_ORDER)]
            self._episode_index += 1

        task = TASKS[self._task_id]

        self._seed     = random.randint(0, 999_999)
        self._scenario = generate_scenario(self._task_id, self._seed)

        self._episode_id        = str(uuid.uuid4())
        self._step_count        = 0
        self._scanned           = {}
        self._engaged           = {}
        self._held              = []
        self._solved            = False
        self._cumulative_reward = 0.0
        self._best_score        = 0.0001
        self._engagement_log    = []

        return DefenseObservation(
            mission_id       = self._scenario["mission_id"],
            task_id          = self._task_id,
            difficulty       = task.difficulty,
            task_description = task.description,
            radar_contacts   = self._scenario["initial_radar"],
            threats_in_scope = self._scenario["threats_in_scope"],
            engaged_targets  = [],
            held_targets     = [],
            action_result    = None,
            action_error     = None,
            partial_score    = 0.0001,
            feedback         = (
                "Mission started. Radar contacts detected. "
                "Scan contacts to identify friend or foe before engaging."
            ),
            done             = False,
            reward           = 0.0,
            steps_taken      = 0,
            max_steps        = task.max_steps,
        )

    # ─── step() ──────────────────────────────────────────────────────────────

    def step(self, action: DefenseAction, timeout_s=None, **kwargs) -> DefenseObservation:
        """Execute one action and return updated observation."""

        if self._scenario is None:
            return self._error_obs("Call reset() before step().")

        if self._solved:
            task = TASKS[self._task_id]
            return DefenseObservation(
                mission_id       = self._scenario["mission_id"],
                task_id          = self._task_id,
                difficulty       = task.difficulty,
                task_description = task.description,
                radar_contacts   = self._current_radar(),
                threats_in_scope = self._scenario["threats_in_scope"],
                action_result    = None,
                action_error     = "Mission already complete. Call reset() to start a new mission.",
                engaged_targets  = list(self._engaged.keys()),
                held_targets     = list(self._held),
                partial_score    = self._best_score,
                feedback         = "Mission already complete.",
                done             = True,
                reward           = 0.0,
                steps_taken      = self._step_count,
                max_steps        = TASKS[self._task_id].max_steps,
            )

        self._step_count += 1
        task      = TASKS[self._task_id]
        max_steps = task.max_steps

        action_type   = (action.action_type or "").strip().lower()
        action_result = None
        action_error  = None
        reward        = 0.0
        partial_score = self._best_score
        feedback      = ""

        if action_type == "scan_airspace":
            action_result, action_error, reward, feedback = self._handle_scan(action)

        elif action_type in ("engage_machine_gun", "weapon_ads_engage"):
            action_result, action_error, reward, feedback = self._handle_engage(
                action, weapon="engage_machine_gun"
            )

        elif action_type in ("launch_abm", "weapon_abm_launch"):
            action_result, action_error, reward, feedback = self._handle_engage(
                action, weapon="launch_abm"
            )

        elif action_type in ("hold_fire", "comm_hello", "comm_warn",
                             "nav_guide", "sys_track_only"):
            # All non-weapon actions map to hold_fire logic in the environment
            action_result, action_error, reward, feedback = self._handle_hold(action)
            # Provide richer feedback for multi-agent actions
            if action_type == "comm_hello" and action_error is None:
                feedback = f"COMM_HELLO: Friendly handshake transmitted to {action.target_id}."
            elif action_type == "comm_warn" and action_error is None:
                feedback = f"COMM_WARN: Formal warning issued to {action.target_id} on emergency freq."
            elif action_type == "nav_guide" and action_error is None:
                feedback = f"NAV_GUIDE: Vectoring and landing instructions sent to {action.target_id}."
            elif action_type == "sys_track_only" and action_error is None:
                feedback = f"SYS_TRACK_ONLY: Radar lock maintained on {action.target_id}. Logged."

        elif action_type == "submit_report":
            action_result, action_error, reward, partial_score, feedback = (
                self._handle_report(action)
            )

        else:
            action_error = (
                f"Unknown action_type '{action.action_type}'. "
                "Valid: scan_airspace | engage_machine_gun | weapon_ads_engage | "
                "launch_abm | weapon_abm_launch | hold_fire | comm_hello | "
                "comm_warn | nav_guide | sys_track_only | submit_report"
            )
            reward = -0.05

        self._cumulative_reward += reward
        if partial_score > self._best_score:
            self._best_score = partial_score

        self._engagement_log.append(
            f"Step {self._step_count}: {action_type} "
            f"[{action.target_id}] -> reward={reward:+.3f}"
        )

        done = self._solved or (self._step_count >= max_steps) or (action_type == "submit_report")
        if done and not self._solved:
            feedback += (
                f" Mission ended after {self._step_count} steps. "
                f"Best score: {self._best_score:.2f}."
            )

        return DefenseObservation(
            mission_id       = self._scenario["mission_id"],
            task_id          = self._task_id,
            difficulty       = task.difficulty,
            task_description = task.description,
            radar_contacts   = self._current_radar(),
            threats_in_scope = self._scenario["threats_in_scope"],
            action_result    = action_result,
            action_error     = action_error,
            engaged_targets  = list(self._engaged.keys()),
            held_targets     = list(self._held),
            partial_score    = partial_score,
            feedback         = feedback,
            done             = done,
            reward           = reward,
            steps_taken      = self._step_count,
            max_steps        = max_steps,
        )

    # ─── state property ──────────────────────────────────────────────────────

    @property
    def state(self) -> DefenseState:
        return DefenseState(
            episode_id        = self._episode_id,
            step_count        = self._step_count,
            task_id           = self._task_id,
            difficulty        = TASKS[self._task_id].difficulty if self._task_id else "",
            cumulative_reward = round(self._cumulative_reward, 4),
            best_score        = round(self._best_score, 4),
            solved            = self._solved,
            engagement_log    = list(self._engagement_log),
        )

    # ─── Action handlers ─────────────────────────────────────────────────────

    def _handle_scan(self, action):
        if not action.target_id:
            return None, "target_id is required for scan_airspace.", -0.05, "Scan failed."

        tid = action.target_id
        if tid not in self._scenario["contacts"]:
            return None, f"Target '{tid}' not found in radar picture.", -0.05, "Unknown target."

        contact = self._scenario["contacts"][tid]
        already = self._scanned.get(tid, False)
        self._scanned[tid] = True

        reward = calculate_step_reward(
            action_type        = "scan_airspace",
            target_id          = tid,
            target_affiliation = contact["affiliation"],
            target_type        = contact["type"],
            already_scanned    = already,
            already_engaged    = tid in self._engaged,
        )

        full_contact = dict(contact)
        full_contact["scanned"] = True

        feedback = (
            f"Scan complete: {tid} identified as {contact['affiliation']} "
            f"{contact['type']} ({contact.get('model', 'unknown')})."
        )
        if already:
            feedback = f"Already scanned {tid}. " + feedback

        return full_contact, None, reward, feedback

    def _handle_engage(self, action, weapon: str):
        if not action.target_id:
            return None, f"target_id is required for {weapon}.", -0.05, "Engagement failed."

        tid = action.target_id
        if tid not in self._scenario["contacts"]:
            return None, f"Target '{tid}' not found.", -0.05, "Unknown target."

        contact = self._scenario["contacts"][tid]
        already_engaged = tid in self._engaged

        reward = calculate_step_reward(
            action_type        = weapon,
            target_id          = tid,
            target_affiliation = contact["affiliation"],
            target_type        = contact["type"],
            already_scanned    = self._scanned.get(tid, False),
            already_engaged    = already_engaged,
        )

        if not already_engaged:
            self._engaged[tid] = weapon

        affil = contact["affiliation"]
        ctype = contact["type"]

        if affil == "FRIENDLY":
            feedback = (
                f"⚠️ FRATRICIDE: {tid} is a FRIENDLY {ctype}! "
                "Severe penalty applied. Check IFF before engaging."
            )
        elif weapon == "engage_machine_gun" and ctype == "fighter_jet" and affil == "ENEMY":
            feedback = f"✓ Enemy jet {tid} ({contact.get('model','')}) engaged with machine gun. Neutralised."
        elif weapon == "launch_abm" and ctype == "missile":
            feedback = f"✓ Missile {tid} ({contact.get('model','')}) intercepted with ABM. Threat eliminated."
        elif weapon == "engage_machine_gun" and ctype == "missile":
            feedback = f"⚠ Machine gun used on missile {tid} — ineffective against hypersonic target."
        elif weapon == "launch_abm" and ctype == "fighter_jet":
            feedback = f"⚠ ABM used on fighter jet {tid} — valid but wasteful. Use machine gun for jets."
        else:
            feedback = f"Target {tid} engaged with {weapon}."

        return f"Engagement logged: {tid} via {weapon}", None, reward, feedback

    def _handle_hold(self, action):
        if not action.target_id:
            return None, "target_id is required for hold_fire.", -0.05, "Hold fire failed."

        tid = action.target_id
        if tid not in self._scenario["contacts"]:
            return None, f"Target '{tid}' not found.", -0.05, "Unknown target."

        contact = self._scenario["contacts"][tid]

        if tid not in self._held:
            self._held.append(tid)

        reward = calculate_step_reward(
            action_type        = "hold_fire",
            target_id          = tid,
            target_affiliation = contact["affiliation"],
            target_type        = contact["type"],
            already_scanned    = self._scanned.get(tid, False),
            already_engaged    = tid in self._engaged,
        )

        if contact["affiliation"] == "FRIENDLY":
            feedback = f"✓ Hold fire on {tid} — confirmed friendly. Good IFF discipline."
        else:
            feedback = f"⚠ Hold fire on {tid} — this is a HOSTILE target! Dereliction of duty."

        return f"Hold fire logged: {tid}", None, reward, feedback

    def _handle_report(self, action):
        if not action.engagements:
            return (
                None,
                "engagements list is required for submit_report.",
                -0.05, self._best_score,
                "Report rejected — no engagements provided."
            )

        grader = GRADERS[self._task_id]

        # Build kwargs based on task difficulty (hard has iff_spoofers)
        grader_kwargs = dict(
            submitted_engagements = action.engagements,
            decisions_summary     = action.decisions_summary,
            enemy_jets            = self._scenario["enemy_jets"],
            friendly_jets         = self._scenario["friendly_jets"],
            missiles              = self._scenario["missiles"],
        )
        if self._task_id == "task_hard":
            grader_kwargs["iff_spoofers"] = self._scenario.get("iff_spoofers", [])

        partial_score = grader(**grader_kwargs)

        reward, _ = calculate_report_reward(
            submitted_engagements = action.engagements,
            decisions_summary     = action.decisions_summary,
            enemy_jets            = self._scenario["enemy_jets"],
            friendly_jets         = self._scenario["friendly_jets"],
            missiles              = self._scenario["missiles"],
            already_solved        = self._solved,
        )

        if partial_score >= SOLVE_THRESHOLD:
            self._solved = True
            feedback = (
                f"Mission accomplished. All threats neutralised. "
                f"Final score: {partial_score:.4f}. Well done, commander."
            )
        elif partial_score >= 0.5:
            feedback = (
                f"Mission partially successful. Score: {partial_score:.4f}. "
                "Some threats were missed or wrong weapon used."
            )
        elif partial_score >= 0.3:
            feedback = (
                f"Mission failed. Score: {partial_score:.4f}. "
                "Significant threats were not neutralised or fratricide occurred."
            )
        else:
            feedback = (
                f"Mission critical failure. Score: {partial_score:.4f}. "
                "Base was not defended. Review engagement protocols."
            )

        return (
            f"Report submitted. Score: {partial_score:.4f}",
            None,
            reward,
            partial_score,
            feedback,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _current_radar(self) -> List[Dict]:
        """Return radar picture, revealing scanned contacts' true affiliation."""
        picture = []
        for tid, contact in self._scenario["contacts"].items():
            if self._scanned.get(tid, False):
                picture.append(dict(contact) | {"scanned": True})
            else:
                picture.append({
                    "target_id":    tid,
                    "type":         contact["type"],
                    "affiliation":  "UNKNOWN",
                    "altitude":     contact["altitude"],
                    "speed":        contact["speed"],
                    "bearing":      contact["bearing"],
                    "iff_code":     contact.get("iff_code"),
                    "threat_level": "UNKNOWN",
                    "scanned":      False,
                })
        return picture

    def _error_obs(self, message: str) -> DefenseObservation:
        task_id = self._task_id or "task_easy"
        task    = TASKS.get(task_id, TASKS["task_easy"])
        return DefenseObservation(
            mission_id       = self._scenario["mission_id"] if self._scenario else "",
            task_id          = task_id,
            difficulty       = task.difficulty,
            task_description = task.description,
            radar_contacts   = [],
            threats_in_scope = [],
            action_result    = None,
            action_error     = message,
            engaged_targets  = [],
            held_targets     = [],
            partial_score    = 0.0001,
            feedback         = message,
            done             = False,
            reward           = 0.0,
            steps_taken      = self._step_count,
            max_steps        = task.max_steps,
        )

    def close(self) -> None:
        self._scenario = None