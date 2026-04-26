"""
Reward functions for the Multi-Agent Defense System.

Covers:
  calculate_step_reward()        — per-step shaped reward (env step)
  calculate_report_reward()      — end-of-episode report reward
  classification_reward()        — Agent 1: reward for correct 7-class label
  action_appropriateness_reward()— Agent 2: reward for correct action given class
"""

from typing import Any, Dict, List, Optional


# ─── STEP-LEVEL REWARD CONSTANTS ─────────────────────────────────────────────

STEP_PENALTY                = -0.02

SCAN_ENEMY_CONTACT          = +0.04
SCAN_FRIENDLY_CONTACT       = +0.04
SCAN_ALREADY_DONE           = -0.02

ENGAGE_ENEMY_JET_CORRECT    = +0.15
ENGAGE_ENEMY_JET_WRONG_TOOL = +0.05
ENGAGE_FRIENDLY_JET         = -0.50   # FRATRICIDE
ENGAGE_MISSILE_CORRECT      = +0.20
ENGAGE_MISSILE_WRONG_TOOL   = -0.05
HOLD_FRIENDLY_CORRECT       = +0.08
HOLD_ENEMY                  = -0.10

# ─── REPORT-LEVEL REWARD CONSTANTS ───────────────────────────────────────────

REPORT_ALL_ENEMIES_NEUTRALISED  = +0.40
REPORT_ALL_MISSILES_INTERCEPTED = +0.30
REPORT_ZERO_FRATRICIDE          = +0.20
REPORT_GOOD_TACTICAL_SUMMARY    = +0.15
SOLVE_BONUS                     = +0.25

REPORT_MISSED_ENEMY             = -0.15
REPORT_MISSED_MISSILE           = -0.25
REPORT_FRATRICIDE_PENALTY       = -0.40

# ─── CLASSIFICATION REWARD CONSTANTS (Agent 1) ───────────────────────────────

CLASS_CORRECT               = +0.30   # Correct 7-class label
CLASS_ALMOST               = +0.10   # Close but wrong (e.g. ENEMY vs FOREIGN_UNPERMITTED)
CLASS_WRONG_LETHAL         = -0.40   # Misclassified friendly as enemy (risk of fratricide)
CLASS_WRONG_MISS           = -0.20   # Misclassified missile as something safe

REASONING_QUALITY_BONUS    = +0.10   # Used kinematics + electronic ID + admin + zone
CONFIDENCE_CALIBRATION     = +0.05   # High confidence on correct label

# ─── ACTION REWARD CONSTANTS (Agent 2) ───────────────────────────────────────

ACTION_CORRECT             = +0.25   # Best action for the classification
ACTION_ACCEPTABLE          = +0.10   # Valid but suboptimal action
ACTION_WRONG_PEACEFUL      = -0.10   # Used a weapon on a non-threat
ACTION_FRATRICIDE_ACTION   = -0.50   # Engaged a friendly contact
ACTION_MISSED_CRITICAL     = -0.30   # Failed to engage MISSILE_INBOUND

# ─── CLAMP HELPERS ───────────────────────────────────────────────────────────

def _clamp_score(score: float) -> float:
    return round(max(0.01, min(0.99, score)), 4)

def _clamp_reward(reward: float) -> float:
    return round(max(-0.99, min(0.99, reward)), 4)


# ─── CORRECT ACTION MAPPING ──────────────────────────────────────────────────
# Maps true contact class → optimal Agent 2 action

CORRECT_ACTIONS = {
    "MISSILE_INBOUND":      {"WEAPON_ABM_LAUNCH"},
    "ENEMY_AIRCRAFT":       {"WEAPON_ADS_ENGAGE", "WEAPON_ABM_LAUNCH"},
    "FRIENDLY_AIRCRAFT":    {"SYS_TRACK_ONLY"},
    "DOMESTIC_FLIGHT":      {"SYS_TRACK_ONLY", "COMM_HELLO"},
    "FOREIGN_PERMITTED":    {"COMM_HELLO", "NAV_GUIDE", "SYS_TRACK_ONLY"},
    "FOREIGN_UNPERMITTED":  {"COMM_WARN", "SYS_TRACK_ONLY"},
    "OWN_ASSET":            {"SYS_TRACK_ONLY"},
}

ACCEPTABLE_ACTIONS = {
    "MISSILE_INBOUND":      {"WEAPON_ADS_ENGAGE"},          # CIWS as fallback
    "ENEMY_AIRCRAFT":       {"COMM_WARN"},                  # warn before engage
    "FRIENDLY_AIRCRAFT":    {"COMM_HELLO", "NAV_GUIDE"},
    "DOMESTIC_FLIGHT":      {"NAV_GUIDE"},
    "FOREIGN_PERMITTED":    {"COMM_WARN"},
    "FOREIGN_UNPERMITTED":  {"NAV_GUIDE"},
    "OWN_ASSET":            {"COMM_HELLO"},
}

LETHAL_CLASSES = {"FRIENDLY_AIRCRAFT", "DOMESTIC_FLIGHT", "FOREIGN_PERMITTED", "OWN_ASSET"}
LETHAL_ACTIONS = {"WEAPON_ABM_LAUNCH", "WEAPON_ADS_ENGAGE"}


# ─── STEP REWARD CALCULATOR (legacy — unchanged) ──────────────────────────────

def calculate_step_reward(
    action_type: str,
    target_id: Optional[str],
    target_affiliation: Optional[str],
    target_type: Optional[str],
    already_scanned: bool,
    already_engaged: bool,
) -> float:
    """Returns shaped reward for one step action (backward-compatible)."""
    reward = STEP_PENALTY

    if action_type == "scan_airspace":
        if already_scanned:
            reward += SCAN_ALREADY_DONE
        elif target_affiliation in ("ENEMY", "HOSTILE"):
            reward += SCAN_ENEMY_CONTACT
        elif target_affiliation == "FRIENDLY":
            reward += SCAN_FRIENDLY_CONTACT

    elif action_type == "engage_machine_gun":
        if already_engaged:
            return _clamp_reward(STEP_PENALTY)
        if target_affiliation == "FRIENDLY":
            reward += ENGAGE_FRIENDLY_JET
        elif target_affiliation in ("ENEMY",) and target_type == "fighter_jet":
            reward += ENGAGE_ENEMY_JET_CORRECT
        elif target_affiliation == "HOSTILE" and target_type == "missile":
            reward += ENGAGE_MISSILE_WRONG_TOOL
        else:
            reward += ENGAGE_ENEMY_JET_WRONG_TOOL

    elif action_type == "launch_abm":
        if already_engaged:
            return _clamp_reward(STEP_PENALTY)
        if target_affiliation == "FRIENDLY":
            reward += ENGAGE_FRIENDLY_JET
        elif target_affiliation == "HOSTILE" and target_type == "missile":
            reward += ENGAGE_MISSILE_CORRECT
        elif target_affiliation == "ENEMY" and target_type == "fighter_jet":
            reward += ENGAGE_ENEMY_JET_WRONG_TOOL
        else:
            reward += -0.05

    elif action_type == "hold_fire":
        if target_affiliation == "FRIENDLY":
            reward += HOLD_FRIENDLY_CORRECT
        elif target_affiliation in ("ENEMY", "HOSTILE"):
            reward += HOLD_ENEMY

    return _clamp_reward(reward)


# ─── REPORT REWARD CALCULATOR (legacy — unchanged) ───────────────────────────

def calculate_report_reward(
    submitted_engagements: Optional[List[Dict[str, str]]],
    decisions_summary: Optional[str],
    enemy_jets: List[str],
    friendly_jets: List[str],
    missiles: List[str],
    already_solved: bool,
) -> tuple:
    """Returns (reward, partial_score) for the submit_report action."""
    if already_solved:
        return (0.0, _clamp_score(1.0))

    reward = STEP_PENALTY
    score  = 0.0
    submitted_engagements = submitted_engagements or []
    decisions_summary     = decisions_summary or ""

    engaged_ids = {e.get("target_id", "") for e in submitted_engagements}
    action_map  = {e.get("target_id", ""): (e.get("action") or "").lower()
                   for e in submitted_engagements}

    # Enemy jets
    enemies_neutralised = sum(
        1 for ej in enemy_jets
        if ej in engaged_ids and action_map.get(ej) == "engage_machine_gun"
    )
    missed_enemies = len(enemy_jets) - enemies_neutralised
    if len(enemy_jets) > 0 and enemies_neutralised == len(enemy_jets):
        reward += REPORT_ALL_ENEMIES_NEUTRALISED
        score  += 0.35
    elif enemies_neutralised > 0:
        ratio   = enemies_neutralised / len(enemy_jets)
        reward += REPORT_ALL_ENEMIES_NEUTRALISED * ratio
        score  += 0.35 * ratio
    reward += missed_enemies * REPORT_MISSED_ENEMY

    # Missiles
    missiles_intercepted = sum(
        1 for m in missiles
        if m in engaged_ids and action_map.get(m) == "launch_abm"
    )
    missed_missiles = len(missiles) - missiles_intercepted
    if len(missiles) > 0 and missiles_intercepted == len(missiles):
        reward += REPORT_ALL_MISSILES_INTERCEPTED
        score  += 0.25
    elif missiles_intercepted > 0:
        ratio   = missiles_intercepted / len(missiles)
        reward += REPORT_ALL_MISSILES_INTERCEPTED * ratio
        score  += 0.25 * ratio
    reward += missed_missiles * REPORT_MISSED_MISSILE

    # Friendly fire
    fratricides = sum(1 for fj in friendly_jets if fj in engaged_ids)
    if fratricides == 0:
        reward += REPORT_ZERO_FRATRICIDE
        score  += 0.20
    else:
        reward += fratricides * REPORT_FRATRICIDE_PENALTY

    # Report quality
    tactical_keywords = [
        "iff", "scan", "missile", "intercept", "abm", "machine gun",
        "enemy", "friendly", "hostile", "engage", "threat",
        "spoofing", "discrimination", "radar",
    ]
    hits = sum(1 for kw in tactical_keywords if kw in decisions_summary.lower())
    if hits >= 2:
        reward += REPORT_GOOD_TACTICAL_SUMMARY
        score  += 0.15

    partial_score = _clamp_score(score)
    if partial_score >= 0.80:
        reward += SOLVE_BONUS

    return (_clamp_reward(reward), partial_score)


# ─── NEW: CLASSIFICATION REWARD (Agent 1) ────────────────────────────────────

def classification_reward(
    predicted_class: str,
    true_class: str,
    confidence: float,
    reasoning: str,
) -> float:
    """
    Reward for Agent 1's contact classification.
    
    Args:
        predicted_class: Agent 1's predicted class label
        true_class:      Ground truth class from scenario
        confidence:      Agent 1's stated confidence (0-1)
        reasoning:       Agent 1's chain-of-thought text
    
    Returns:
        Shaped reward float
    """
    reward = 0.0

    if predicted_class == true_class:
        reward += CLASS_CORRECT

        # Calibration bonus: high confidence on correct label
        if confidence >= 0.80:
            reward += CONFIDENCE_CALIBRATION

        # Reasoning quality bonus
        reasoning_lower = reasoning.lower()
        axes_covered = sum([
            any(kw in reasoning_lower for kw in ["mach", "speed", "altitude", "trajectory"]),
            any(kw in reasoning_lower for kw in ["iff", "transponder", "rcs", "radar cross"]),
            any(kw in reasoning_lower for kw in ["flight plan", "clearance", "diplomatic", "registration"]),
            any(kw in reasoning_lower for kw in ["zone", "buffer", "critical", "range"]),
        ])
        if axes_covered >= 3:
            reward += REASONING_QUALITY_BONUS
        elif axes_covered >= 2:
            reward += REASONING_QUALITY_BONUS * 0.5

    else:
        # Partial credit for close misses
        _close_pairs = {
            ("ENEMY_AIRCRAFT", "FOREIGN_UNPERMITTED"),
            ("FOREIGN_UNPERMITTED", "ENEMY_AIRCRAFT"),
            ("FRIENDLY_AIRCRAFT", "DOMESTIC_FLIGHT"),
            ("DOMESTIC_FLIGHT", "FRIENDLY_AIRCRAFT"),
            ("FOREIGN_PERMITTED", "DOMESTIC_FLIGHT"),
            ("OWN_ASSET", "FRIENDLY_AIRCRAFT"),
        }
        if (predicted_class, true_class) in _close_pairs:
            reward += CLASS_ALMOST
        elif true_class in LETHAL_CLASSES and predicted_class in {"ENEMY_AIRCRAFT", "MISSILE_INBOUND"}:
            # Classified a safe target as hostile → fratricide risk
            reward += CLASS_WRONG_LETHAL
        elif true_class == "MISSILE_INBOUND":
            # Missed an incoming missile
            reward += CLASS_WRONG_MISS
        else:
            reward += -0.10  # Generic wrong classification

    return _clamp_reward(reward)


# ─── NEW: ACTION APPROPRIATENESS REWARD (Agent 2) ────────────────────────────

def action_appropriateness_reward(
    chosen_action: str,
    true_class: str,
    predicted_class: str,
) -> float:
    """
    Reward for Agent 2's action choice.
    
    Rewards based on true_class (not predicted_class), since environment
    gives real consequences regardless of Agent 1's classification.
    
    Args:
        chosen_action:   Agent 2's chosen action category
        true_class:      Ground truth contact class
        predicted_class: Agent 1's predicted class (may differ)
    
    Returns:
        Shaped reward float
    """
    reward = 0.0

    correct_set    = CORRECT_ACTIONS.get(true_class, set())
    acceptable_set = ACCEPTABLE_ACTIONS.get(true_class, set())

    if chosen_action in correct_set:
        reward += ACTION_CORRECT

        # Bonus for correct action on missile (time-critical)
        if true_class == "MISSILE_INBOUND" and chosen_action == "WEAPON_ABM_LAUNCH":
            reward += 0.10

    elif chosen_action in acceptable_set:
        reward += ACTION_ACCEPTABLE

    elif true_class in LETHAL_CLASSES and chosen_action in LETHAL_ACTIONS:
        # Firing at a non-threat (civilian, friendly, own asset)
        if true_class in {"FRIENDLY_AIRCRAFT", "OWN_ASSET"}:
            reward += ACTION_FRATRICIDE_ACTION
        else:
            reward += ACTION_WRONG_PEACEFUL

    elif true_class == "MISSILE_INBOUND" and chosen_action not in LETHAL_ACTIONS:
        # Failed to engage an inbound missile
        reward += ACTION_MISSED_CRITICAL

    else:
        reward += -0.05  # Suboptimal but not catastrophic

    return _clamp_reward(reward)


# ─── COMBINED REWARD (for GRPO training) ─────────────────────────────────────

def combined_step_reward(
    predicted_class: str,
    chosen_action: str,
    true_class: str,
    confidence: float,
    reasoning: str,
    env_reward: float,
    w_class: float = 0.35,
    w_action: float = 0.35,
    w_env: float = 0.30,
) -> Dict[str, float]:
    """
    Weighted combination of classification + action + environment rewards.
    
    Returns a dict with individual components and total.
    """
    r_class  = classification_reward(predicted_class, true_class, confidence, reasoning)
    r_action = action_appropriateness_reward(chosen_action, true_class, predicted_class)
    r_env    = _clamp_reward(env_reward)

    total = w_class * r_class + w_action * r_action + w_env * r_env

    return {
        "classification_reward":  round(r_class,  4),
        "action_reward":          round(r_action, 4),
        "env_reward":             round(r_env,    4),
        "total_reward":           round(_clamp_reward(total), 4),
    }
