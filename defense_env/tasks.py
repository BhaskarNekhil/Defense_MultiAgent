"""
Task definitions and graders for the Defense AI Environment.

Each task defines:
  task_id          : unique string identifier
  difficulty       : easy | medium | hard
  description      : what the agent must accomplish
  max_steps        : episode step limit
  grader()         : pure function → float 0.0-1.0

Reward design philosophy:
  - Engage every enemy jet with machine gun       → positive
  - Engage every missile with ABM                 → positive
  - Engage a friendly jet                         → heavy penalty (fratricide)
  - Miss an enemy / missile                       → partial penalty
  - Correct engagement tool for target type       → bonus
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── TASK DATACLASS ───────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id:     str
    difficulty:  str
    description: str
    max_steps:   int
    hints:       List[str] = field(default_factory=list)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    return round(max(0.01, min(0.99, score)), 4)


TACTICAL_KEYWORDS = [
    "iff", "identification", "radar", "scan", "spoofing",
    "missile", "intercept", "abm", "machine gun", "engage",
    "fratricide", "threat", "bearing", "altitude", "friendly",
    "enemy", "hostile", "discrimination",
]
TACTICAL_KEYWORD_MIN = 2


def _report_quality(summary: Optional[str]) -> float:
    if not summary:
        return 0.0
    lower = summary.lower()
    hits = sum(1 for kw in TACTICAL_KEYWORDS if kw in lower)
    return 0.15 if hits >= TACTICAL_KEYWORD_MIN else 0.0


def _engagement_score(
    submitted_engagements: Optional[List[Dict[str, str]]],
    enemy_jets: List[str],
    friendly_jets: List[str],
    missiles: List[str],
    per_enemy: float,
    per_missile: float,
    per_fratricide: float,
    wrong_weapon_penalty: float,
) -> tuple[float, int, int, int]:
    """
    Returns (score, correct_engagements, fratricides, wrong_weapon_count).
    """
    if not submitted_engagements:
        return 0.0, 0, 0, 0

    score = 0.0
    correct = 0
    fratricides = 0
    wrong_weapon = 0

    acted_on = set()

    for eng in submitted_engagements:
        tid    = eng.get("target_id", "")
        action = (eng.get("action") or "").lower()

        if tid in acted_on:
            continue  # skip duplicate actions on same target
        acted_on.add(tid)

        if tid in friendly_jets:
            # hold_fire on a friendly is CORRECT — not fratricide
            # Only destructive weapons (machine gun / ABM) on a friendly = fratricide
            if action in ("engage_machine_gun", "launch_abm"):
                score -= per_fratricide
                fratricides += 1
            else:
                # hold_fire on friendly = correct engagement
                correct += 1

        elif tid in enemy_jets:
            if action == "engage_machine_gun":
                score += per_enemy
                correct += 1
            else:
                # Wrong weapon (e.g. ABM on a jet)
                score += per_enemy * 0.3    # partial credit for engaging, but wrong tool
                wrong_weapon += 1

        elif tid in missiles:
            if action == "launch_abm":
                score += per_missile
                correct += 1
            else:
                # Wrong weapon (e.g. machine gun on supersonic missile)
                score += per_missile * 0.2  # machine gun can't realistically kill a missile
                wrong_weapon += 1

    return score, correct, fratricides, wrong_weapon


# ─── TASK 1 — EASY ────────────────────────────────────────────────────────────

TASK_EASY = Task(
    task_id    = "task_easy",
    difficulty = "easy",
    max_steps  = 10,
    description = (
        "Radar has detected 1-2 unidentified aircraft approaching your base "
        "at high speed with no IFF transponder signal. "
        "Scan the contacts to confirm hostile status, then engage all enemy "
        "fighter jets with machine gun fire. "
        "Submit your report when all threats are neutralised."
    ),
    hints = [
        "Use scan_airspace on each contact to get full IFF data.",
        "No IFF code + high speed + hostile bearing = enemy jet.",
        "Engage enemy jets with engage_machine_gun — not ABM.",
        "Submit your report to end the mission.",
    ],
)


def grade_easy(
    submitted_engagements: Optional[List[Dict[str, str]]],
    decisions_summary: Optional[str],
    enemy_jets: List[str],
    friendly_jets: List[str],
    missiles: List[str],
) -> float:
    """
    Scoring (max 1.0):
      0.55 — all enemy jets correctly engaged with machine gun (split evenly)
      0.15 — no fratricides (0 friendly jets engaged)
      0.15 — correct weapon choice (machine gun, not ABM)
      0.15 — tactical report quality
    """
    n_enemies = max(len(enemy_jets), 1)
    per_enemy = 0.55 / n_enemies

    eng_score, correct, fratricides, wrong_weapon = _engagement_score(
        submitted_engagements=submitted_engagements,
        enemy_jets=enemy_jets,
        friendly_jets=friendly_jets,
        missiles=missiles,
        per_enemy=per_enemy,
        per_missile=0.0,
        per_fratricide=0.30,
        wrong_weapon_penalty=0.10,
    )

    score = max(0.0, eng_score)

    # No fratricide bonus
    if fratricides == 0:
        score += 0.15

    # Correct weapon bonus (machine gun used on jets)
    if wrong_weapon == 0 and correct > 0:
        score += 0.15

    # Report quality
    score += _report_quality(decisions_summary)

    return _clamp(score)


# ─── TASK 2 — MEDIUM ──────────────────────────────────────────────────────────

TASK_MEDIUM = Task(
    task_id    = "task_medium",
    difficulty = "medium",
    max_steps  = 18,
    description = (
        "A mixed air picture: enemy fighter jets, friendly aircraft, and "
        "an inbound missile are all present simultaneously. "
        "Scan all contacts to discriminate friend from foe. "
        "Engage enemy jets with machine gun, intercept the missile with ABM, "
        "and hold fire on all friendly aircraft. "
        "Fratricide (engaging a friendly) is mission failure."
    ),
    hints = [
        "Always scan_airspace before engaging — friendlies have valid IFF codes.",
        "Missiles are fast, low-altitude, no IFF — use launch_abm.",
        "Enemy jets have no IFF code and hostile bearings — use engage_machine_gun.",
        "Use hold_fire on any contact with a valid IFF transponder.",
        "Prioritise the missile — it has a time-to-impact countdown.",
    ],
)


def grade_medium(
    submitted_engagements: Optional[List[Dict[str, str]]],
    decisions_summary: Optional[str],
    enemy_jets: List[str],
    friendly_jets: List[str],
    missiles: List[str],
) -> float:
    """
    Scoring (max 1.0):
      0.35 — enemy jets correctly engaged with machine gun
      0.25 — missiles intercepted with ABM
      0.10 — friendlies held / not engaged
      0.15 — correct weapon per target type
      0.15 — tactical report quality
    """
    n_enemies  = max(len(enemy_jets), 1)
    n_missiles = max(len(missiles), 1)
    per_enemy  = 0.35 / n_enemies
    per_msle   = 0.25 / n_missiles

    eng_score, correct, fratricides, wrong_weapon = _engagement_score(
        submitted_engagements=submitted_engagements,
        enemy_jets=enemy_jets,
        friendly_jets=friendly_jets,
        missiles=missiles,
        per_enemy=per_enemy,
        per_missile=per_msle,
        per_fratricide=0.40,  # Heavy penalty in mixed environment
        wrong_weapon_penalty=0.10,
    )

    score = max(0.0, eng_score)

    # Friendly fire avoided
    if fratricides == 0:
        score += 0.10

    # Correct weapon for all engagements
    if wrong_weapon == 0 and correct > 0:
        score += 0.15

    score += _report_quality(decisions_summary)

    return _clamp(score)


# ─── TASK 3 — HARD ────────────────────────────────────────────────────────────

TASK_HARD = Task(
    task_id    = "task_hard",
    difficulty = "hard",
    max_steps  = 25,
    description = (
        "SATURATION ATTACK: Multiple enemy jets, friendlies, and missiles "
        "are incoming simultaneously. Enemy jets may be SPOOFING IFF codes "
        "to appear as friendlies — do not trust IFF alone. "
        "Use scan_airspace for full intelligence on each contact. "
        "Identify spoofers by cross-referencing IFF code with speed, "
        "altitude, bearing, and threat level. "
        "Engage all true enemy jets with machine gun, "
        "intercept all missiles with ABM, "
        "and protect all genuine friendlies. "
        "Submit a full tactical report."
    ),
    hints = [
        "IFF code alone is not reliable — enemies may spoof.",
        "Cross-check: a 'friendly' flying at supersonic speed toward base is suspicious.",
        "Scan EVERY contact — the scan reveals threat_level and true affiliation.",
        "Missiles take priority — use launch_abm immediately on any missile contact.",
        "After missile intercepts, engage verified enemy jets with machine gun.",
        "Hold fire on contacts confirmed as FRIENDLY after scan.",
    ],
)


def grade_hard(
    submitted_engagements: Optional[List[Dict[str, str]]],
    decisions_summary: Optional[str],
    enemy_jets: List[str],
    friendly_jets: List[str],
    missiles: List[str],
    iff_spoofers: Optional[List[str]] = None,
) -> float:
    """
    Scoring (max 1.0):
      0.30 — enemy jets (including spoofers) correctly engaged with machine gun
      0.25 — all missiles intercepted with ABM
      0.12 — zero fratricide
      0.08 — spoofers correctly identified and engaged (not held)
      0.10 — correct weapon per target type
      0.15 — tactical report mentions spoofing, scanning, discrimination
    """
    iff_spoofers = iff_spoofers or []
    n_enemies    = max(len(enemy_jets), 1)
    n_missiles   = max(len(missiles), 1)
    per_enemy    = 0.30 / n_enemies
    per_msle     = 0.25 / n_missiles

    eng_score, correct, fratricides, wrong_weapon = _engagement_score(
        submitted_engagements=submitted_engagements,
        enemy_jets=enemy_jets,
        friendly_jets=friendly_jets,
        missiles=missiles,
        per_enemy=per_enemy,
        per_missile=per_msle,
        per_fratricide=0.45,  # Maximum penalty in saturation attack
        wrong_weapon_penalty=0.10,
    )

    score = max(0.0, eng_score)

    # Zero fratricide
    if fratricides == 0:
        score += 0.12

    # Spoofers correctly engaged (extra credit)
    if iff_spoofers and submitted_engagements:
        engaged_ids = {e.get("target_id") for e in submitted_engagements}
        spoofers_caught = sum(1 for s in iff_spoofers if s in engaged_ids)
        spoofer_ratio = spoofers_caught / max(len(iff_spoofers), 1)
        score += 0.08 * spoofer_ratio

    # Correct weapon for all engagements
    if wrong_weapon == 0 and correct > 0:
        score += 0.10

    # Report quality with extra credit for mentioning spoofing
    rq = _report_quality(decisions_summary)
    if decisions_summary and "spoof" in decisions_summary.lower():
        rq = min(rq + 0.05, 0.15)
    score += rq

    return _clamp(score)


# ─── TASK REGISTRY ────────────────────────────────────────────────────────────

TASKS = {
    "task_easy":   TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard":   TASK_HARD,
}

TASK_ORDER = ["task_easy", "task_medium", "task_hard"]

GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}
