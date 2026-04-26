"""
Training dataset builder for GRPO fine-tuning.
Converts agent trajectories into (prompt, completion, reward) triples.
"""

import json
import random
from typing import Any, Dict, List, Tuple

from agents.radar_agent import RADAR_SYSTEM_PROMPT, RADAR_USER_TEMPLATE
from agents.actor_agent  import ACTOR_SYSTEM_PROMPT, ACTOR_USER_TEMPLATE


def trajectory_to_radar_samples(trajectory: List[Dict]) -> List[Dict]:
    """
    Convert trajectory steps into Agent 1 (radar classification) training samples.

    Each sample:
      prompt      : system + user (raw telemetry)
      completion  : JSON classification the agent produced
      reward      : classification_reward signal
      true_class  : ground truth label
    """
    samples = []
    for step in trajectory:
        contact = step.get("contact", {})
        clf     = step.get("classification", {})
        reward  = step.get("classification_reward", 0.0)
        true_cls = step.get("true_class", "UNKNOWN")

        user_msg = RADAR_USER_TEMPLATE.format(
            target_id            = contact.get("target_id","?"),
            type                 = contact.get("type","?"),
            velocity_mach        = contact.get("velocity_mach","?"),
            speed                = contact.get("speed","?"),
            altitude             = contact.get("altitude","?"),
            altitude_ft          = contact.get("altitude_ft","?"),
            trajectory_type      = contact.get("trajectory_type","?"),
            bearing              = contact.get("bearing","?"),
            iff_code             = contact.get("iff_code","None"),
            transponder          = contact.get("transponder","None"),
            rcs_m2               = contact.get("rcs_m2","?"),
            rcs_category         = contact.get("rcs_category","?"),
            flight_plan_exists   = contact.get("flight_plan_exists","?"),
            diplomatic_clearance = contact.get("diplomatic_clearance","?"),
            zone                 = contact.get("zone","buffer"),
        )

        completion = json.dumps({
            "contact_class":        clf.get("contact_class","UNKNOWN"),
            "confidence":           clf.get("confidence", 0.5),
            "threat_level":         clf.get("threat_level","UNKNOWN"),
            "kinematics_summary":   clf.get("kinematics_summary",""),
            "electronic_id_summary": clf.get("electronic_id_summary",""),
            "admin_data_summary":   clf.get("admin_data_summary",""),
            "spatial_zone":         clf.get("spatial_zone","BUFFER"),
            "reasoning":            clf.get("reasoning",""),
        })

        samples.append({
            "prompt":      f"<|system|>\n{RADAR_SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n<|assistant|>\n",
            "completion":  completion,
            "reward":      reward,
            "true_class":  true_cls,
            "target_id":   contact.get("target_id","?"),
        })

    return samples


def trajectory_to_actor_samples(trajectory: List[Dict]) -> List[Dict]:
    """
    Convert trajectory steps into Agent 2 (actor) training samples.

    Each sample:
      prompt      : system + classification (from Agent 1)
      completion  : JSON action the agent produced
      reward      : action_reward signal
    """
    samples = []
    for step in trajectory:
        clf    = step.get("classification", {})
        action = step.get("action", {})
        reward = step.get("action_reward", 0.0)

        user_msg = ACTOR_USER_TEMPLATE.format(
            target_id             = clf.get("target_id","?"),
            contact_class         = clf.get("contact_class","UNKNOWN"),
            confidence            = clf.get("confidence","?"),
            threat_level          = clf.get("threat_level","?"),
            spatial_zone          = clf.get("spatial_zone","BUFFER"),
            kinematics_summary    = clf.get("kinematics_summary","N/A"),
            electronic_id_summary = clf.get("electronic_id_summary","N/A"),
            admin_data_summary    = clf.get("admin_data_summary","N/A"),
            reasoning             = clf.get("reasoning","N/A"),
        )

        completion = json.dumps({
            "action":        action.get("action","SYS_TRACK_ONLY"),
            "justification": action.get("justification",""),
            "priority":      action.get("priority","ROUTINE"),
        })

        samples.append({
            "prompt":      f"<|system|>\n{ACTOR_SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n<|assistant|>\n",
            "completion":  completion,
            "reward":      reward,
            "true_class":  step.get("true_class","UNKNOWN"),
            "target_id":   clf.get("target_id","?"),
        })

    return samples


def build_grpo_dataset(
    trajectories: List[Dict],
    agent: str = "radar",   # "radar" | "actor"
    train_ratio: float = 0.85,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build train/eval split from trajectories.

    Args:
        trajectories: flat list of step records
        agent:        which agent's samples to build
        train_ratio:  fraction for training
        shuffle:      shuffle before split
        seed:         random seed

    Returns:
        (train_samples, eval_samples)
    """
    if agent == "radar":
        samples = trajectory_to_radar_samples(trajectories)
    else:
        samples = trajectory_to_actor_samples(trajectories)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    split = int(len(samples) * train_ratio)
    return samples[:split], samples[split:]


def save_dataset(samples: List[Dict], path: str) -> None:
    """Save samples to JSONL file."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[Dataset] Saved {len(samples)} samples >> {path}")


def load_dataset(path: str) -> List[Dict]:
    """Load samples from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
