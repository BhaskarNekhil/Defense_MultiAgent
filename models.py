"""
Data models for the Defense AI Environment — Multi-Agent Edition.

Models:
  DefenseAction          — legacy environment action (backward-compatible)
  DefenseObservation     — environment observation
  DefenseState           — episode metadata

  ContactClassification  — Agent 1 output (7-class label + reasoning)
  AgentAction            — Agent 2 output (6 action categories)
  MultiAgentStep         — full pipeline record for one contact
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# ── Classification labels (Agent 1) ──────────────────────────────────────────

CONTACT_CLASSES = Literal[
    "MISSILE_INBOUND",
    "ENEMY_AIRCRAFT",
    "FRIENDLY_AIRCRAFT",
    "DOMESTIC_FLIGHT",
    "FOREIGN_PERMITTED",
    "FOREIGN_UNPERMITTED",
    "OWN_ASSET",
    "UNKNOWN",
]

# ── Action categories (Agent 2) ───────────────────────────────────────────────

ACTION_CATEGORIES = Literal[
    "COMM_HELLO",
    "COMM_WARN",
    "NAV_GUIDE",
    "SYS_TRACK_ONLY",
    "WEAPON_ABM_LAUNCH",
    "WEAPON_ADS_ENGAGE",
]

# ── Action → environment step mapping ─────────────────────────────────────────
# Maps Agent 2's semantic action to the existing DefenseEnvironment action type

ACTION_TO_ENV = {
    "COMM_HELLO":         "hold_fire",            # welcome/acknowledge
    "COMM_WARN":          "hold_fire",            # warn but don't fire yet
    "NAV_GUIDE":          "hold_fire",            # guide to landing
    "SYS_TRACK_ONLY":     "hold_fire",            # radar lock only
    "WEAPON_ABM_LAUNCH":  "launch_abm",           # intercept missile
    "WEAPON_ADS_ENGAGE":  "engage_machine_gun",   # activate ADS/CIWS
}

# ── Default action per classification ─────────────────────────────────────────

DEFAULT_ACTION_MAP: Dict[str, str] = {
    "MISSILE_INBOUND":      "WEAPON_ABM_LAUNCH",
    "ENEMY_AIRCRAFT":       "WEAPON_ADS_ENGAGE",
    "FRIENDLY_AIRCRAFT":    "SYS_TRACK_ONLY",
    "DOMESTIC_FLIGHT":      "SYS_TRACK_ONLY",
    "FOREIGN_PERMITTED":    "COMM_HELLO",
    "FOREIGN_UNPERMITTED":  "COMM_WARN",
    "OWN_ASSET":            "SYS_TRACK_ONLY",
    "UNKNOWN":              "SYS_TRACK_ONLY",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models (with dataclass fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, Field as PField

    # ── Legacy environment models ─────────────────────────────────────────────

    class DefenseAction(BaseModel):
        action_type:       str
        target_id:         Optional[str]                  = None
        decisions_summary: Optional[str]                  = None
        engagements:       Optional[List[Dict[str, str]]] = None
        model_config = {"extra": "allow"}

    class DefenseObservation(BaseModel):
        mission_id:       str                  = ""
        task_id:          str                  = ""
        difficulty:       str                  = ""
        task_description: str                  = ""
        radar_contacts:   List[Dict[str, Any]] = PField(default_factory=list)
        threats_in_scope: List[str]            = PField(default_factory=list)
        action_result:    Optional[Any]        = None
        action_error:     Optional[str]        = None
        engaged_targets:  List[str]            = PField(default_factory=list)
        held_targets:     List[str]            = PField(default_factory=list)
        partial_score:    float                = 0.0
        feedback:         str                  = ""
        done:             bool                 = False
        reward:           float                = 0.0
        steps_taken:      int                  = 0
        max_steps:        int                  = 15
        model_config = {"extra": "allow"}
        def model_dump(self): return self.__dict__.copy()

    class DefenseState(BaseModel):
        episode_id:        str       = ""
        step_count:        int       = 0
        task_id:           str       = ""
        difficulty:        str       = ""
        cumulative_reward: float     = 0.0
        best_score:        float     = 0.0
        solved:            bool      = False
        engagement_log:    List[str] = PField(default_factory=list)
        model_config = {"extra": "allow"}
        def model_dump(self): return self.__dict__.copy()

    # ── Agent 1 output — Contact Classification ───────────────────────────────

    class ContactClassification(BaseModel):
        """Output of Agent 1 (QwenRadarAgent) for a single radar contact."""
        target_id:      str
        contact_class:  str   = "UNKNOWN"   # one of CONTACT_CLASSES
        confidence:     float = 0.0         # 0.0 – 1.0
        threat_level:   str   = "UNKNOWN"   # NONE | LOW | MEDIUM | HIGH | CRITICAL
        reasoning:      str   = ""          # full chain-of-thought text
        # Reasoning axes breakdown
        kinematics_summary:  str = ""
        electronic_id_summary: str = ""
        admin_data_summary:  str = ""
        spatial_zone:        str = ""       # BUFFER | CRITICAL
        model_config = {"extra": "allow"}
        def model_dump(self): return self.__dict__.copy()

    # ── Agent 2 output — Tactical Action ─────────────────────────────────────

    class AgentAction(BaseModel):
        """Output of Agent 2 (QwenActorAgent) for a classified contact."""
        target_id:       str
        action:          str   = "SYS_TRACK_ONLY"   # one of ACTION_CATEGORIES
        justification:   str   = ""
        priority:        str   = "ROUTINE"           # IMMEDIATE | HIGH | ROUTINE | MONITOR
        env_action_type: str   = "hold_fire"         # mapped environment action
        model_config = {"extra": "allow"}
        def model_dump(self): return self.__dict__.copy()

    # ── Full pipeline record for one contact ──────────────────────────────────

    class MultiAgentStep(BaseModel):
        """Complete record of both agents processing one radar contact."""
        episode_id:      str                = ""
        step_num:        int                = 0
        contact:         Dict[str, Any]     = PField(default_factory=dict)
        classification:  Optional[Dict]     = None   # ContactClassification.model_dump()
        action:          Optional[Dict]     = None   # AgentAction.model_dump()
        env_reward:      float              = 0.0
        classification_reward: float        = 0.0
        action_reward:   float              = 0.0
        total_reward:    float              = 0.0
        model_config = {"extra": "allow"}
        def model_dump(self): return self.__dict__.copy()

except ImportError:
    # ── Dataclass fallback ────────────────────────────────────────────────────

    @dataclass
    class DefenseAction:
        action_type:       str
        target_id:         Optional[str]                  = None
        decisions_summary: Optional[str]                  = None
        engagements:       Optional[List[Dict[str, str]]] = None

    @dataclass
    class DefenseObservation:
        mission_id:       str                  = ""
        task_id:          str                  = ""
        difficulty:       str                  = ""
        task_description: str                  = ""
        radar_contacts:   List[Dict[str, Any]] = field(default_factory=list)
        threats_in_scope: List[str]            = field(default_factory=list)
        action_result:    Optional[Any]        = None
        action_error:     Optional[str]        = None
        engaged_targets:  List[str]            = field(default_factory=list)
        held_targets:     List[str]            = field(default_factory=list)
        partial_score:    float                = 0.0
        feedback:         str                  = ""
        done:             bool                 = False
        reward:           float                = 0.0
        steps_taken:      int                  = 0
        max_steps:        int                  = 15
        def model_dump(self):
            import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class DefenseState:
        episode_id:        str       = ""
        step_count:        int       = 0
        task_id:           str       = ""
        difficulty:        str       = ""
        cumulative_reward: float     = 0.0
        best_score:        float     = 0.0
        solved:            bool      = False
        engagement_log:    List[str] = field(default_factory=list)
        def model_dump(self):
            import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class ContactClassification:
        target_id:             str   = ""
        contact_class:         str   = "UNKNOWN"
        confidence:            float = 0.0
        threat_level:          str   = "UNKNOWN"
        reasoning:             str   = ""
        kinematics_summary:    str   = ""
        electronic_id_summary: str   = ""
        admin_data_summary:    str   = ""
        spatial_zone:          str   = ""
        def model_dump(self):
            import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentAction:
        target_id:       str   = ""
        action:          str   = "SYS_TRACK_ONLY"
        justification:   str   = ""
        priority:        str   = "ROUTINE"
        env_action_type: str   = "hold_fire"
        def model_dump(self):
            import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class MultiAgentStep:
        episode_id:            str   = ""
        step_num:              int   = 0
        contact:               dict  = field(default_factory=dict)
        classification:        Optional[dict] = None
        action:                Optional[dict] = None
        env_reward:            float = 0.0
        classification_reward: float = 0.0
        action_reward:         float = 0.0
        total_reward:          float = 0.0
        def model_dump(self):
            import dataclasses; return dataclasses.asdict(self)
