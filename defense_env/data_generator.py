"""
Extended Synthetic Scenario Generator — Multi-Agent 7-Class Edition.

Generates randomised air-defense scenarios with rich telemetry:
  - velocity_mach      : float (Mach number)
  - rcs_m2             : float (Radar Cross-Section in m²)
  - trajectory_type    : "ballistic" | "corridor" | "direct_intercept" | "erratic" | "loiter"
  - zone               : "buffer" | "critical"
  - flight_plan_exists : bool
  - diplomatic_clearance: bool

Contact types (7-class taxonomy):
  MISSILE_INBOUND     — inbound ballistic/cruise missile
  ENEMY_AIRCRAFT      — confirmed hostile military jet
  FRIENDLY_AIRCRAFT   — allied military jet with valid IFF
  DOMESTIC_FLIGHT     — civilian aircraft, home country
  FOREIGN_PERMITTED   — foreign aircraft with approved flight plan
  FOREIGN_UNPERMITTED — foreign aircraft with no authorisation
  OWN_ASSET           — own missiles/drones already airborne
"""

import random
import uuid
from typing import Any, Dict, List


# ─── Aircraft / weapon model pools ────────────────────────────────────────────

ENEMY_JET_MODELS = [
    "MiG-29", "Su-27", "Su-35", "J-20", "F-7PG",
    "JF-17", "MiG-21", "Mirage-2000-EG", "F-16-Block-52",
]

FRIENDLY_JET_MODELS = [
    "Rafale", "Tejas-Mk1A", "Su-30MKI", "MiG-29K",
    "Mirage-2000H", "Jaguar-IS", "F/A-18E",
]

MISSILE_TYPES = [
    "Brahmos-NG", "BVR-R77", "R-60MK", "SD-10", "PL-15",
    "Kh-35", "Kh-59", "C-802", "P-270-Moskit",
]

CIVILIAN_AIRCRAFT_MODELS = [
    "Boeing-737", "Airbus-A320", "ATR-72", "Cessna-172", "Boeing-777",
    "Airbus-A380", "Embraer-E175", "Bombardier-CRJ900",
]

DRONE_MODELS = [
    "Heron-Mk1", "Rustom-2", "TAPAS-BH-201", "Predator-B",
    "Harop", "MALE-UAV", "Orbiter-3",
]

SQUAWK_CODES = ["7000", "7500", "7600", "7700", "2000", "1200", "6000"]
DOMESTIC_SQUAWK = ["7001", "7002", "7003", "7004", "7005"]

BEARING_RANGE = list(range(0, 360, 15))


# ─── ID generator ─────────────────────────────────────────────────────────────

def _random_id(prefix: str, n: int = 6) -> str:
    return f"{prefix}-{random.randint(10**(n-1), 10**n - 1)}"


# ─── Rich telemetry helpers ────────────────────────────────────────────────────

def _mach_to_label(mach: float) -> str:
    if mach < 0.8:
        return "subsonic"
    if mach < 1.2:
        return "transonic"
    if mach < 5.0:
        return "supersonic"
    return "hypersonic"


def _rcs_category(rcs: float) -> str:
    """Human-readable RCS category."""
    if rcs < 0.01:
        return "stealth"
    if rcs < 0.5:
        return "small"
    if rcs < 5.0:
        return "medium"
    return "large"


# ─── Contact builders ─────────────────────────────────────────────────────────

def _build_missile(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(2.5, 8.0), 2)
    rcs  = round(rng.uniform(0.005, 0.1), 4)
    return {
        "target_id":            _random_id("MSL"),
        "type":                 "missile",
        "contact_class":        "MISSILE_INBOUND",
        "affiliation":          "HOSTILE",
        "model":                rng.choice(MISSILE_TYPES),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             rng.choice(["low", "medium"]),
        "altitude_ft":          rng.randint(500, 25000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      rng.choice(["ballistic", "direct_intercept"]),
        # Electronic ID
        "iff_code":             None,
        "transponder":          None,
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   False,
        "diplomatic_clearance": False,
        # Spatial
        "zone":                 rng.choice(["buffer", "critical"]),
        "threat_level":         "CRITICAL",
        "time_to_impact_s":     rng.randint(20, 120),
        "scanned":              False,
    }


def _build_enemy_jet(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(1.2, 2.5), 2)
    rcs  = round(rng.uniform(0.5, 6.0), 3)
    return {
        "target_id":            _random_id("TGT"),
        "type":                 "fighter_jet",
        "contact_class":        "ENEMY_AIRCRAFT",
        "affiliation":          "ENEMY",
        "model":                rng.choice(ENEMY_JET_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             rng.choice(["low", "medium", "high"]),
        "altitude_ft":          rng.randint(10000, 50000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      rng.choice(["direct_intercept", "erratic"]),
        # Electronic ID
        "iff_code":             None,
        "transponder":          None,
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   False,
        "diplomatic_clearance": False,
        # Spatial
        "zone":                 rng.choice(["buffer", "critical"]),
        "threat_level":         rng.choice(["HIGH", "CRITICAL"]),
        "scanned":              False,
    }


def _build_friendly_jet(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(0.7, 1.8), 2)
    rcs  = round(rng.uniform(1.0, 8.0), 3)
    return {
        "target_id":            _random_id("TGT"),
        "type":                 "fighter_jet",
        "contact_class":        "FRIENDLY_AIRCRAFT",
        "affiliation":          "FRIENDLY",
        "model":                rng.choice(FRIENDLY_JET_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             rng.choice(["medium", "high"]),
        "altitude_ft":          rng.randint(20000, 45000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      rng.choice(["corridor", "loiter"]),
        # Electronic ID
        "iff_code":             _random_id("IFF", 4),
        "transponder":          rng.choice(DOMESTIC_SQUAWK),
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   True,
        "diplomatic_clearance": True,
        # Spatial
        "zone":                 rng.choice(["buffer", "critical"]),
        "threat_level":         "NONE",
        "scanned":              False,
    }


def _build_domestic_flight(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(0.6, 0.85), 2)
    rcs  = round(rng.uniform(20.0, 100.0), 1)
    return {
        "target_id":            _random_id("CVL"),
        "type":                 "civilian_aircraft",
        "contact_class":        "DOMESTIC_FLIGHT",
        "affiliation":          "CIVILIAN",
        "model":                rng.choice(CIVILIAN_AIRCRAFT_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             "high",
        "altitude_ft":          rng.randint(28000, 41000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      "corridor",
        # Electronic ID
        "iff_code":             _random_id("DOM", 4),
        "transponder":          rng.choice(SQUAWK_CODES),
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   True,
        "diplomatic_clearance": True,
        # Spatial
        "zone":                 "buffer",
        "threat_level":         "NONE",
        "scanned":              False,
    }


def _build_foreign_permitted(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(0.6, 0.85), 2)
    rcs  = round(rng.uniform(15.0, 80.0), 1)
    return {
        "target_id":            _random_id("FRN"),
        "type":                 "civilian_aircraft",
        "contact_class":        "FOREIGN_PERMITTED",
        "affiliation":          "FOREIGN",
        "model":                rng.choice(CIVILIAN_AIRCRAFT_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             "high",
        "altitude_ft":          rng.randint(30000, 43000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      "corridor",
        # Electronic ID
        "iff_code":             _random_id("FP", 4),
        "transponder":          rng.choice(SQUAWK_CODES),
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   True,
        "diplomatic_clearance": True,
        # Spatial
        "zone":                 "buffer",
        "threat_level":         "NONE",
        "scanned":              False,
    }


def _build_foreign_unpermitted(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(0.5, 1.1), 2)
    rcs  = round(rng.uniform(5.0, 60.0), 1)
    return {
        "target_id":            _random_id("UNP"),
        "type":                 "civilian_aircraft",
        "contact_class":        "FOREIGN_UNPERMITTED",
        "affiliation":          "FOREIGN",
        "model":                rng.choice(CIVILIAN_AIRCRAFT_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             rng.choice(["medium", "high"]),
        "altitude_ft":          rng.randint(15000, 40000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      rng.choice(["corridor", "erratic"]),
        # Electronic ID
        "iff_code":             None,
        "transponder":          rng.choice(SQUAWK_CODES),
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   False,
        "diplomatic_clearance": False,
        # Spatial
        "zone":                 rng.choice(["buffer", "critical"]),
        "threat_level":         "MEDIUM",
        "scanned":              False,
    }


def _build_own_asset(rng: random.Random) -> Dict[str, Any]:
    mach = round(rng.uniform(0.4, 3.5), 2)
    rcs  = round(rng.uniform(0.01, 2.0), 4)
    is_missile = rng.random() < 0.5
    return {
        "target_id":            _random_id("OWN"),
        "type":                 "own_missile" if is_missile else "own_drone",
        "contact_class":        "OWN_ASSET",
        "affiliation":          "OWN",
        "model":                rng.choice(MISSILE_TYPES if is_missile else DRONE_MODELS),
        # Kinematics
        "velocity_mach":        mach,
        "speed":                _mach_to_label(mach),
        "altitude":             rng.choice(["low", "medium", "high"]),
        "altitude_ft":          rng.randint(500, 35000),
        "bearing":              rng.choice(BEARING_RANGE),
        "trajectory_type":      rng.choice(["ballistic", "loiter", "corridor"]),
        # Electronic ID
        "iff_code":             _random_id("OWN", 4),
        "transponder":          rng.choice(DOMESTIC_SQUAWK),
        "rcs_m2":               rcs,
        "rcs_category":         _rcs_category(rcs),
        # Administrative
        "flight_plan_exists":   True,
        "diplomatic_clearance": True,
        # Spatial
        "zone":                 rng.choice(["buffer", "critical"]),
        "threat_level":         "NONE",
        "scanned":              False,
    }


def _build_iff_spoofer(rng: random.Random) -> Dict[str, Any]:
    """Enemy jet spoofing a valid IFF code (hard-mode deception)."""
    jet = _build_enemy_jet(rng)
    jet["iff_code"]   = _random_id("IFF", 4)   # fake IFF
    jet["spoofing"]   = True
    jet["contact_class"] = "ENEMY_AIRCRAFT"     # true class despite fake IFF
    return jet


# ─── Radar picture (pre-scan) ─────────────────────────────────────────────────

def _initial_radar_picture(contacts: List[Dict]) -> List[Dict]:
    """Return radar picture with affiliation and class hidden (pre-scan state)."""
    picture = []
    for c in contacts:
        picture.append({
            "target_id":    c["target_id"],
            "type":         c["type"],
            "affiliation":  "UNKNOWN",
            "contact_class": "UNKNOWN",
            # Kinematic data is visible on raw radar
            "velocity_mach": c["velocity_mach"],
            "speed":        c["speed"],
            "altitude":     c["altitude"],
            "altitude_ft":  c["altitude_ft"],
            "bearing":      c["bearing"],
            "trajectory_type": c["trajectory_type"],
            # Electronic signals partially visible
            "iff_code":     c.get("iff_code"),      # code visible, authenticity unknown
            "transponder":  c.get("transponder"),
            "rcs_m2":       c["rcs_m2"],
            "rcs_category": c["rcs_category"],
            # Administrative/spatial hidden until scan
            "flight_plan_exists":   None,
            "diplomatic_clearance": None,
            "zone":         c["zone"],
            "threat_level": "UNKNOWN",
            "scanned":      False,
        })
    return picture


# ─── Scenario builders ────────────────────────────────────────────────────────

def generate_scenario(task_id: str, seed: int) -> Dict[str, Any]:
    """Return a fully-specified scenario dict for the given task."""
    rng = random.Random(seed)
    builders = {
        "task_easy":   _easy_scenario,
        "task_medium": _medium_scenario,
        "task_hard":   _hard_scenario,
        "task_full":   _full_scenario,   # all 7 classes
    }
    builder = builders.get(task_id, _easy_scenario)
    return builder(rng)


def _easy_scenario(rng: random.Random) -> Dict[str, Any]:
    """Easy: 1-2 enemy jets only. Agent must scan → engage."""
    n_enemies = rng.randint(1, 2)
    contacts  = [_build_enemy_jet(rng) for _ in range(n_enemies)]
    return _pack_scenario(contacts, task_type="easy")


def _medium_scenario(rng: random.Random) -> Dict[str, Any]:
    """Medium: enemy jets + friendlies + 1 missile."""
    enemies   = [_build_enemy_jet(rng)    for _ in range(rng.randint(2, 3))]
    friendlies = [_build_friendly_jet(rng) for _ in range(rng.randint(1, 2))]
    missiles  = [_build_missile(rng)      for _ in range(1)]
    contacts  = enemies + friendlies + missiles
    rng.shuffle(contacts)
    return _pack_scenario(contacts, task_type="medium")


def _hard_scenario(rng: random.Random) -> Dict[str, Any]:
    """Hard: saturation + IFF spoofing."""
    n_enemies = rng.randint(3, 5)
    enemies   = [_build_enemy_jet(rng)    for _ in range(n_enemies)]
    friendlies = [_build_friendly_jet(rng) for _ in range(rng.randint(2, 3))]
    missiles  = [_build_missile(rng)      for _ in range(rng.randint(2, 3))]

    # Inject IFF spoofers
    n_spoofers = rng.randint(1, min(2, n_enemies))
    spoofer_ids = []
    for i in range(n_spoofers):
        enemies[i]["iff_code"] = _random_id("IFF", 4)
        enemies[i]["spoofing"] = True
        spoofer_ids.append(enemies[i]["target_id"])

    contacts = enemies + friendlies + missiles
    rng.shuffle(contacts)
    scenario = _pack_scenario(contacts, task_type="hard")
    scenario["iff_spoofers"] = spoofer_ids
    return scenario


def _full_scenario(rng: random.Random) -> Dict[str, Any]:
    """Full 7-class scenario: all contact types present."""
    contacts = [
        _build_missile(rng),
        _build_enemy_jet(rng),
        _build_enemy_jet(rng),
        _build_friendly_jet(rng),
        _build_domestic_flight(rng),
        _build_foreign_permitted(rng),
        _build_foreign_unpermitted(rng),
        _build_own_asset(rng),
    ]
    rng.shuffle(contacts)
    return _pack_scenario(contacts, task_type="full")


def _pack_scenario(contacts: List[Dict], task_type: str) -> Dict[str, Any]:
    """Package contacts into a scenario dict."""
    contact_map = {c["target_id"]: c for c in contacts}

    enemy_jets   = [c["target_id"] for c in contacts if c["contact_class"] == "ENEMY_AIRCRAFT"]
    friendly_jets = [c["target_id"] for c in contacts if c["contact_class"] == "FRIENDLY_AIRCRAFT"]
    missiles     = [c["target_id"] for c in contacts if c["contact_class"] == "MISSILE_INBOUND"]
    domestic     = [c["target_id"] for c in contacts if c["contact_class"] == "DOMESTIC_FLIGHT"]
    foreign_perm = [c["target_id"] for c in contacts if c["contact_class"] == "FOREIGN_PERMITTED"]
    foreign_unp  = [c["target_id"] for c in contacts if c["contact_class"] == "FOREIGN_UNPERMITTED"]
    own_assets   = [c["target_id"] for c in contacts if c["contact_class"] == "OWN_ASSET"]

    # Ground truth class map (used by grader)
    ground_truth = {c["target_id"]: c["contact_class"] for c in contacts}

    return {
        "mission_id":       str(uuid.uuid4()),
        "task_type":        task_type,
        "contacts":         contact_map,
        "threats_in_scope": list(contact_map.keys()),
        # Legacy keys (backward compat)
        "enemy_jets":       enemy_jets,
        "friendly_jets":    friendly_jets,
        "missiles":         missiles,
        # New class lists
        "domestic_flights":    domestic,
        "foreign_permitted":   foreign_perm,
        "foreign_unpermitted": foreign_unp,
        "own_assets":          own_assets,
        # Ground truth for grader
        "ground_truth_classes": ground_truth,
        "iff_spoofers":         [],
        "initial_radar":        _initial_radar_picture(contacts),
        "alerts":               _generate_alerts(contacts, task_type),
    }


def _generate_alerts(contacts: List[Dict], task_type: str) -> List[Dict]:
    classes = [c["contact_class"] for c in contacts]
    n_missiles = classes.count("MISSILE_INBOUND")
    n_enemy    = classes.count("ENEMY_AIRCRAFT")
    n_unknown  = sum(1 for c in contacts if c.get("contact_class") == "FOREIGN_UNPERMITTED")

    alerts = []
    if n_missiles:
        alerts.append({
            "alert_id":  str(uuid.uuid4()),
            "type":      "MISSILE_ALERT",
            "message":   (
                f"⚠ MISSILE WARNING: {n_missiles} high-speed ballistic object(s) detected. "
                "No IFF. Time-critical. Launch ABM immediately after confirmation."
            ),
            "timestamp": "T+000",
        })
    if n_enemy:
        alerts.append({
            "alert_id":  str(uuid.uuid4()),
            "type":      "AIR_INTRUSION",
            "message":   (
                f"{n_enemy} unidentified military-profile aircraft. "
                "No valid IFF transponder. Scan to confirm hostility."
            ),
            "timestamp": "T+000",
        })
    if n_unknown:
        alerts.append({
            "alert_id":  str(uuid.uuid4()),
            "type":      "AIRSPACE_VIOLATION",
            "message":   (
                f"{n_unknown} foreign aircraft with no flight plan or clearance. "
                "Transmit warning on emergency frequency."
            ),
            "timestamp": "T+001",
        })
    if task_type == "hard":
        alerts.append({
            "alert_id":  str(uuid.uuid4()),
            "type":      "SIGINT_WARNING",
            "message":   (
                "SIGINT: Enemy aircraft may be spoofing IFF codes. "
                "Cross-reference speed, altitude, RCS, and trajectory. "
                "Do NOT rely on IFF alone."
            ),
            "timestamp": "T+000",
        })
    return alerts
