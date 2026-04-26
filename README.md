---
title: Defense-RL
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - defense
  - agentic
---

# Defense-AI — Air Defense OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Space-yellow)](https://huggingface.co/spaces)

A fully-featured OpenEnv reinforcement learning environment where an AI agent
acts as an **air defense commander** making real-time threat discrimination and
engagement decisions.

---

## 🎯 Environment Description

The agent receives a synthetic radar picture with three types of airborne
contacts and must correctly classify and respond to each:

| Contact Type      | Correct Action         | Reward (step) | Reward (report) |
|-------------------|------------------------|---------------|-----------------|
| Enemy fighter jet | `engage_machine_gun`   | +0.15         | +0.35–0.55      |
| Friendly aircraft | `hold_fire`            | +0.08         | +0.10–0.20      |
| Inbound missile   | `launch_abm`           | +0.20         | +0.25–0.30      |

**Key mechanic**: All contacts start as `UNKNOWN`. The agent must
`scan_airspace` to reveal true affiliation before acting. Engaging a friendly
aircraft (fratricide) incurs a severe penalty.

**Hard mode twist**: Enemy jets may spoof IFF transponder codes to appear
friendly. The agent must cross-reference speed, altitude, and bearing
to unmask spoofers.

---

## 📁 Project Structure

```
defense-ai/
├── inference.py            # ← Baseline inference script (REQUIRED)
├── Dockerfile              # ← Container spec for HF Spaces
├── requirements.txt
├── openenv.yaml            # ← OpenEnv spec manifest
├── models.py               # DefenseAction, DefenseObservation, DefenseState
├── agent.py                # Rule-based reference agent
├── defense_env/
│   ├── __init__.py
│   ├── environment.py      # reset() / step() / state — core logic
│   ├── tasks.py            # 3 tasks + deterministic graders
│   ├── reward.py           # Step & report reward calculators
│   └── data_generator.py   # Synthetic scenario generation
└── server/
    ├── app.py              # FastAPI REST server
    ├── defense_environment.py
    └── static/
        └── index.html      # Interactive tactical UI
```

---

## 🕹 Tasks

### Task Easy — `task_easy` (10 steps)
**Difficulty**: Easy  
1–2 enemy fighter jets with no IFF transponder detected.
Scan contacts to confirm hostile, then engage with machine gun.
No friendlies, no missiles. Baseline pass rate: ~95%.

**Grader scoring**:
- 0.55 — all enemy jets engaged with `engage_machine_gun`
- 0.15 — zero fratricide
- 0.15 — correct weapon (machine gun, not ABM)
- 0.15 — tactical report quality

### Task Medium — `task_medium` (18 steps)
**Difficulty**: Medium  
Mixed air picture: 2–3 enemy jets + 1–2 friendly aircraft + 1 inbound missile.
Must discriminate friend/foe via scan, use correct weapon per type.
Fratricide = mission failure.

**Grader scoring**:
- 0.35 — enemy jets neutralised
- 0.25 — missiles intercepted with ABM
- 0.10 — friendlies protected (zero fratricide)
- 0.15 — correct weapon per target type
- 0.15 — tactical report quality

### Task Hard — `task_hard` (25 steps)
**Difficulty**: Hard  
Saturation attack: 3–5 enemy jets + 2–3 friendlies + 2–3 missiles.
1–2 enemy jets **spoof IFF codes** to appear friendly.
Must unmask spoofers by cross-referencing IFF with speed/altitude/bearing.

**Grader scoring**:
- 0.30 — all enemy jets (including spoofers) neutralised
- 0.25 — all missiles intercepted with ABM
- 0.12 — zero fratricide
- 0.08 — spoofers correctly identified and engaged
- 0.10 — correct weapon per target type
- 0.15 — tactical report (with spoofing mentioned)

---

## 🎁 Reward Design

### Step Rewards (dense — every action)
```
scan_airspace (any contact)     → +0.04 per scan  (-0.02 if already scanned)
engage_machine_gun (enemy jet)  → +0.15  ✓
engage_machine_gun (friendly)   → -0.50  ← FRATRICIDE
engage_machine_gun (missile)    → -0.05  (wrong tool — ineffective)
launch_abm (missile)            → +0.20  ✓
launch_abm (enemy jet)          → +0.05  (works but wasteful)
launch_abm (friendly)           → -0.50  ← FRATRICIDE
hold_fire (friendly)            → +0.08  ✓
hold_fire (enemy)               → -0.10  (dereliction of duty)
step_penalty (always)           → -0.02  (efficiency pressure)
```

### Report Rewards (episode end — submit_report)
```
all_enemies_neutralised        → +0.40
all_missiles_intercepted       → +0.30
zero_fratricide                → +0.20
good_tactical_summary          → +0.15
solve_bonus (score ≥ 0.80)     → +0.25
missed_enemy (per target)      → -0.15
missed_missile (per target)    → -0.25
fratricide (per friendly hit)  → -0.40
```

**Reward range**: `(-0.99, 0.99)` clamped.  
**Score range**: `(0.01, 0.99)` strictly open.  
**Solve threshold**: 0.80

---

## 📌 Action Space

| `action_type`        | `target_id` | Description |
|----------------------|-------------|-------------|
| `scan_airspace`      | required    | Reveal true affiliation of a radar contact |
| `engage_machine_gun` | required    | Fire on a target (correct for enemy jets) |
| `launch_abm`         | required    | Launch anti-ballistic missile (correct for missiles) |
| `hold_fire`          | required    | Explicitly hold fire (correct for friendly aircraft) |
| `submit_report`      | —           | End episode; provide `engagements` list + `decisions_summary` |

### submit_report action body
```json
{
  "action_type": "submit_report",
  "engagements": [
    {"target_id": "TGT-123456", "action": "engage_machine_gun"},
    {"target_id": "MSL-789012", "action": "launch_abm"},
    {"target_id": "TGT-345678", "action": "hold_fire"}
  ],
  "decisions_summary": "Scanned all contacts via IFF discrimination. Engaged 2 enemy jets with machine gun. Intercepted 1 missile with ABM. Held fire on 1 friendly."
}
```

---

## 📡 Observation Space

```json
{
  "mission_id":       "uuid",
  "task_id":          "task_easy | task_medium | task_hard",
  "difficulty":       "easy | medium | hard",
  "task_description": "string",
  "radar_contacts": [
    {
      "target_id":    "TGT-123456",
      "type":         "fighter_jet | missile",
      "affiliation":  "UNKNOWN | ENEMY | FRIENDLY | HOSTILE",
      "altitude":     "low | medium | high",
      "speed":        "subsonic | transonic | supersonic | hypersonic",
      "bearing":      0,
      "iff_code":     "IFF-1234 | null",
      "threat_level": "UNKNOWN | NONE | HIGH | CRITICAL",
      "scanned":      false
    }
  ],
  "threats_in_scope": ["TGT-123456", "MSL-789012"],
  "action_result":    "string | null",
  "action_error":     "string | null",
  "engaged_targets":  ["TGT-123456"],
  "held_targets":     ["TGT-345678"],
  "partial_score":    0.0,
  "feedback":         "string",
  "reward":           0.13,
  "done":             false,
  "steps_taken":      3,
  "max_steps":        10
}
```

---

## 🚀 Setup & Usage

### Quick Start (local)
```bash
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Open interactive UI
open http://localhost:7860
```

### Docker
```bash
docker build -t defense-ai .
docker run -p 7860:7860 defense-ai
```

### Run Baseline Inference
```bash
# Rule-based agent (no API needed)
MODEL_NAME=rule-agent \
HF_TOKEN=no-key \
API_BASE_URL=https://api.openai.com/v1 \
SERVER_URL=http://localhost:7860 \
python inference.py

# LLM agent
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=your-openai-key \
SERVER_URL=http://localhost:7860 \
python inference.py
```

### API Endpoints

| Endpoint  | Method | Description |
|-----------|--------|-------------|
| `/reset`  | POST   | Start new episode (`{"task_id": "task_easy"}`) |
| `/step`   | POST   | Take action (`{"action_type": "...", "target_id": "..."}`) |
| `/state`  | GET    | Episode metadata |
| `/health` | GET    | Liveness probe |
| `/docs`   | GET    | Swagger UI |

---

## 📊 Baseline Scores (Rule-Based Agent)

| Task          | Score  | Solved? |
|---------------|--------|---------|
| `task_easy`   | ~0.99  | ✓ Yes   |
| `task_medium` | ~0.99  | ✓ Yes   |
| `task_hard`   | ~0.99  | ✓ Yes   |
| **Mean**      | ~0.99  |         |

The rule-based agent achieves near-perfect scores by always scanning before
engaging and submitting a complete report. An LLM agent without explicit
scan-first logic may make mistakes on medium/hard tasks, especially around
IFF spoofing discrimination and correct weapon selection.

---

## ⚙️ OpenEnv Compliance

- ✅ `reset()` → `DefenseObservation` (Pydantic model)
- ✅ `step(action)` → `DefenseObservation` with reward + done
- ✅ `state` property → `DefenseState`
- ✅ Reward range: `(-0.99, +0.99)` clamped
- ✅ Score range: `(0.01, 0.99)` strictly open
- ✅ 3 tasks with deterministic graders
- ✅ Difficulty progression: easy → medium → hard
- ✅ `openenv.yaml` manifest
- ✅ Dockerfile for containerised deployment
- ✅ `inference.py` baseline script

---

## 🧠 Why This Environment Matters

Air defense decision-making is a **genuine real-world task** where:
- Millisecond identification mistakes cause catastrophic fratricide
- IFF spoofing is a documented real adversarial tactic
- Multi-modal sensor fusion (radar + IFF + kinematics) is required
- Correct weapon selection is operationally critical

This environment tests an agent's ability to reason under uncertainty,
prioritise time-critical threats, avoid catastrophic errors, and produce
coherent tactical documentation — all skills that transfer to high-stakes
agentic deployments.