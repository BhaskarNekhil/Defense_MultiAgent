# Defense-AI: Training a Two-Agent Qwen System for Air Defense Decision-Making with GRPO

> **Environment:** [Bhaskar111/defense-ai](https://huggingface.co/spaces/Bhaskar111/defense-ai)  
> **Training Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bhaskar111/defense-ai/blob/main/train_colab.ipynb)  
> **Models:** [Radar Adapter](https://huggingface.co/Bhaskar111/defense-rl-radar-adapter) · [Actor Adapter](https://huggingface.co/Bhaskar111/defense-rl-actor-adapter)

---

## Overview

Defense-AI is an **OpenEnv reinforcement learning environment** that places an AI agent in the role of an air defense commander. The agent receives a live synthetic radar picture of airborne contacts — enemy jets, inbound missiles, and friendly aircraft — and must:

1. **Scan** each contact to reveal its true identity via IFF (Identify Friend or Foe)
2. **Engage** enemy jets with machine gun fire
3. **Intercept** inbound missiles with anti-ballistic missiles (ABM)
4. **Hold fire** on friendly aircraft to avoid fratricide
5. **Submit** a tactical engagement report

The hard mode adds **IFF spoofing**: enemy jets transmit fake IFF codes to appear friendly. The agent must cross-reference radar kinematics (Mach speed, RCS, trajectory) to unmask them.

---

## The Two-Agent Architecture

Rather than a single monolithic policy, Defense-AI uses a **two-agent pipeline** inspired by real military command-and-control systems:

```
Radar Telemetry
      │
      ▼
┌─────────────────────────────┐
│  Agent 1: QwenRadarAgent    │  ← Radar Intelligence Analyst
│  Classifies into 7 classes: │
│  MISSILE_INBOUND            │
│  ENEMY_AIRCRAFT             │
│  FRIENDLY_AIRCRAFT          │
│  DOMESTIC_FLIGHT            │
│  FOREIGN_PERMITTED          │
│  FOREIGN_UNPERMITTED        │
│  OWN_ASSET                  │
└─────────────┬───────────────┘
              │ ContactClassification
              ▼
┌─────────────────────────────┐
│  Agent 2: QwenActorAgent    │  ← Tactical Commander
│  Selects tactical response: │
│  WEAPON_ABM_LAUNCH          │
│  WEAPON_ADS_ENGAGE          │
│  COMM_HELLO / COMM_WARN     │
│  SYS_TRACK_ONLY / NAV_GUIDE │
└─────────────┬───────────────┘
              │ Action
              ▼
     DefenseEnvironment
     (reward + observation)
```

Both agents are built on **Qwen/Qwen2.5-1.5B-Instruct** and fine-tuned with **GRPO (Group Relative Policy Optimization)**.

---

## Why GRPO?

GRPO (Group Relative Policy Optimization) is an RL algorithm designed specifically for language models. Unlike PPO which requires a separate critic network, GRPO:

- **Samples G completions** per prompt from the current policy
- **Computes relative rewards** within the group (no baseline network needed)
- **Updates the policy** to increase probability of high-reward completions
- **Reduces memory** by ~50% vs PPO (no critic model stored)

This makes it ideal for fine-tuning a 1.5B Qwen model on a single T4 GPU.

```python
# GRPO reward signal for Agent 1 (Radar Classification)
def classification_reward(predicted_class, true_class, confidence):
    if predicted_class == true_class:
        return 1.0 * confidence          # Correct + confident = full reward
    elif is_dangerous_mistake(predicted_class, true_class):
        return -1.0                       # Fratricide risk = max penalty
    else:
        return -0.3                       # Wrong but safe = small penalty
```

---

## Environment Design

### Three-Task Curriculum

| Task | Difficulty | Contacts | Key Challenge |
|------|-----------|----------|---------------|
| `task_easy` | Easy | 1–2 enemy jets | Basic IFF scan + engage |
| `task_medium` | Medium | 2–3 enemies + 1–2 friendlies + 1 missile | Friend/foe discrimination |
| `task_hard` | Hard | 3–5 enemies + 2–3 friendlies + 2–3 missiles | IFF spoofing detection |

### Reward Structure

The reward function is carefully designed to penalize catastrophic errors heavily:

```
scan_airspace           → +0.04  (information gathering)
engage_machine_gun ✓   → +0.15  (correct engagement)
engage_machine_gun ✗   → -0.50  (FRATRICIDE — severe penalty)
launch_abm ✓           → +0.20  (missile intercept)
hold_fire ✓            → +0.08  (friendly protection)
submit_report          → +0.40 to +0.25 (mission completion bonus)
```

The **solve threshold** is `0.80` — an agent must score above 80% to be considered mission-capable.

---

## Training Results

Training `Qwen/Qwen2.5-1.5B-Instruct` with GRPO for 3 epochs on 100 episodes across all three tasks:

### Before Training (Base Qwen, API mode)
| Task | Score | Solved |
|------|-------|--------|
| `task_easy` | 0.99 | ✓ |
| `task_medium` | 0.85 | ✓ |
| `task_hard` | 0.62 | ✗ |

### After GRPO Fine-Tuning
| Task | Score | Solved |
|------|-------|--------|
| `task_easy` | 0.99 | ✓ |
| `task_medium` | 0.93 | ✓ |
| `task_hard` | 0.84 | ✓ |

The most notable improvement is on **`task_hard`** — the GRPO-trained agent learned to detect IFF spoofers by cross-referencing radar kinematics, which the base model often failed to do consistently.

---

## Running the Environment

### Interactive Demo

Try the live demo at: **https://huggingface.co/spaces/Bhaskar111/defense-ai**

Switch to **AI Agent mode** to watch the Qwen pipeline play automatically.

### Training Your Own Agent

Use the Colab notebook to run GRPO fine-tuning on a free T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bhaskar111/defense-ai/blob/main/train_colab.ipynb)

```bash
# Or run locally with GPU
python train.py \
    --backend local \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 100 \
    --epochs 3 \
    --device cuda
```

### API Usage

```python
import httpx

# Reset environment
obs = httpx.post("https://bhaskar111-defense-ai.hf.space/reset",
                 json={"task_id": "task_easy"}).json()

# Step through episode
while not obs["done"]:
    action = your_agent.act(obs)
    obs = httpx.post("https://bhaskar111-defense-ai.hf.space/step",
                     json=action).json()
    print(f"Reward: {obs['reward']:.3f} | Score: {obs['partial_score']:.4f}")
```

---

## Key Takeaways

1. **Two-agent specialization works**: Separating radar classification from tactical decision-making allows each agent to develop expertise in its domain.

2. **GRPO is GPU-efficient**: Fine-tuning a 1.5B model with LoRA (rank 16) fits on a single T4 GPU with batch size 1.

3. **Curriculum matters**: Training on easy → medium → hard builds foundational skills before tackling IFF spoofing.

4. **Dense rewards accelerate learning**: Step-level rewards (not just episode-end) give the agent immediate signal about scan quality and weapon choice.

5. **Catastrophic penalties shape behavior**: The -0.50 fratricide penalty ensures the agent learns to always scan before engaging.

---

## Links

- 🎮 **Live Demo**: https://huggingface.co/spaces/Bhaskar111/defense-ai
- 📓 **Training Notebook**: [train_colab.ipynb](train_colab.ipynb)
- 🤖 **Radar Adapter**: https://huggingface.co/Bhaskar111/defense-rl-radar-adapter
- 🎯 **Actor Adapter**: https://huggingface.co/Bhaskar111/defense-rl-actor-adapter
- 📋 **OpenEnv Spec**: [openenv.yaml](openenv.yaml)
