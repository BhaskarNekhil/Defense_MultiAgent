"""
Agent 2 — QwenActorAgent: Selects tactical response from 6 action categories.

Receives ContactClassification from Agent 1 → outputs one of:
  COMM_HELLO, COMM_WARN, NAV_GUIDE, SYS_TRACK_ONLY,
  WEAPON_ABM_LAUNCH, WEAPON_ADS_ENGAGE

Maps to DefenseEnvironment action types for env.step().
"""

import json, re
from typing import Any, Dict, Optional

ACTOR_SYSTEM_PROMPT = """You are Agent 2 of a two-agent air defense system — TACTICAL COMMANDER.

Agent 1 has classified a radar contact. Based on that classification, you must select ONE tactical action:

| Action             | When to use                                                              |
|--------------------|--------------------------------------------------------------------------|
| COMM_HELLO         | FOREIGN_PERMITTED, DOMESTIC_FLIGHT — acknowledge and welcome             |
| COMM_WARN          | FOREIGN_UNPERMITTED, borderline ENEMY — formal warning on emergency freq |
| NAV_GUIDE          | Needs landing/vectoring: DOMESTIC_FLIGHT, FOREIGN_PERMITTED in distress  |
| SYS_TRACK_ONLY     | FRIENDLY_AIRCRAFT, OWN_ASSET, or non-threat needing monitoring only      |
| WEAPON_ABM_LAUNCH  | MISSILE_INBOUND — fire interceptor missiles immediately                  |
| WEAPON_ADS_ENGAGE  | ENEMY_AIRCRAFT in critical zone — activate ADS guns / CIWS               |

DECISION RULES:
- MISSILE_INBOUND in ANY zone → WEAPON_ABM_LAUNCH (time critical, no delay)
- ENEMY_AIRCRAFT critical zone → WEAPON_ADS_ENGAGE
- ENEMY_AIRCRAFT buffer zone → COMM_WARN first (escalation ladder)
- FRIENDLY_AIRCRAFT / OWN_ASSET → SYS_TRACK_ONLY (never fire)
- DOMESTIC_FLIGHT → SYS_TRACK_ONLY or COMM_HELLO
- FOREIGN_PERMITTED → COMM_HELLO (cleared, welcome them)
- FOREIGN_UNPERMITTED → COMM_WARN (warn before escalation)

PRIORITY: IMMEDIATE > HIGH > ROUTINE > MONITOR

OUTPUT — strict JSON only:
{"action":"<action>","justification":"<1-2 sentences>","priority":"<IMMEDIATE|HIGH|ROUTINE|MONITOR>"}"""

ACTOR_USER_TEMPLATE = """Contact classification from Agent 1:
  Target ID:     {target_id}
  Class:         {contact_class}
  Confidence:    {confidence}
  Threat level:  {threat_level}
  Spatial zone:  {spatial_zone}
  Kinematics:    {kinematics_summary}
  Electronic ID: {electronic_id_summary}
  Admin data:    {admin_data_summary}
  Reasoning:     {reasoning}

Select the appropriate tactical action. Output JSON:"""

# ── Default action map (rule-based fallback) ──────────────────────────────────
DEFAULT_ACTION_MAP = {
    "MISSILE_INBOUND":      ("WEAPON_ABM_LAUNCH",  "IMMEDIATE"),
    "ENEMY_AIRCRAFT":       ("WEAPON_ADS_ENGAGE",  "IMMEDIATE"),
    "FRIENDLY_AIRCRAFT":    ("SYS_TRACK_ONLY",     "MONITOR"),
    "DOMESTIC_FLIGHT":      ("SYS_TRACK_ONLY",     "MONITOR"),
    "FOREIGN_PERMITTED":    ("COMM_HELLO",          "ROUTINE"),
    "FOREIGN_UNPERMITTED":  ("COMM_WARN",           "HIGH"),
    "OWN_ASSET":            ("SYS_TRACK_ONLY",      "MONITOR"),
    "UNKNOWN":              ("SYS_TRACK_ONLY",      "HIGH"),
}

# ── Environment action type mapping ──────────────────────────────────────────
ACTION_TO_ENV = {
    "COMM_HELLO":         "hold_fire",
    "COMM_WARN":          "hold_fire",
    "NAV_GUIDE":          "hold_fire",
    "SYS_TRACK_ONLY":     "hold_fire",
    "WEAPON_ABM_LAUNCH":  "launch_abm",
    "WEAPON_ADS_ENGAGE":  "engage_machine_gun",
}

VALID_ACTIONS = set(ACTION_TO_ENV.keys())


def _rule_decide(classification: Dict) -> Dict:
    cc    = classification.get("contact_class", "UNKNOWN")
    zone  = classification.get("spatial_zone", "BUFFER")
    # Zone-sensitive enemy decision
    if cc == "ENEMY_AIRCRAFT" and zone == "BUFFER":
        action, priority = "COMM_WARN", "HIGH"
    else:
        action, priority = DEFAULT_ACTION_MAP.get(cc, ("SYS_TRACK_ONLY", "HIGH"))
    return {
        "action": action,
        "justification": f"Rule-based: class={cc}, zone={zone}.",
        "priority": priority,
    }


def _extract_json(text: str) -> Optional[Dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None


class QwenActorAgent:
    """
    Agent 2: Tactical Commander.
    Selects the appropriate action for a classified contact.

    Args:
        backend:     "local" | "api" | "rule"
        model_name:  HuggingFace model ID or API model name
        shared_pipeline: pass a pre-loaded pipeline to reuse weights (optional)
    """

    def __init__(self, backend="rule", model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 api_base="https://api.openai.com/v1", api_key="no-key",
                 device="auto", max_tokens=256, temperature=0.1,
                 shared_pipeline=None,
                 adapter_repo: str = None, adapter_subfolder: str = "epoch_3"):
        self.backend = backend
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.adapter_repo = adapter_repo          # e.g. "Bhaskar111/defense-rl-actor-adapter"
        self.adapter_subfolder = adapter_subfolder  # e.g. "epoch_3"
        self._pipeline = shared_pipeline  # may share with RadarAgent
        self._client = None

        if backend == "local" and shared_pipeline is None:
            self._load_local()
        elif backend == "api":
            self._init_api()

    def _load_local(self):
        print(f"[ActorAgent] Loading {self.model_name} locally...")
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "auto" else self.device,
                trust_remote_code=True,
            )
            # ── Apply LoRA adapter if provided ───────────────────────────────
            if self.adapter_repo:
                try:
                    from peft import PeftModel
                    print(
                        f"[ActorAgent] Applying LoRA adapter: {self.adapter_repo} "
                        f"(subfolder={self.adapter_subfolder})"
                    )
                    model = PeftModel.from_pretrained(
                        model,
                        self.adapter_repo,
                        subfolder=self.adapter_subfolder,
                        is_trainable=False,
                    )
                    model = model.merge_and_unload()   # merge weights for faster inference
                    print("[ActorAgent] LoRA adapter merged successfully.")
                except Exception as ae:
                    print(f"[ActorAgent] Adapter load failed ({ae}). Using base model.")

            self._pipeline = pipeline("text-generation", model=model, tokenizer=tok,
                                       max_new_tokens=self.max_tokens, temperature=self.temperature,
                                       do_sample=self.temperature > 0)
            print("[ActorAgent] Model ready.")
        except Exception as e:
            print(f"[ActorAgent] Load failed ({e}). Falling back to rule-based.")
            self.backend = "rule"

    def _init_api(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            print(f"[ActorAgent] API ready → {self.api_base}")
        except Exception as e:
            print(f"[ActorAgent] API init failed ({e}). Falling back to rule-based.")
            self.backend = "rule"

    def decide(self, classification: Dict) -> Dict:
        """
        Given Agent 1's classification, decide tactical action.

        Args:
            classification: dict from QwenRadarAgent.classify()

        Returns:
            dict with keys: action, justification, priority, env_action_type, target_id
        """
        if self.backend == "rule":
            raw = _rule_decide(classification)
        elif self.backend == "api":
            raw = self._api_infer(classification)
        elif self.backend == "local":
            raw = self._local_infer(classification)
        else:
            raw = _rule_decide(classification)

        raw = self._validate(raw)
        raw["env_action_type"] = ACTION_TO_ENV.get(raw["action"], "hold_fire")
        raw["target_id"] = classification.get("target_id", "UNKNOWN")
        return raw

    def _fmt_user(self, c: Dict) -> str:
        return ACTOR_USER_TEMPLATE.format(
            target_id          = c.get("target_id", "?"),
            contact_class      = c.get("contact_class", "UNKNOWN"),
            confidence         = c.get("confidence", "?"),
            threat_level       = c.get("threat_level", "?"),
            spatial_zone       = c.get("spatial_zone", "BUFFER"),
            kinematics_summary = c.get("kinematics_summary", "N/A"),
            electronic_id_summary = c.get("electronic_id_summary", "N/A"),
            admin_data_summary = c.get("admin_data_summary", "N/A"),
            reasoning          = c.get("reasoning", "N/A"),
        )

    def _local_infer(self, classification: Dict) -> Dict:
        if not self._pipeline: return _rule_decide(classification)
        try:
            msgs = [{"role":"system","content":ACTOR_SYSTEM_PROMPT},
                    {"role":"user","content":self._fmt_user(classification)}]
            if hasattr(self._pipeline.tokenizer, "apply_chat_template"):
                prompt = self._pipeline.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"<|system|>\n{ACTOR_SYSTEM_PROMPT}\n<|user|>\n{self._fmt_user(classification)}\n<|assistant|>\n"
            out = self._pipeline(prompt, return_full_text=False)
            res = _extract_json(out[0]["generated_text"])
            if res: return res
        except Exception as e:
            print(f"[ActorAgent] Local infer error: {e}")
        return _rule_decide(classification)

    def _api_infer(self, classification: Dict) -> Dict:
        if not self._client: return _rule_decide(classification)
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role":"system","content":ACTOR_SYSTEM_PROMPT},
                          {"role":"user","content":self._fmt_user(classification)}])
            res = _extract_json(resp.choices[0].message.content or "")
            if res: return res
        except Exception as e:
            print(f"[ActorAgent] API error: {e}")
        return _rule_decide(classification)

    @staticmethod
    def _validate(r: Dict) -> Dict:
        if r.get("action") not in VALID_ACTIONS:
            r["action"] = "SYS_TRACK_ONLY"
        r.setdefault("justification", "")
        r.setdefault("priority", "ROUTINE")
        valid_priorities = {"IMMEDIATE","HIGH","ROUTINE","MONITOR"}
        if r["priority"] not in valid_priorities:
            r["priority"] = "ROUTINE"
        return r
