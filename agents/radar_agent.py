"""
Agent 1 — QwenRadarAgent: Classifies radar contacts into 7 categories.

Backends: "local" (transformers), "api" (OpenAI-compatible), "rule" (deterministic)
"""

import json, os, re
from typing import Any, Dict, Optional

RADAR_SYSTEM_PROMPT = """You are Agent 1 of a two-agent air defense system — RADAR INTELLIGENCE ANALYST.

Classify every radar contact into ONE of 7 categories:
  MISSILE_INBOUND     — High-speed ballistic/cruise missile, no transponder
  ENEMY_AIRCRAFT      — Hostile military jet, no valid IFF
  FRIENDLY_AIRCRAFT   — Allied military jet with valid encrypted IFF
  DOMESTIC_FLIGHT     — Commercial/civilian aircraft in home country
  FOREIGN_PERMITTED   — Foreign aircraft with approved flight plan
  FOREIGN_UNPERMITTED — Foreign aircraft with no authorisation
  OWN_ASSET           — Own missiles/drones already airborne

REASONING — work through ALL FOUR axes:
A. KINEMATICS: Mach>2.5=missile; 1.2-2.5=military jet; <0.9=civilian. Trajectory "ballistic"/"direct_intercept"=threat.
B. ELECTRONIC ID: IFF None + military profile = hostile. RCS >20m²=large commercial, 0.5-8=military jet, <0.5=missile/stealth.
C. ADMINISTRATIVE: flight_plan_exists + diplomatic_clearance = True → civilian/permitted. Both False + no IFF = threat.
D. SPATIAL ZONE: "critical" zone = immediate action required.

OUTPUT — strict JSON only:
{"contact_class":"<class>","confidence":<0-1>,"threat_level":"<NONE|LOW|MEDIUM|HIGH|CRITICAL>",
 "kinematics_summary":"<sentence>","electronic_id_summary":"<sentence>",
 "admin_data_summary":"<sentence>","spatial_zone":"<BUFFER|CRITICAL>","reasoning":"<2-4 sentences>"}"""

RADAR_USER_TEMPLATE = """Classify this contact:
ID: {target_id} | Type: {type}
A. KINEMATICS: Mach {velocity_mach} ({speed}), Alt {altitude} ({altitude_ft}ft), Traj: {trajectory_type}, Bearing: {bearing}°
B. ELECTRONIC: IFF={iff_code}, Transponder={transponder}, RCS={rcs_m2}m² ({rcs_category})
C. ADMIN: FlightPlan={flight_plan_exists}, DiploClearance={diplomatic_clearance}
D. ZONE: {zone}
Reason A→B→C→D then output JSON:"""


def _rule_classify(contact: Dict) -> Dict:
    ctype = contact.get("type", "")
    mach  = float(contact.get("velocity_mach", 0.8))
    traj  = contact.get("trajectory_type", "corridor")
    iff   = contact.get("iff_code")
    fp    = contact.get("flight_plan_exists", False)
    diplo = contact.get("diplomatic_clearance", False)
    affil = contact.get("affiliation", "UNKNOWN")
    zone  = contact.get("zone", "buffer")

    def _r(cls, conf, threat):
        return {"contact_class": cls, "confidence": conf, "threat_level": threat,
                "kinematics_summary": f"Mach {mach}, traj={traj}.",
                "electronic_id_summary": f"IFF={iff}, RCS={contact.get('rcs_m2','?')}m².",
                "admin_data_summary": f"FP={fp}, clearance={diplo}.",
                "spatial_zone": zone.upper(), "reasoning": f"Rule: type={ctype},mach={mach},iff={iff},fp={fp}."}

    if ctype in ("own_missile","own_drone") or affil == "OWN":
        return _r("OWN_ASSET", 0.95, "NONE")
    if ctype == "missile" or (mach > 2.0 and traj in ("ballistic","direct_intercept") and not iff):
        return _r("MISSILE_INBOUND", 0.92, "CRITICAL")
    if ctype == "fighter_jet":
        if affil == "FRIENDLY" or (iff and fp):
            return _r("FRIENDLY_AIRCRAFT", 0.88, "NONE")
        return _r("ENEMY_AIRCRAFT", 0.85, "HIGH")
    if ctype == "civilian_aircraft":
        if fp and diplo:
            return _r("DOMESTIC_FLIGHT" if affil == "CIVILIAN" else "FOREIGN_PERMITTED", 0.90, "NONE")
        return _r("FOREIGN_UNPERMITTED", 0.75, "MEDIUM")
    return _r("UNKNOWN", 0.30, "UNKNOWN")


def _extract_json(text: str) -> Optional[Dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None


class QwenRadarAgent:
    """Agent 1: Classifies radar contacts using Qwen chain-of-thought."""

    def __init__(self, backend="rule", model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 api_base="https://api.openai.com/v1", api_key="no-key",
                 device="auto", max_tokens=512, temperature=0.1):
        self.backend = backend
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._pipeline = None
        self._client = None
        if backend == "local":
            self._load_local()
        elif backend == "api":
            self._init_api()

    def _load_local(self):
        print(f"[RadarAgent] Loading {self.model_name} locally...")
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
            self._pipeline = pipeline("text-generation", model=model, tokenizer=tok,
                                       max_new_tokens=self.max_tokens, temperature=self.temperature,
                                       do_sample=self.temperature > 0)
            print("[RadarAgent] Model ready.")
        except Exception as e:
            print(f"[RadarAgent] Load failed ({e}). Falling back to rule-based.")
            self.backend = "rule"

    def _init_api(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            print(f"[RadarAgent] API ready → {self.api_base}")
        except Exception as e:
            print(f"[RadarAgent] API init failed ({e}). Falling back to rule-based.")
            self.backend = "rule"

    def classify(self, contact: Dict) -> Dict:
        """Classify a single radar contact. Returns classification dict."""
        if self.backend == "rule":
            result = _rule_classify(contact)
        elif self.backend == "api":
            result = self._api_infer(contact)
        elif self.backend == "local":
            result = self._local_infer(contact)
        else:
            result = _rule_classify(contact)
        result["target_id"] = contact.get("target_id", "UNKNOWN")
        return self._validate(result)

    def classify_batch(self, contacts: list) -> list:
        return [self.classify(c) for c in contacts]

    def _fmt_user(self, c: Dict) -> str:
        return RADAR_USER_TEMPLATE.format(
            target_id=c.get("target_id","?"), type=c.get("type","?"),
            velocity_mach=c.get("velocity_mach","?"), speed=c.get("speed","?"),
            altitude=c.get("altitude","?"), altitude_ft=c.get("altitude_ft","?"),
            trajectory_type=c.get("trajectory_type","?"), bearing=c.get("bearing","?"),
            iff_code=c.get("iff_code","None"), transponder=c.get("transponder","None"),
            rcs_m2=c.get("rcs_m2","?"), rcs_category=c.get("rcs_category","?"),
            flight_plan_exists=c.get("flight_plan_exists","?"),
            diplomatic_clearance=c.get("diplomatic_clearance","?"),
            zone=c.get("zone","buffer"))

    def _local_infer(self, contact: Dict) -> Dict:
        if not self._pipeline: return _rule_classify(contact)
        try:
            msgs = [{"role":"system","content":RADAR_SYSTEM_PROMPT},
                    {"role":"user","content":self._fmt_user(contact)}]
            if hasattr(self._pipeline.tokenizer, "apply_chat_template"):
                prompt = self._pipeline.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"<|system|>\n{RADAR_SYSTEM_PROMPT}\n<|user|>\n{self._fmt_user(contact)}\n<|assistant|>\n"
            out = self._pipeline(prompt, return_full_text=False)
            res = _extract_json(out[0]["generated_text"])
            if res: return res
        except Exception as e:
            print(f"[RadarAgent] Local infer error: {e}")
        return _rule_classify(contact)

    def _api_infer(self, contact: Dict) -> Dict:
        if not self._client: return _rule_classify(contact)
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role":"system","content":RADAR_SYSTEM_PROMPT},
                          {"role":"user","content":self._fmt_user(contact)}])
            res = _extract_json(resp.choices[0].message.content or "")
            if res: return res
        except Exception as e:
            print(f"[RadarAgent] API error: {e}")
        return _rule_classify(contact)

    @staticmethod
    def _validate(r: Dict) -> Dict:
        valid = {"MISSILE_INBOUND","ENEMY_AIRCRAFT","FRIENDLY_AIRCRAFT","DOMESTIC_FLIGHT",
                 "FOREIGN_PERMITTED","FOREIGN_UNPERMITTED","OWN_ASSET","UNKNOWN"}
        if r.get("contact_class") not in valid: r["contact_class"] = "UNKNOWN"
        r.setdefault("confidence", 0.5)
        r.setdefault("threat_level", "UNKNOWN")
        r.setdefault("kinematics_summary", "")
        r.setdefault("electronic_id_summary", "")
        r.setdefault("admin_data_summary", "")
        r.setdefault("spatial_zone", "BUFFER")
        r.setdefault("reasoning", "")
        r["confidence"] = max(0.0, min(1.0, float(r["confidence"])))
        return r
