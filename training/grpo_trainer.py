"""
GRPO Trainer — Group Relative Policy Optimization
===================================================
Fine-tunes Qwen agents using GRPO (the RL algorithm from DeepSeek-R1).

Key idea:
  For each prompt, sample G completions (a "group").
  Normalize rewards within the group (mean-center).
  Apply policy gradient update using normalized advantages.
  No separate value/critic model needed.

Supports:
  - LoRA fine-tuning via PEFT (memory-efficient)
  - 4-bit quantization (GPU with bitsandbytes)
  - CPU training (slow but functional for testing)
  - Checkpoint save/load
"""

import json
import os
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW


class GRPOConfig:
    """Configuration for GRPO training."""

    def __init__(
        self,
        model_name:       str   = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_rank:        int   = 16,
        lora_alpha:       int   = 32,
        lora_dropout:     float = 0.05,
        lora_target_modules: List[str] = None,
        group_size:       int   = 4,      # G: completions per prompt
        lr:               float = 2e-5,
        epochs:           int   = 3,
        batch_size:       int   = 2,
        max_grad_norm:    float = 1.0,
        kl_coeff:         float = 0.01,   # KL divergence penalty coefficient
        temperature:      float = 0.8,    # sampling temperature for group
        max_new_tokens:   int   = 256,
        checkpoint_dir:   str   = "checkpoints",
        device:           str   = "auto",
        use_4bit:         bool  = False,  # require bitsandbytes
        gradient_checkpointing: bool = True,
        seed:             int   = 42,
    ):
        self.model_name       = model_name
        self.lora_rank        = lora_rank
        self.lora_alpha       = lora_alpha
        self.lora_dropout     = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.group_size       = group_size
        self.lr               = lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.max_grad_norm    = max_grad_norm
        self.kl_coeff         = kl_coeff
        self.temperature      = temperature
        self.max_new_tokens   = max_new_tokens
        self.checkpoint_dir   = checkpoint_dir
        self.device           = device
        self.use_4bit         = use_4bit
        self.gradient_checkpointing = gradient_checkpointing
        self.seed             = seed


class GRPOTrainer:
    """
    GRPO fine-tuning trainer for Qwen agents.

    Usage:
        cfg     = GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", epochs=3)
        trainer = GRPOTrainer(cfg)
        trainer.train(train_samples, eval_samples)
        trainer.save_checkpoint("checkpoints/final")
    """

    def __init__(self, config: GRPOConfig):
        self.cfg     = config
        self.model   = None
        self.tokenizer = None
        self.ref_model = None  # frozen reference model for KL penalty
        self.optimizer = None
        self.device    = None
        self._setup()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup(self):
        """Load model, apply LoRA, set up optimizer."""
        print(f"[GRPO] Setting up trainer | model={self.cfg.model_name}")
        torch.manual_seed(self.cfg.seed)

        # Determine device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        print(f"[GRPO] Device: {self.device}")

        self._load_model()
        self._apply_lora()
        self._setup_optimizer()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(trust_remote_code=True)

        if self.cfg.use_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb_cfg
                load_kwargs["device_map"] = "auto"
                print("[GRPO] Using 4-bit quantization.")
            except ImportError:
                print("[GRPO] bitsandbytes not available. Loading in fp16.")
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        elif str(self.device) == "cpu":
            load_kwargs["torch_dtype"] = torch.float32
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)

        if self.cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Frozen reference model for KL penalty
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        print(f"[GRPO] Model loaded. Params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _apply_lora(self):
        """Apply LoRA adapters to the policy model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type      = TaskType.CAUSAL_LM,
                r              = self.cfg.lora_rank,
                lora_alpha     = self.cfg.lora_alpha,
                lora_dropout   = self.cfg.lora_dropout,
                target_modules = self.cfg.lora_target_modules,
                bias           = "none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        except ImportError:
            print("[GRPO] WARNING: peft not installed. Training all parameters (memory-intensive).")

    def _setup_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.cfg.lr, weight_decay=0.01)
        print(f"[GRPO] Optimizer ready. Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(
        self,
        train_samples: List[Dict],
        eval_samples:  List[Dict],
        log_every:     int = 10,
    ) -> List[Dict]:
        """
        Main GRPO training loop.

        Args:
            train_samples: list of {prompt, completion, reward} dicts
            eval_samples:  list of {prompt, completion, reward} dicts
            log_every:     log metrics every N batches

        Returns:
            training_log: list of metric dicts per epoch
        """
        print(f"\n[GRPO] Starting training | epochs={self.cfg.epochs} | train={len(train_samples)} | eval={len(eval_samples)}")
        training_log = []

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\n{'='*50}")
            print(f"[GRPO] EPOCH {epoch}/{self.cfg.epochs}")
            print(f"{'='*50}")

            epoch_loss     = 0.0
            epoch_reward   = 0.0
            batch_count    = 0
            t0 = time.time()

            # Shuffle training data each epoch
            import random
            random.shuffle(train_samples)

            for i in range(0, len(train_samples), self.cfg.batch_size):
                batch = train_samples[i : i + self.cfg.batch_size]
                loss, mean_reward = self._grpo_step(batch)

                epoch_loss   += loss
                epoch_reward += mean_reward
                batch_count  += 1

                if batch_count % log_every == 0:
                    elapsed = time.time() - t0
                    print(f"  Batch {batch_count} | loss={loss:.4f} | mean_reward={mean_reward:.4f} | {elapsed:.1f}s elapsed")

            # Evaluation
            eval_reward = self._evaluate(eval_samples)
            epoch_log   = {
                "epoch":        epoch,
                "train_loss":   round(epoch_loss   / max(batch_count, 1), 4),
                "train_reward": round(epoch_reward / max(batch_count, 1), 4),
                "eval_reward":  round(eval_reward, 4),
            }
            training_log.append(epoch_log)
            print(f"\n[GRPO] Epoch {epoch} summary: {epoch_log}")

            # Save checkpoint locally
            ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch}")
            self.save_checkpoint(ckpt_path)

            # Upload to HF Hub after every epoch (so progress is never lost)
            self._upload_epoch_checkpoint(ckpt_path, epoch)

        print(f"\n[GRPO] Training complete.")
        return training_log

    def _upload_epoch_checkpoint(self, ckpt_path: str, epoch: int) -> None:
        """Upload per-epoch checkpoint to HF Hub immediately after saving."""
        hf_token  = os.environ.get("HF_TOKEN", "")
        username  = os.environ.get("HF_USERNAME", "Bhaskar111")
        agent_tag = "radar" if "radar" in ckpt_path.lower() else "actor"
        repo_id   = f"{username}/defense-rl-{agent_tag}-adapter"

        if not hf_token:
            print(f"[GRPO] HF_TOKEN not set — skipping epoch {epoch} upload.")
            return
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi(token=hf_token)
            create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
            api.upload_folder(
                folder_path = ckpt_path,
                repo_id     = repo_id,
                repo_type   = "model",
                token       = hf_token,
                commit_message = f"GRPO epoch {epoch} checkpoint",
            )
            print(f"[GRPO] ✅ Epoch {epoch} uploaded → https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[GRPO] Upload failed (epoch {epoch}): {e} — checkpoint saved locally.")

    # ── GRPO step ─────────────────────────────────────────────────────────────

    def _grpo_step(self, batch: List[Dict]) -> Tuple[float, float]:
        """
        One GRPO gradient update for a mini-batch.

        For each sample in the batch:
          1. Sample G completions from the policy model
          2. Score each completion with the reward function
          3. Normalize rewards within the group (mean-centering)
          4. Compute policy gradient loss with KL penalty
          5. Backpropagate and update LoRA weights
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_loss    = 0.0
        total_reward  = 0.0

        for sample in batch:
            prompt     = sample["prompt"]
            ext_reward = float(sample.get("reward", 0.0))

            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)

            # Sample G completions
            with torch.no_grad():
                group_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens    = self.cfg.max_new_tokens,
                    do_sample         = True,
                    temperature       = self.cfg.temperature,
                    num_return_sequences = self.cfg.group_size,
                    pad_token_id      = self.tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]

            # Score each completion
            group_rewards = []
            for out in group_outputs:
                completion_ids  = out[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                # Heuristic reward: combine external signal + format correctness
                format_reward  = self._format_reward(completion_text)
                total_r        = 0.7 * ext_reward + 0.3 * format_reward
                group_rewards.append(total_r)

            # Normalize rewards (GRPO advantage)
            rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
            if rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                advantages = rewards_t - rewards_t.mean()

            total_reward += rewards_t.mean().item()

            # Policy gradient loss (one forward pass per completion)
            sample_loss = 0.0
            for k, (out, adv) in enumerate(zip(group_outputs, advantages)):
                completion_ids = out[prompt_len:].unsqueeze(0)
                full_ids       = out.unsqueeze(0)
                attention_mask = torch.ones_like(full_ids)

                # Policy log-probs
                logits = self.model(input_ids=full_ids, attention_mask=attention_mask).logits
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                tok_log_probs = log_probs.gather(
                    2, full_ids[:, 1:].clamp(0, log_probs.shape[-1]-1).unsqueeze(-1)
                ).squeeze(-1)
                comp_log_probs = tok_log_probs[:, prompt_len-1:].mean()

                # Reference log-probs for KL penalty
                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids=full_ids, attention_mask=attention_mask).logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                ref_tok_lp    = ref_log_probs.gather(
                    2, full_ids[:, 1:].clamp(0, ref_log_probs.shape[-1]-1).unsqueeze(-1)
                ).squeeze(-1)
                ref_comp_lp   = ref_tok_lp[:, prompt_len-1:].mean()

                kl_penalty  = (comp_log_probs - ref_comp_lp).clamp(-10, 10)
                grpo_loss   = -(adv.to(self.device) * comp_log_probs) + self.cfg.kl_coeff * kl_penalty
                sample_loss += grpo_loss

            sample_loss = sample_loss / self.cfg.group_size
            sample_loss.backward()
            total_loss += sample_loss.item()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg.max_grad_norm
        )
        self.optimizer.step()

        n = len(batch)
        return total_loss / n, total_reward / n

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, eval_samples: List[Dict]) -> float:
        """Compute mean external reward on eval set (no gradient)."""
        if not eval_samples:
            return 0.0
        rewards = [float(s.get("reward", 0.0)) for s in eval_samples]
        return sum(rewards) / len(rewards)

    # ── Format reward helper ───────────────────────────────────────────────────

    @staticmethod
    def _format_reward(completion_text: str) -> float:
        """
        Reward model completions for correct JSON format.
        +0.3 if valid JSON, +0.2 if has required keys, +0.5 if class is valid.
        """
        import json
        try:
            data = json.loads(completion_text.strip())
            reward = 0.3
            if "contact_class" in data or "action" in data:
                reward += 0.2
            valid_classes = {
                "MISSILE_INBOUND","ENEMY_AIRCRAFT","FRIENDLY_AIRCRAFT",
                "DOMESTIC_FLIGHT","FOREIGN_PERMITTED","FOREIGN_UNPERMITTED",
                "OWN_ASSET","UNKNOWN",
                "COMM_HELLO","COMM_WARN","NAV_GUIDE","SYS_TRACK_ONLY",
                "WEAPON_ABM_LAUNCH","WEAPON_ADS_ENGAGE",
            }
            cc = data.get("contact_class") or data.get("action","")
            if cc in valid_classes:
                reward += 0.5
            return reward
        except Exception:
            return 0.0

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save LoRA adapter weights and tokenizer."""
        os.makedirs(path, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save config
        with open(os.path.join(path, "grpo_config.json"), "w") as f:
            json.dump(vars(self.cfg), f, indent=2)
        print(f"[GRPO] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load LoRA adapter weights from checkpoint."""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, path)
            print(f"[GRPO] Checkpoint loaded ← {path}")
        except Exception as e:
            print(f"[GRPO] Load checkpoint failed: {e}")
