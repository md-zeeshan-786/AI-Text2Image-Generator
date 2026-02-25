import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Tuple

SYSTEM_POLICY = """You are a content safety classifier.
Decide whether a user prompt is ALLOW, BLOCK_SENSITIVE, or BLOCK_ILLOGICAL.
- BLOCK_SENSITIVE includes sexual content (nudity, pornography), graphic violence, illegal activity, hate, or personal data abuse.
- BLOCK_ILLOGICAL includes physically impossible, contradictory, or nonsensical instructions (e.g., "an apple tree in the middle of the ocean" or "Sachin Tendulkar stealing a purse").
Reply ONLY with one of: ALLOW | BLOCK_SENSITIVE | BLOCK_ILLOGICAL.
"""

class Phi3Guard:
    def __init__(self, model_id: str, device: str = "cpu", dtype=torch.float32):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        Load tokenizer and model. If device indicates CUDA, prefer loading to GPU using
        device_map='auto' and fp16 dtype when CUDA is available for better performance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=True)

        # Decide device_map and dtype based on requested device
        device_map = self.device
        load_dtype = self.dtype

        if isinstance(self.device, str) and self.device.startswith("cuda"):
            # Use 'auto' device_map so HF places layers on GPU properly; use fp16 if CUDA present
            device_map = "auto"
            if torch.cuda.is_available():
                load_dtype = torch.float16
            else:
                load_dtype = self.dtype

        # from_pretrained: low_cpu_mem_usage kept as before
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=load_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )

        # If the model did not already get moved to CUDA by device_map='auto', ensure it's on device
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            try:
                # move model to the requested cuda device if needed
                self.model.to(self.device)
            except Exception:
                # if already on correct devices (via device_map='auto') this may fail — ignore
                pass

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify(self, prompt: str, max_new_tokens: int = 3, temperature: float = 0.7) -> str:
        assert self.model is not None and self.tokenizer is not None, "Call load() first"
        messages = [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt.strip()}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")
        if self.device == "cpu":
            pass
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        out_text = self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = out_text.strip().split()[0].upper()
        if verdict not in {"ALLOW", "BLOCK_SENSITIVE", "BLOCK_ILLOGICAL"}:
            verdict = "BLOCK_SENSITIVE"  # default safe
        return verdict
