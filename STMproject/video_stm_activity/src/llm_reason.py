# src/llm_reason.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ActivityClassifierLLM:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        system_prompt: str = ""
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.device = device

        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto" if device != "cpu" else None,
            "torch_dtype": torch_dtype,
        }

        # 4-bit quantisation only if on GPU
        if load_in_4bit and device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def freeform(self, prompt: str) -> str:
        """Run a free-form prompt and return decoded output text."""
        # Build chat prompt if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if self.system_prompt.strip():
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = (self.system_prompt.strip() + "\n\n" + prompt).strip()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return decoded

    def predict_label(self, memory_text: str, allowed_labels: list[str]) -> str:
        prompt = f"""
You are analysing a video described by timestamped observations.

Choose ONE label from this exact list:
{allowed_labels}

Rules:
- Pick the MAIN activity / intent.
- Output ONLY one line in this exact format:
label=<one_label>

OBSERVATIONS:
{memory_text}
"""
        out = self.freeform(prompt)

        m = re.search(r"label\s*=\s*([a-zA-Z_]+)", out)
        if not m:
            return "other"
        label = m.group(1).strip().lower()
        return label if label in allowed_labels else "other"