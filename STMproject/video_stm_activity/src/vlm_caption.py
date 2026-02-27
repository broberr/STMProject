import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


class VLMCaptioner:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 80,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        torch_dtype = getattr(torch, dtype)

        self.processor = AutoProcessor.from_pretrained(model_id)

        model_kwargs = {"dtype": torch_dtype, "device_map": "cuda" if device != "cpu" else None}



        # ✅ proper 4-bit quant config (instead of load_in_4bit kwarg)
        if load_in_4bit and device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    @staticmethod
    def _bgr_to_pil(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def caption(self, bgr_img, prompt: str) -> str:
        image = self._bgr_to_pil(bgr_img)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(text=text, images=image, return_tensors="pt")

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if "ASSISTANT:" in decoded:
            decoded = decoded.split("ASSISTANT:", 1)[1].strip()
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]", 1)[-1].strip()

        return decoded




