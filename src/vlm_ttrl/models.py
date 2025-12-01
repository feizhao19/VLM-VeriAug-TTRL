# src/vlm_ttrl/models.py

from typing import List, Dict, Any
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# You can switch models here
DEFAULT_VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# DEFAULT_VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class Qwen3VLWrapper:
    """
    A simple wrapper that:
    - Forces the model to be placed entirely on GPU0
    - Provides a single-image multimodal dialog interface: chat_single_image
    """

    def __init__(self, model_id: str = DEFAULT_VLM_MODEL_ID, verbose: bool = True):
        self.model_id = model_id
        self.verbose = verbose

        # 1. Choose device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if self.verbose:
                print("[Init] CUDA is available, using device:", self.device)
                print("[Init] Number of visible GPUs:", torch.cuda.device_count())
                print("[Init] Current GPU name:", torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("[Init] CUDA is not available, falling back to CPU")

        # 2. Set dtype
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if self.verbose:
            print("[Init] Using tensor dtype:", self.dtype)

        # 3. Load model (force all parameters onto GPU0)
        if self.verbose:
            print(f"[Init] Start loading model from {self.model_id} ...")
            t0 = time.time()

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map={"": self.device.index if self.device.type == "cuda" else "cpu"},
        )

        if self.verbose:
            t1 = time.time()
            print(f"[Init] Model loaded, time elapsed {t1 - t0:.2f} seconds")
            # Print the device of the first parameter
            first_param_device = next(self.model.parameters()).device
            print("[Init] First parameter device:", first_param_device)
            # Print hf_device_map if it exists
            hf_map = getattr(self.model, "hf_device_map", None)
            print("[Init] hf_device_map:", hf_map)

        # Ensure using KV cache
        try:
            self.model.generation_config.use_cache = True
            if self.verbose:
                print("[Init] Set generation_config.use_cache = True")
        except Exception as e:
            if self.verbose:
                print("[Init] Failed to set use_cache:", e)

        # 4. Load Processor (usually on CPU and that is fine)
        if self.verbose:
            print(f"[Init] Start loading Processor from {self.model_id} ...")
            t2 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.verbose:
            t3 = time.time()
            print(f"[Init] Processor loaded, time elapsed {t3 - t2:.2f} seconds")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                used = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
                print(f"[Init] GPU memory allocated: {used:.1f} MiB, reserved: {reserved:.1f} MiB")

    def _build_messages(self, image: Image.Image, text: str) -> List[Dict[str, Any]]:
        """
        Build chat-style input format:
        [
          {
            "role": "user",
            "content": [
              {"type": "image", "image": <PIL.Image>},
              {"type": "text", "text": "your prompt"}
            ]
          }
        ]
        """
        if self.verbose:
            print("[Build] Building messages, text prompt:", text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        if self.verbose:
            print("[Build] Messages structure:", messages)
        return messages

    def chat_single_image(
        self,
        image_path: str,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
    ) -> str:
        """
        Perform multimodal dialog inference on a single image and return the generated text.
        Prints debug information at each step to verify data is on the GPU.
        """
        if self.verbose:
            print("\n[Chat] Start single-image inference")
            print("[Chat] Image path:", image_path)
            print("[Chat] Text prompt:", text)
            print("[Chat] max_new_tokens:", max_new_tokens)
            print("[Chat] temperature:", temperature)

        t_total_start = time.time()

        # 1. Read image
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_file).convert("RGB")
        if self.verbose:
            print("[Chat] Image loaded, size:", image.size, "mode:", image.mode)

        # 2. Build messages
        messages = self._build_messages(image, text)

        # 3. Use processor to build model inputs (doing this on CPU is fine)
        if self.verbose:
            print("[Chat] Using processor.apply_chat_template to build inputs...")
            t_p0 = time.time()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if self.verbose:
            t_p1 = time.time()
            print(f"[Chat] Processor finished, time elapsed {t_p1 - t_p0:.2f} seconds")
            print("[Chat] Processor output keys:", list(inputs.keys()))
            for k, v in inputs.items():
                print(f"[Chat]   {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")

        # 4. Move all inputs to the model device
        if self.verbose:
            print("[Chat] Moving inputs to model device:", self.device)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.verbose:
            for k, v in inputs.items():
                print(f"[Chat]   {k} is now on device={v.device}, shape={tuple(v.shape)}, dtype={v.dtype}")

            if self.device.type == "cuda":
                used = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
                print(f"[Chat] After moving inputs, GPU memory allocated: {used:.1f} MiB, reserved: {reserved:.1f} MiB")

        # 5. Call generate
        if self.verbose:
            print("[Chat] Start generation...")
            print("[Chat] generation_config:", self.model.generation_config)

        t_gen0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
            )
        t_gen1 = time.time()

        if self.verbose:
            print(f"[Chat] Generation finished, time elapsed {t_gen1 - t_gen0:.2f} seconds")
            print("[Chat] generated_ids shape:", tuple(generated_ids.shape))

        # 6. Strip prompt tokens and keep only newly generated tokens
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = []
        total_new_tokens = 0
        for in_ids, out_ids in zip(input_ids, generated_ids):
            trimmed = out_ids[len(in_ids):]
            generated_ids_trimmed.append(trimmed)
            total_new_tokens += trimmed.numel()

        if self.verbose:
            print("[Chat] Number of newly generated tokens per output (total):", total_new_tokens)

        # 7. Decode outputs
        if self.verbose:
            print("[Chat] Start decoding output text...")

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        t_total_end = time.time()
        if self.verbose:
            print("[Chat] Decoding finished")
            print(f"[Chat] Total time for chat_single_image: {t_total_end - t_total_start:.2f} seconds")
            print("[Chat] Final output text:")
            print(output_text)

        return output_text