# scripts/test_qwen3_vl_infer_two_images.py

import os
import sys
import time

# Add src/ into Python search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # .../VLM-VeriAug-TTRL
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pathlib import Path
from PIL import Image

from vlm_ttrl.models import Qwen3VLWrapper


def build_two_image_messages(image1, image2, text):
    """
    Build chat messages that contain two images:
    [
      {
         "role": "user",
         "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "text",  "text": text},
         ]
      }
    ]
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": text},
            ],
        }
    ]


def main():
    print("[Main] Starting two-image test script")
    print("[Main] ROOT_DIR:", ROOT_DIR)
    print("[Main] SRC_DIR:", SRC_DIR)

    # Initialize wrapper
    t0 = time.time()
    model = Qwen3VLWrapper(verbose=True)
    t1 = time.time()
    print(f"[Main] Model wrapper initialized, time elapsed {t1 - t0:.2f} seconds")

    # Input images
    img1_path = "./data/images/test_img1.jpg"
    img2_path = "./data/images/test_img2.jpg"

    print("[Main] Image 1 path:", img1_path)
    print("[Main] Image 2 path:", img2_path)

    # Load images
    img1 = Image.open(Path(img1_path)).convert("RGB")
    img2 = Image.open(Path(img2_path)).convert("RGB")

    prompt = "Compare these two images and describe their differences."
    print("[Main] Prompt:", prompt)

    # Build messages manually
    messages = build_two_image_messages(img1, img2, prompt)

    # Use processor + generate manually (bypassing chat_single_image)
    inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    t2 = time.time()
    output_ids = model.model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=True,
    )
    t3 = time.time()

    # Strip prompt
    trimmed = output_ids[0][len(inputs["input_ids"][0]):]

    # Decode
    output_text = model.processor.decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n===== MODEL OUTPUT =====")
    print(output_text)
    print("========================")
    print(f"[Main] Total inference time {t3 - t2:.2f} seconds")


if __name__ == "__main__":
    main()