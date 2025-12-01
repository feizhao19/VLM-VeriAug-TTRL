# scripts/test_qwen3_vl_infer.py

import os
import sys
import time

# Add src/ into Python search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # .../VLM-VeriAug-TTRL
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vlm_ttrl.models import Qwen3VLWrapper


def main():
    print("[Main] Starting test script: test_qwen3_vl_infer.py")
    print("[Main] ROOT_DIR:", ROOT_DIR)
    print("[Main] SRC_DIR:", SRC_DIR)

    t0 = time.time()
    # verbose=True prints detailed information for each step
    model = Qwen3VLWrapper(verbose=True)
    t1 = time.time()
    print(f"[Main] Model wrapper initialized, time elapsed {t1 - t0:.2f} seconds")

    # Test image path
    img_path = "/home/ubuntu/disks/400g/project/VLM-VeriAug-TTRL/data/images/test_img2.jpg"
    print("[Main] Test image path:", img_path)

    # prompt = "Describe this image in one short sentence."
    prompt = "Describe this image in one short sentence. Do you see any persons in the image?"
    print("[Main] Test prompt:", prompt)

    t2 = time.time()
    out = model.chat_single_image(img_path, prompt, max_new_tokens=64, temperature=0.2)
    t3 = time.time()

    print("\n===== MODEL OUTPUT =====")
    print(out)
    print("========================")
    print(f"[Main] Total inference time {t3 - t2:.2f} seconds")


if __name__ == "__main__":
    main()