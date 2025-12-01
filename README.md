# VLM-VeriAug-TTRL

A research framework for exploring **Reinforcement Learning (RL)** on **Vision-Language Models (VLMs)** using verified augmentations and structured reasoning at inference time.

---

## 1. Create Conda Environment

```bash
conda create -n vl-ttrl python=3.10 -y
conda activate vl-ttrl


⸻

2. Install PyTorch (use the official website)

Visit:

https://pytorch.org

Select:
	•	OS: Linux
	•	Package: pip or conda
	•	Compute Platform: your CUDA version

Install using the command provided by the website. Example:

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


⸻

3. Install Project Dependencies

After PyTorch is installed:

pip install -r requirements.txt

This installs:
	•	transformers, accelerate
	•	trl
	•	peft, bitsandbytes
	•	datasets, pillow, opencv-python
	•	utilities: numpy, scipy, huggingface_hub, tqdm

⸻

5. Test Qwen2.5-VL

Set default model in:

src/vlm_ttrl/models.py

DEFAULT_VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

Run inference:

python scripts/test_vl.py

Expected output:

===== MODEL OUTPUT =====
A photo of ...


⸻

6. Test Qwen3-VL

Switch default model:

DEFAULT_VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

Run again:

python scripts/test_vl.py

Note: Qwen3-VL requires more GPU memory.

⸻
