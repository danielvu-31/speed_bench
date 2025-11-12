# 🧪 Speed Bench

Benchmark scripts for evaluating **SANA**, **BrushNet**, and their integration with **SDXL** models.

---
## Settings
If possible, benchmark on an empty node for best accuracy

## ⚙️ 1. Environment Setup

### 1.1 For SANA

```bash
conda create -n testsana python=3.10 -y
conda activate testsana
conda install git-lfs -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_sana.txt
conda deactivate
```

### 1.1 For BrushNet
```bash
conda create -n testbrush python=3.9 -y
conda activate testbrush
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_sana.txt
cp dynamic_modules_utils.py src/diffusers/src/diffusers/utils
conda deactivate
```

## 💾 2. Checkpoints and Data (≈24 GB)
PLEASE make sure git-lfs is installed

```bash
conda activate testsana
git lfs install
git clone https://huggingface.co/ducvh5/checkpoint
conda deactivate
```

## 🚀 3. Run Experiments
Run the following scripts one by one to reproduce benchmark results:
```bash
bash scripts/sana.sh
bash scripts/sanasprint.sh
bash scripts/sanasprint+inverfill.sh
bash scripts/sdxl.sh
bash scripts/sdxl+inverfill.sh
bash scripts/sdxl+brushnet.sh
bash scripts/sdxl+inverfill+brushnet.sh
bash scripts/sdxlinpaint.sh
bash scripts/jugg+brushnet.sh
```

