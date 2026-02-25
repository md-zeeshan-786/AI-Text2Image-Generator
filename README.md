# SDXL + Phi-3 Guard: Prompt-Filtered Image Generator (CUDA-Optimized)

This project runs a **small, fast LLM (Microsoft Phi-3-mini)** as a **prompt filter** on CPU and **SDXL** on GPU for image generation.
It supports **8 GB VRAM** GPUs like the **RTX 3060 Ti** using **xFormers, attention slicing, VAE tiling, sequential CPU offload** and **medvram** strategies.

## Hardware Target
- CPU: Intel i5-13600K (used to run the filter LLM on CPU)
- GPU: NVIDIA RTX 3060 Ti 8GB (used to run SDXL)
- RAM: 16 GB+ recommended
- OS: Windows 10/11

## Quick Start (Windows, fresh venv)
1. Install Python 3.10 or 3.11 (64-bit).
2. Open **CMD** or **PowerShell** in the project folder:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
3. Install PyTorch with CUDA (choose your CUDA build from https://pytorch.org/get-started/locally/):
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
   ```
   If you already have Torch installed with CUDA, you can skip this step.

4. Install project deps:
   ```bash
   pip install -r requirements.txt
   ```

5. (Optional) If Hugging Face requires auth for your region/models, login:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

6. Run the app:
   ```bash
   streamlit run app/app.py
   ```
   The app will open in your browser at http://localhost:8501

## SDXL Model
This app uses Diffusers with:
- Base: `stabilityai/stable-diffusion-xl-base-1.0`

Optimizations included:
- xFormers memory efficient attention
- attention slicing
- VAE tiling
- channels-last tensors
- sequential CPU offload (optional)
- memory cleanup between runs

## Prompt Filter (Microsoft Phi-3-mini)
- Default: `microsoft/Phi-3-mini-4k-instruct` loaded on CPU
- Role: Classify user prompt into one of:
  - `ALLOW`
  - `BLOCK_SENSITIVE`
  - `BLOCK_ILLOGICAL`
- If blocked, the UI shows:
  - Sensitive: "We're sorry! We cannot process the request as the prompt has sensitive words"
  - Illogical: "We're sorry! We cannot process the request as the prompt is logically wrong"
- If unclear, we default to **block**.

### Why unload/reload models each prompt?
With 8GB VRAM, SDXL is tight. The app uses the following sequence for each prompt:
1. Load Phi-3-mini on CPU.
2. Classify. If `ALLOW`, unload Phi-3-mini.
3. Load SDXL on GPU.
4. Generate image.
5. Unload SDXL to free VRAM.

This matches the user's VRAM constraints and request.

## Notes on Safety
- We apply a **fast regex prefilter** for obvious sensitive or prohibited content to avoid wasting compute.
- The LLM guard adds contextual judgement (e.g., flags illogical prompts like "an apple tree in the middle of the ocean" or illegal content).
- **You can edit `app/safety_filter.py`** to expand blocked categories or terms.

## Troubleshooting
- **CUDA OOM**: Reduce image size, steps, or enable all memory savers in the sidebar. Close other GPU apps. Ensure xFormers is installed.
- **xFormers build**: On Windows, pip has prebuilt wheels for common CUDA versions. If installation fails, run without xFormers (it still works, just slower).
- **Slow first run**: Models download on the first run. Subsequent runs are faster.
- **Torch CUDA check**: If the app says CUDA is unavailable, confirm your GPU drivers and CUDA toolkits are installed correctly.

## Project Layout
```
app/
  app.py                  # Streamlit UI and orchestration
  sdxl_generator.py       # SDXL pipeline with CUDA/medvram optimizations
  safety_filter.py        # Regex prefilter + logic checks
  filter_llm.py           # Phi-3-mini classifier (CPU)
  utils.py                # Memory mgmt, logging, helpers
  assets/style.css        # Extra styles
models/
  model_manifest.json     # Central place to swap models/ids
outputs/                  # Saved images
```

---

**Authoring note**: This bundle does **not** ship model weights. It references open model repos for automatic runtime download.
