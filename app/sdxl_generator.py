import torch, time, gc
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from typing import Optional
from PIL import Image


# memory_mode choices:
#   "gpu"                -> keep everything on CUDA
#   "sequential_offload" -> enable_sequential_cpu_offload (best for low VRAM)
#   "model_offload"      -> enable_model_cpu_offload (moderate VRAM)
def build_sdxl_pipeline(
    model_id: str,
    use_xformers: bool = True,
    memory_mode: str = "gpu",
):
    print(f"[INFO] Loading SDXL model: {model_id} with memory_mode={memory_mode}")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not load model '{model_id}': {e}")

    # Replace scheduler with DPM++
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable memory-efficient attention if possible
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] Enabled xFormers attention")
        except Exception as e:
            print(f"[WARN] Could not enable xFormers: {e}")

    # General optimizations
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_tiling()

    # Apply the selected memory strategy
    memory_mode = (memory_mode or "gpu").lower()
    if memory_mode == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
        print("[INFO] Using sequential CPU offload")
    elif memory_mode == "model_offload":
        pipe.enable_model_cpu_offload()
        print("[INFO] Using model CPU offload")
    else:
        pipe.to("cuda")
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            print("[INFO] Pipeline loaded fully on GPU")
        except Exception:
            pass

    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 30,
    guidance: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate a single image using SDXL.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    print(f"[INFO] Generating image: steps={steps}, guidance={guidance}, seed={seed}")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )

    return result.images[0]


def unload_sdxl(pipe):
    """
    Clean up pipeline and free VRAM/CPU memory.
    """
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("[INFO] SDXL pipeline unloaded and memory cleared")
