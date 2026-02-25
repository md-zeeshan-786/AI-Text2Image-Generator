import os
import time
import json
import streamlit as st
from PIL import Image

# Absolute imports
from safety_filter import fast_prefilter, normalize_prompt
from filter_llm import Phi3Guard
from sdxl_generator import build_sdxl_pipeline, generate_image, unload_sdxl
from utils import free_all

# ==============================
# Load manifest
# ==============================
manifest_path = os.path.join("models", "model_manifest.json")
if not os.path.exists(manifest_path):
    st.error(f"❌ model_manifest.json not found at {manifest_path}")
    st.stop()

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# ==============================
# Streamlit config
# ==============================
st.set_page_config(
    page_title="SDXL + Phi-3 Guard",
    page_icon="🎨",
    layout="wide"
)

# ==============================
# Load CSS
# ==============================
css_path = None
for candidate in [
    os.path.join("app", "assets", "style.css"),
    os.path.join("assets", "style.css"),
]:
    if os.path.exists(candidate):
        css_path = candidate
        break

if css_path:
    with open(css_path, "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ==============================
# UI
# ==============================
st.title("Phi-3-Guided SDXL")
st.caption("Prompt filter on CPU, SDXL on GPU.")

# Sidebar settings
with st.sidebar:
    st.subheader("Generation Settings")
    steps = st.slider("Steps", 10, 60, 28, 1)
    guidance = st.slider("Guidance Scale", 1.0, 12.0, 5.5, 0.5)
    width = st.selectbox("Width", [640, 768, 832, 896, 1024], index=4)
    height = st.selectbox("Height", [640, 768, 832, 896, 1024], index=4)
    seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
    seed = None if seed == 0 else int(seed)

    st.subheader("Memory & Performance")
    use_xformers = st.toggle("Use xFormers", value=True)
    medvram = st.toggle("MedVRAM mode", value=True)
    cpu_offload = st.toggle("Enable CPU offload", value=True)

    unload_between = st.toggle(
        "Unload models between runs",
        value=True,
        help="Keeps VRAM low by unloading after each generation"
    )

    st.divider()
    st.caption("Models Loaded from manifest.json")
    st.json(manifest)

# Prompt inputs
col1, col2 = st.columns([1, 1])
with col1:
    prompt = st.text_area("Your prompt", height=120, placeholder="Describe the image you want")
with col2:
    negative_prompt = st.text_area("Negative prompt (optional)", height=120, placeholder="Things to avoid")

run = st.button("Generate", type="primary")

# Placeholders
status_placeholder = st.empty()
gallery = st.container()
log_box = st.expander("Run Log / Details", expanded=False)

# ==============================
# Helper - error messages
# ==============================
def block_msg(kind: str):
    if kind == "SENSITIVE":
        st.error("❌ Sorry! This prompt contains sensitive words.")
    else:
        st.error("❌ Sorry! This prompt is logically invalid.")

# ==============================
# Main Logic
# ==============================
if run:
    if not prompt or not prompt.strip():
        st.error("Prompt is required!")
    else:
        t0 = time.time()
        p_norm = normalize_prompt(prompt)
        allow, reason = fast_prefilter(p_norm)
        verdict = None

        if not allow:
            verdict = "BLOCK_" + reason
            block_msg(reason)
        else:
            # Load Phi-3-mini guard (force GPU)
            status_placeholder.info("Loading Phi-3-mini filter on GPU...")
            guard = Phi3Guard(model_id=manifest.get("filter_llm"), device="cuda")
            guard.load()

            verdict = guard.classify(p_norm)
            if verdict != "ALLOW":
                if verdict == "BLOCK_SENSITIVE":
                    block_msg("SENSITIVE")
                else:
                    block_msg("ILLOGICAL")

            # Unload guard
            status_placeholder.info("Unloading Phi-3-mini filter...")
            guard.unload()

        if verdict == "ALLOW":
            # Force SDXL to load fully on GPU
            memory_mode = "gpu"

            # Load SDXL
            model_id = manifest.get("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0")
            status_placeholder.info(f"Loading SDXL ({model_id}) with memory_mode='{memory_mode}'...")
            pipe = build_sdxl_pipeline(
                model_id=model_id,
                use_xformers=use_xformers,
                memory_mode=memory_mode
            )

            # Generate
            status_placeholder.info("Generating image...")
            image = generate_image(
                pipe,
                prompt=p_norm,
                negative_prompt=negative_prompt if negative_prompt else None,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                seed=seed
            )

            # Save output
            os.makedirs("outputs", exist_ok=True)
            fname = f"outputs/sdxl_{int(time.time())}.png"
            image.save(fname)

            with gallery:
                st.image(Image.open(fname), caption=fname, use_column_width=True)

            status_placeholder.success("✅ Done. Image saved to " + fname)

            if unload_between:
                status_placeholder.info("Unloading SDXL to free VRAM...")
                unload_sdxl(pipe)

            free_all()

        # Log
        t1 = time.time()
        with log_box:
            st.write(f"Verdict: {verdict}")
            st.write(f"Elapsed: {t1 - t0:.2f}s")

st.markdown('<div class="footer">© 2025 SDXL + Phi-3 Guard</div>', unsafe_allow_html=True)
