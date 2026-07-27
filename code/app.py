"""
CMFD demo dashboard (Streamlit)
===============================
Upload a .jpg / .jpeg / .png image, run a copy-move forgery detector, and get:
FORGED / AUTHENTIC verdict + a forgery heatmap + bounding boxes marking where
the image was copy-move forged.

Two engines:
  * SIFT + RANSAC (default) - fast, robust classical keypoint matching with
    geometric + compactness verification. Best for distinct copied objects.
  * Swap-test QNN - the quantum model. Kept for the quantum component of the
    project; over-detects, so treated as experimental.

Run:
    cd code
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sift_cmfd import SiftForgeryDetector       # noqa: E402
from inference import ForgeryDetector           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints_full" / "swaptest_qnn_epoch10.npy"

st.set_page_config(page_title="Copy-Move Forgery Detector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner=False)
def get_sift(work_long_side: int):
    return SiftForgeryDetector(work_long_side=work_long_side)


@st.cache_resource(show_spinner=False)
def get_qnn(ckpt, n_qubits, n_layers, img_size, stride):
    return ForgeryDetector(ckpt, n_qubits=n_qubits, n_layers=n_layers,
                           img_size=img_size, patch_size=32, stride=stride)


# ----------------------------- sidebar -------------------------------------
st.sidebar.title("⚙️ Settings")
engine = st.sidebar.radio("Detection engine",
                          ["SIFT + RANSAC  (fast, recommended)", "Swap-test QNN  (quantum, slow)"])
use_sift = engine.startswith("SIFT")

if use_sift:
    st.sidebar.markdown("**SIFT + RANSAC**")
    work_side = st.sidebar.select_slider("Working resolution (long side)",
                                         [512, 768, 1000, 1400], value=1000)
    ratio = st.sidebar.slider("g2NN match ratio", 0.3, 0.8, 0.5, 0.05)
    ransac_thresh = st.sidebar.slider("RANSAC tolerance (px)", 2.0, 12.0, 6.0, 0.5)
    min_cluster = st.sidebar.slider("Decision: min compact-cluster size", 4, 60, 12)
    st.sidebar.caption("Higher cluster size → fewer detections. Repetitive "
                       "facades (regular windows) can still cause false "
                       "positives — a known hard case for all methods.")
else:
    st.sidebar.markdown("**Swap-test QNN** (must match the checkpoint)")
    ckpt = st.sidebar.text_input("Checkpoint (.npy)", value=str(DEFAULT_CKPT))
    n_qubits = st.sidebar.number_input("Qubits per register", 2, 6, 4)
    n_layers = st.sidebar.number_input("Layers", 1, 4, 2)
    img_size = st.sidebar.select_slider("Working resolution", [128, 192, 256], value=256)
    stride = st.sidebar.select_slider("Patch stride", [8, 12, 16, 24], value=16)
    sim_threshold = st.sidebar.slider("Similarity threshold", 0.50, 0.99, 0.90, 0.01)
    min_offset_votes = st.sidebar.slider("Min offset votes", 1, 15, 6)
    decision_votes = st.sidebar.slider("Decision: forged if peak ≥", 1, 40, 8)

# ----------------------------- main ----------------------------------------
st.title("🔍 Copy-Move Forgery Detection")
st.caption("Upload an image to check whether a region was duplicated, and where.")

up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
run = st.button("▶ Run analysis", type="primary", disabled=up is None)
if up is None:
    st.info("Upload a .jpg, .jpeg or .png image to begin.")

if run and up is not None:
    image = Image.open(up)
    bar = st.progress(0.0, text="Analyzing…")
    try:
        if use_sift:
            det = get_sift(int(work_side))
            result = det.analyze(image, ratio=ratio, ransac_thresh=ransac_thresh,
                                 min_cluster=int(min_cluster),
                                 progress=lambda f: bar.progress(min(f, 1.0), text="SIFT matching…"))
            signal_label = "Compact cluster"
        else:
            if not Path(ckpt).exists():
                st.error(f"Checkpoint not found: {ckpt}"); st.stop()
            det = get_qnn(ckpt, int(n_qubits), int(n_layers), int(img_size), int(stride))
            result = det.analyze(image, sim_threshold=sim_threshold,
                                 min_offset_votes=int(min_offset_votes),
                                 decision_votes=int(decision_votes),
                                 progress=lambda f: bar.progress(min(f, 1.0), text="QNN scoring…"))
            signal_label = "Peak offset votes"
    except Exception as e:
        bar.empty(); st.error(f"Analysis failed: {e}"); st.stop()
    bar.empty()

    if result.is_forged:
        st.error(f"### 🚩 FORGED  ·  confidence {result.confidence:.0%}")
    else:
        st.success(f"### ✅ AUTHENTIC  ·  confidence {(1 - result.confidence):.0%}")

    m1, m2, m3 = st.columns(3)
    m1.metric(signal_label, result.peak_votes)
    m2.metric("Keypoint/patch matches", result.n_matched_pairs)
    m3.metric("Forged regions (boxes)", len(result.boxes))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Input image**"); c1.image(result.original, use_container_width=True)
    c2.markdown("**Forgery heatmap**"); c2.image(result.heatmap_overlay, use_container_width=True)
    c3.markdown("**Localized regions / matches**"); c3.image(result.box_overlay, use_container_width=True)

    if result.is_forged and result.boxes:
        st.markdown(f"**Detected region(s):** {result.boxes}  ·  "
                    f"dominant shift (dy, dx) = {result.dominant_offset}")

    with st.expander("What am I looking at?"):
        st.markdown(
            "- **Heatmap** — brighter = stronger duplication evidence.\n"
            "- **Boxes / lines** — matched duplicated regions; copy-move marks BOTH "
            "the source and the pasted copy. In SIFT mode, gold lines link matched keypoints.\n"
            f"- **{signal_label}** — the core signal driving the verdict.\n\n"
            "⚠️ Works best on distinct copied *objects*. Highly repetitive scenes "
            "(regular building facades) can trigger false positives — a documented hard "
            "case for copy-move detection in general.")

st.markdown("---")
st.caption("FYP Phase-2 · Quantum-Image-Processing/qnn")
