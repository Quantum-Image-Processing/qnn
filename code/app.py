"""
CMFD demo dashboard (Streamlit)
===============================
Upload a .jpg / .jpeg / .png image, run a copy-move forgery detector, and get:
FORGED / AUTHENTIC verdict + a forgery heatmap + bounding boxes marking where
the image was copy-move forged.

Two engines, no tuning knobs - the sidebar offers exactly one choice:
  * Pure classical - SIFT + RANSAC keypoint matching with geometric +
    compactness verification. Fast, and the one that actually works.
  * Pure quantum - the v3 Siamese swap-test QNN. Genuinely quantum end to
    end: amplitude-encoded raw pixels, exhaustive circuit-scored pairs, an
    in-circuit offset register, and no classical feature/prefilter/vote/
    morphology stage anywhere. Slower and less accurate than SIFT; that is
    the honest cost of removing the classical scaffolding.

Every threshold is pinned in SIFT_CFG / QNN_CFG below at the values the
Phase-2 evaluation used, so the two methods are compared as whole methods.

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
# v3 (pure QNN) checkpoints are .npz - v2's .npy weights are a different
# circuit shape and are deliberately rejected by SwapTestSiameseQNN.load().
DEFAULT_CKPT = ROOT / "checkpoints_pure" / "swaptest_qnn_epoch10.npz"

# Fixed operating points — the settings the Phase-2 results were produced with.
SIFT_CFG = dict(work_long_side=1000, ratio=0.5, ransac_thresh=6.0, min_cluster=12)
# stride == patch_size so patches TILE. A larger stride leaves the image
# partly unexamined: at 256px/stride 16 a forged image and its authentic
# original returned identical results because the forged region was never
# sampled. img_size 128 keeps tiled scoring (~33k pairs, ~20 min/image)
# affordable; drop to 96 for ~6 min if the demo wait is too long.
QNN_CFG = dict(n_layers=3, img_size=128, patch_size=8, stride=8,
               sim_threshold=0.90, decision_pairs=8)

st.set_page_config(page_title="Copy-Move Forgery Detector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner=False)
def get_sift(work_long_side: int):
    return SiftForgeryDetector(work_long_side=work_long_side)


@st.cache_resource(show_spinner=False)
def get_qnn(ckpt, n_layers, img_size, patch_size, stride):
    return ForgeryDetector(ckpt, n_layers=n_layers, img_size=img_size,
                           patch_size=patch_size, stride=stride)


# ----------------------------- sidebar -------------------------------------
# Only the engine choice is exposed. Every knob below is fixed at the value
# the Phase-2 evaluation was run with, so the two engines are compared as
# whole methods rather than as two piles of sliders.
CLASSICAL = "Pure classical  ·  SIFT + RANSAC"
QUANTUM = "Pure quantum  ·  swap-test QNN"

st.sidebar.title("⚙️ Engine")
engine = st.sidebar.radio("Detection method", [CLASSICAL, QUANTUM], label_visibility="collapsed")
use_sift = engine == CLASSICAL

# ----------------------------- main ----------------------------------------
st.title("🔍 Copy-Move Forgery Detection")
st.caption("Upload an image to check whether a region was duplicated, and where.")

if not use_sift:
    st.warning(
        "**Experimental — pure-QNN mode (v3).** Every classical detection stage has "
        "been removed, including the offset-consistency vote that previously "
        "suppressed false matches on homogeneous regions (sky, walls), so expect "
        "more over-detection than the SIFT engine and than the earlier hybrid. "
        "Scoring is exhaustive over patch pairs and runs one circuit per pair: "
        "**expect roughly 20 minutes per image**, not seconds. Treat this as a "
        "demonstration of a fully quantum detector, not as a production detector.")

up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
run = st.button("▶ Run analysis", type="primary", disabled=up is None)
if up is None:
    st.info("Upload a .jpg, .jpeg or .png image to begin.")

if run and up is not None:
    image = Image.open(up)
    bar = st.progress(0.0, text="Analyzing…")
    try:
        if use_sift:
            det = get_sift(SIFT_CFG["work_long_side"])
            result = det.analyze(image, ratio=SIFT_CFG["ratio"],
                                 ransac_thresh=SIFT_CFG["ransac_thresh"],
                                 min_cluster=SIFT_CFG["min_cluster"],
                                 progress=lambda f: bar.progress(min(f, 1.0), text="SIFT matching…"))
            signal_label = "Compact cluster"
        else:
            if not DEFAULT_CKPT.exists():
                st.error(f"Checkpoint not found: {DEFAULT_CKPT}"); st.stop()
            det = get_qnn(str(DEFAULT_CKPT), QNN_CFG["n_layers"], QNN_CFG["img_size"],
                          QNN_CFG["patch_size"], QNN_CFG["stride"])
            result = det.analyze(image, sim_threshold=QNN_CFG["sim_threshold"],
                                 decision_pairs=QNN_CFG["decision_pairs"],
                                 progress=lambda f: bar.progress(min(f, 1.0),
                                                                 text="Quantum pair scoring…"))
            signal_label = "Matched pairs"
    except Exception as e:
        bar.empty(); st.error(f"Analysis failed: {e}"); st.stop()
    bar.empty()

    if result.is_forged:
        st.error(f"### 🚩 FORGED  ·  confidence {result.confidence:.0%}")
    else:
        st.success(f"### ✅ AUTHENTIC  ·  confidence {(1 - result.confidence):.0%}")

    m1, m2, m3 = st.columns(3)
    m1.metric(signal_label,
              result.peak_votes if use_sift else result.n_matched_pairs)
    m2.metric("Keypoint/patch matches" if use_sift else "Mean pair score",
              result.n_matched_pairs if use_sift else f"{result.mean_similarity:.3f}")
    m3.metric("Forged regions (boxes)", len(result.boxes))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Input image**"); c1.image(result.original, use_container_width=True)
    c2.markdown("**Forgery heatmap**"); c2.image(result.heatmap_overlay, use_container_width=True)
    c3.markdown("**Localized regions / matches**"); c3.image(result.box_overlay, use_container_width=True)

    if result.is_forged and result.boxes:
        detail = f"**Detected region(s):** {result.boxes}"
        if use_sift:
            detail += f"  ·  dominant shift (dy, dx) = {result.dominant_offset}"
        else:
            detail += f"  ·  peak pair score = {result.max_similarity:.3f}"
        st.markdown(detail)

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
