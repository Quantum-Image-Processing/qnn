"""Siamese swap-test figure: how one patch pair is compared, on a copy-move example.
Run from the repo root:  python docs/figures/gen_siamese_swaptest.py
Output: docs/figures/phase3_siamese_swaptest.svg

The example uses a real CoMoFoD test scene and the trained 9-wire model
(models/zernike_9w.npz). One 8x8 patch is rotated by 45 degrees and pasted
elsewhere to illustrate a copy; every score printed in the figure is computed
here, and the two pair scores are checked against the real 9-wire circuit.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmfd_zernike_swaptest import ZernikeSwapQNN, zernike_state, image_states  # noqa: E402
from evaluate_zernike import load  # noqa: E402
from gen_zernike_modules import (Fig, INK, MUTED, TITLE, CREAM, CREAMB, GREY, GREYB,  # noqa: E402
                                 BLUE, BLUEB, TEAL, GREEN, GREENB, AMBER, AMBERB, QBLUE)

SCENE, SRC, DST, ANGLE = '026', (88, 40), (24, 88), 45

# ---------------------------------------------------------------- numbers
q = ZernikeSwapQNN(n_layers=3, img_size=128, seed=1)
q.load(ROOT / 'models' / 'zernike_9w.npz')
cal = json.load(open(ROOT / 'models' / 'zernike_9w_calibration.json'))
tau = cal['sim_threshold']
root = ROOT / 'data' / 'comofod'
data = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root
arr = load(data / f'{SCENE}_O.png')

y, x = SRC
crop = Image.fromarray(np.uint8(arr[y - 12:y + 20, x - 12:x + 20] * 255))
copy = np.asarray(crop.rotate(ANGLE, Image.BICUBIC), dtype=np.float64)[12:20, 12:20] / 255.0
src = arr[y:y + 8, x:x + 8].astype(np.float64)
forged = arr.copy(); forged[DST[0]:DST[0] + 8, DST[1]:DST[1] + 8] = copy

coords, states = image_states(arr)
S = q.all_pair_scores(states)
i_src = coords.index(SRC)
row = S[i_src].copy(); row[i_src] = np.nan
far = [k for k, (cy, cx) in enumerate(coords) if (abs(cy - y) > 24 or abs(cx - x) > 24)
       and 8 <= cy <= 112 and 8 <= cx <= 112 and (abs(cy - DST[0]) > 16 or abs(cx - DST[1]) > 16)]
k_oth = min(far, key=lambda k: abs(row[k] - np.nanmedian(row)))     # a typical other patch
oth = arr[coords[k_oth][0]:coords[k_oth][0] + 8, coords[k_oth][1]:coords[k_oth][1] + 8].astype(np.float64)

psi = {'i': zernike_state(src), 'j': zernike_state(copy), 'k': zernike_state(oth)}
wires = list(range(4)); w = np.asarray(q.weights, dtype=float)

@qml.qnode(qml.device('default.qubit', wires=4), diff_method=None)
def register(s):
    qml.AmplitudeEmbedding(s, wires=wires, normalize=True, pad_with=0.0)
    q._reupload_block(w, wires, s)
    return qml.state()

phi = {n: np.asarray(register(v)) for n, v in psi.items()}
s_ij = float(abs(np.vdot(phi['i'], phi['j'])) ** 2)
s_ik = float(abs(np.vdot(phi['i'], phi['k'])) ** 2)
real_ij = float(np.ravel(q.score(psi['i'][None], psi['j'][None]))[0])
real_ik = float(np.ravel(q.score(psi['i'][None], psi['k'][None]))[0])
assert abs(real_ij - s_ij) < 1e-6 and abs(real_ik - s_ik) < 1e-6, (real_ij, s_ij, real_ik, s_ik)
frac = float(np.mean(S[np.triu_indices(256, 1)] >= tau))
print(f'copy pair {s_ij:.3f} (circuit {real_ij:.3f}), other pair {s_ik:.3f} (circuit {real_ik:.3f}), '
      f'pairs >= {tau}: {frac:.2f}, other patch at {coords[k_oth]}')


def png(a, size):
    im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).resize((size, size), Image.NEAREST)
    b = io.BytesIO(); im.save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


# ---------------------------------------------------------------- figure
W, H = 1420, 1010
f = Fig(W, H, "Siamese swap test on a copy-move example: a source patch and its rotated copy get almost the "
              "same Zernike state, the two identical twin circuits with shared weights map them to almost the "
              "same quantum state, and the swap test on one ancilla measures their overlap as the similarity score.")
f.header("Phase 3 &#8212; Swap-test algorithm and Siamese QNN: how one patch pair is compared",
         f"a real CoMoFoD test scene ({SCENE}) and the trained 9-wire model; every number below is computed, not drawn", 3)

def img(xx, yy, size, a, px):
    f.o.append(f'<image x="{xx}" y="{yy}" width="{size}" height="{size}" href="{png(a, px)}"/>')

# 1 the problem
f.panel(36, 150, 300, 262, "1", "the problem: copy-move")
IX, IY, IS = 96, 166, 180
img(IX, IY, IS, forged, 720)
sc = IS / 128
for (cy, cx), col, lab in ((SRC, QBLUE, "i"), (DST, TITLE, "j"), (coords[k_oth], "#ffffff", "k")):
    f.box(IX + cx * sc - 2, IY + cy * sc - 2, 8 * sc + 4, 8 * sc + 4, "none", col, rx=1, sw=2.4)
    f.circ(IX + cx * sc - 4, IY + cy * sc - 4, 7.5, col if col != "#ffffff" else MUTED, "#ffffff", 1.2)
    f.t(IX + cx * sc - 4, IY + cy * sc - 0.5, lab, 10, "middle", "#ffffff", "bold")
f.t(186, 366, f"<tspan fill='{QBLUE}' font-weight='bold'>i</tspan> source &#183; "
              f"<tspan fill='{TITLE}' font-weight='bold'>j</tspan> copy, rotated {ANGLE}&#176; &#183; "
              f"<tspan fill='{MUTED}' font-weight='bold'>k</tspan> other", 10)
f.t(186, 384, "question for every pair: same content?", 10.5, "middle", INK, "bold")
f.t(186, 400, "(copy pasted here for illustration)", 9.5, "middle", MUTED)

# 2 Zernike states
f.panel(372, 150, 300, 262, "2", "from Phase 1: Zernike state per patch")
for r, (n, patch, col) in enumerate((('i', src, QBLUE), ('j', copy, TITLE), ('k', oth, GREYB))):
    yy = 170 + r * 68
    img(392, yy, 52, patch, 208)
    f.box(392, yy, 52, 52, "none", col, rx=0, sw=2)
    f.t(462, yy + 30, f"&#968;<tspan font-size='9'>{n}</tspan>", 12, "middle", col, "bold")
    for b, v in enumerate(psi[n][:12]):
        h = v * 52
        f.box(478 + b * 14, yy + 52 - h, 11, h, CREAM if n != 'k' else GREY, CREAMB if n != 'k' else GREYB, rx=0.5, sw=0.7)
f.t(522, 386, "12 Zernike magnitudes per patch", 9.5, "middle", MUTED)
f.t(522, 402, "&#968;<tspan font-size='8'>i</tspan> &#8776; &#968;<tspan font-size='8'>j</tspan> although j is rotated; &#968;<tspan font-size='8'>k</tspan> differs", 10.5, "middle", INK, "bold")

# 3 Siamese twins
f.panel(708, 150, 676, 262, "3", "Phase 3a: Siamese QNN, one circuit and one set of weights")
for r, (n, col) in enumerate((('i', QBLUE), ('j', TITLE))):
    yy = 186 + r * 128
    f.t(740, yy + 22, f"&#968;<tspan font-size='9'>{n}</tspan>", 13, "middle", col, "bold")
    f.arrow(756, yy + 18, 780, yy + 18)
    f.box(784, yy, 92, 36, AMBER, AMBERB, rx=3, sw=1.6)
    f.t(830, yy + 15, "AMP EMBED", 9.5, "middle", INK, "bold"); f.t(830, yy + 28, "4 qubits", 9, "middle", INK)
    f.arrow(876, yy + 18, 896, yy + 18)
    f.box(900, yy, 128, 36, GREEN, "#3d7a3d", rx=3, sw=1.6)
    f.t(964, yy + 15, "U(w, &#968;) &#215; 3 layers", 10, "middle", INK, "bold")
    f.t(964, yy + 28, "RX RY RZ + CNOT ring", 9, "middle", INK)
    f.arrow(1028, yy + 18, 1050, yy + 18)
    f.t(1074, yy + 22, f"|&#966;<tspan font-size='9'>{n}</tspan>&#10217;", 13, "middle", col, "bold")
    p = np.abs(phi[n]) ** 2
    for b, v in enumerate(p):
        h = v / p.max() * 34 if p.max() > 0 else 0
        f.box(1100 + b * 15, yy + 36 - h, 12, h, BLUE, BLUEB, rx=0.5, sw=0.7)
    if n == 'i':
        f.t(1220, yy + 50, "16 measured probabilities of the state", 9, "middle", MUTED)
f.box(904, 250, 120, 40, "#ffffff", "#3d7a3d", rx=3, sw=1.6, dash="5 3")
f.t(964, 267, "shared w", 11, "middle", "#3d7a3d", "bold")
f.t(964, 282, "36 trained angles", 9.5, "middle", "#3d7a3d")
f.line(964, 222, 964, 250, "#3d7a3d", 1.6, "4 3"); f.line(964, 290, 964, 314, "#3d7a3d", 1.6, "4 3")
f.t(1046, 365, "same circuit and same weights on both sides, so a copy gives (almost) the same state", 10.5, "middle", INK, "bold")
f.t(1046, 383, "and the score is symmetric: S(i, j) = S(j, i). The data is re-uploaded in every layer: angle = w + 2&#960;&#183;&#968;", 9.8, "middle", MUTED)
f.t(1046, 399, "(without re-uploading, the shared weights would cancel out in the swap test and nothing could be trained)", 9.8, "middle", MUTED)

# 4 swap test step by step
f.panel(36, 452, 1348, 300, "4", "Phase 3b: the swap-test algorithm on one ancilla, step by step")
ya, yA, yB = 520, 590, 650
XS, XE = 130, 600
for yy, lab, col in ((ya, "ancilla |0&#10217;", QBLUE), (yA, f"|&#966;<tspan font-size='9'>i</tspan>&#10217;  (q0&#8211;q3)", QBLUE),
                     (yB, f"|&#966;<tspan font-size='9'>j</tspan>&#10217;  (q4&#8211;q7)", TITLE)):
    f.line(XS, yy, XE, yy, QBLUE if yy == ya else INK, 2.2 if yy == ya else 3.4)
    f.t(XS - 8, yy + 4, lab, 11, "end", col, "bold")
for yy in (yA, yB):
    f.line(XS + 18, yy + 8, XS + 28, yy - 8, INK, 1.4); f.t(XS + 32, yy - 10, "4", 9, "start", INK)
f.box(220, ya - 14, 28, 28, "#ffffff", QBLUE, rx=3, sw=2.2); f.t(234, ya + 5, "H", 13, "middle", QBLUE, "bold")
xc = 340
f.line(xc, ya, xc, yB, QBLUE, 2); f.circ(xc, ya, 5, QBLUE)
for yy in (yA, yB):
    f.line(xc - 8, yy - 8, xc + 8, yy + 8, QBLUE, 2.4); f.line(xc + 8, yy - 8, xc - 8, yy + 8, QBLUE, 2.4)
f.t(xc, yB + 30, "CSWAP: swap the whole registers", 9.5, "middle", QBLUE, "bold")
f.t(xc, yB + 44, "(4 CSWAP gates, one per qubit pair)", 9, "middle", MUTED)
f.box(440, ya - 14, 28, 28, "#ffffff", QBLUE, rx=3, sw=2.2); f.t(454, ya + 5, "H", 13, "middle", QBLUE, "bold")
f.box(520, ya - 16, 44, 32, QBLUE, "#08316b", rx=3, sw=1.6)
f.o.append(f'<path d="M 530 {ya + 9} A 12 12 0 0 1 554 {ya + 9}" fill="none" stroke="#ffffff" stroke-width="2"/>')
f.line(542, ya + 9, 554, ya - 8, "#ffffff", 2)
for n, xx in (("1", 190), ("2", 290), ("3", 400), ("4", 495)):
    f.line(xx, ya - 30, xx, yB + 14, GREYB, 1, "3 3"); f.badge(xx, ya - 38, n, MUTED, 9)

EX = 640
steps = [("1", "start", f"|0&#10217; &#8855; |&#966;<tspan font-size='9'>i</tspan>&#10217;|&#966;<tspan font-size='9'>j</tspan>&#10217;"),
         ("2", "after H", f"(1/&#8730;2) (|0&#10217; + |1&#10217;) &#8855; |&#966;<tspan font-size='9'>i</tspan>&#10217;|&#966;<tspan font-size='9'>j</tspan>&#10217;"),
         ("3", "after CSWAP", f"(1/&#8730;2) ( |0&#10217;|&#966;<tspan font-size='9'>i</tspan>&#10217;|&#966;<tspan font-size='9'>j</tspan>&#10217; + |1&#10217;|&#966;<tspan font-size='9'>j</tspan>&#10217;|&#966;<tspan font-size='9'>i</tspan>&#10217; )   the |1&#10217; branch holds the swapped pair"),
         ("4", "after H", f"&#189; |0&#10217;( |&#966;<tspan font-size='9'>i</tspan>&#966;<tspan font-size='9'>j</tspan>&#10217; + |&#966;<tspan font-size='9'>j</tspan>&#966;<tspan font-size='9'>i</tspan>&#10217; ) + &#189; |1&#10217;( |&#966;<tspan font-size='9'>i</tspan>&#966;<tspan font-size='9'>j</tspan>&#10217; &#8722; |&#966;<tspan font-size='9'>j</tspan>&#966;<tspan font-size='9'>i</tspan>&#10217; )")]
for r, (n, lab, eq) in enumerate(steps):
    yy = 506 + r * 34
    f.badge(EX + 10, yy - 4, n, MUTED, 9)
    f.t(EX + 28, yy, lab, 10.5, "start", INK, "bold")
    f.t(EX + 120, yy, eq, 11, "start", INK)
f.box(EX, 640, 728, 92, BLUE, BLUEB, rx=3)
f.t(EX + 364, 664, f"measure the ancilla:   P(0) = &#189; (1 + |&#10216;&#966;<tspan font-size='9'>i</tspan>|&#966;<tspan font-size='9'>j</tspan>&#10217;|&#178;)", 12, "middle", QBLUE, "bold")
f.t(EX + 364, 690, f"score = &#10216;Z&#10217; = P(0) &#8722; P(1) = |&#10216;&#966;<tspan font-size='9'>i</tspan>|&#966;<tspan font-size='9'>j</tspan>&#10217;|&#178;  &#8712; [0, 1]", 13, "middle", QBLUE, "bold")
f.t(EX + 364, 716, "identical states: the |1&#10217; branch cancels, the ancilla always reads 0, score 1.  Unrelated states: score near 0.", 10, "middle", INK)
f.t(330, 730, "on hardware, &#10216;Z&#10217; is estimated from repeated runs (shots)", 9.5, "middle", MUTED)

# 5 results for this example
f.panel(36, 790, 1348, 196, "5", "result for this example, with the trained model and the calibrated threshold")
BW = 420
for c, (title, s, sc_real, hit, col) in enumerate((
        (f"pair (i, j): source and its {ANGLE}&#176; copy", s_ij, real_ij, s_ij >= tau, TITLE),
        ("pair (i, k): a typical other patch", s_ik, real_ik, s_ik >= tau, GREYB))):
    xx = 60 + c * (BW + 24)
    f.box(xx, 810, BW, 92, GREEN if hit else "#ffffff", GREENB if hit else GREYB, rx=3)
    f.t(xx + BW / 2, 834, title, 11, "middle", INK, "bold")
    f.t(xx + BW / 2, 862, f"score &#10216;Z&#10217; = {s:.3f}   {'&#8805;' if hit else '&lt;'} {tau}", 15, "middle", INK, "bold")
    f.t(xx + BW / 2, 888, ("MATCH: both patches are painted in the mask" if hit else "no match"), 11, "middle",
        TITLE if hit else MUTED, "bold")
xx = 60 + 2 * (BW + 24)
f.box(xx, 810, 1360 - xx, 92, CREAM, CREAMB, rx=3)
f.t(xx + (1360 - xx) / 2, 834, "then Phase 4", 11, "middle", INK, "bold")
f.t(xx + (1360 - xx) / 2, 858, "this comparison is repeated for all", 10.5, "middle", INK)
f.t(xx + (1360 - xx) / 2, 876, "32,640 pairs of the image, and the", 10.5, "middle", INK)
f.t(xx + (1360 - xx) / 2, 894, "number of matches gives the verdict", 10.5, "middle", INK)
f.t(710, 930, "Both scores were checked against a full run of the 9-wire circuit (same values). The pair is judged on its own: in this image "
              f"{frac * 100:.0f}% of all pairs also reach {tau},", 10.2, "middle", MUTED)
f.t(710, 948, "mostly smooth patches that resemble each other, which is why the image-level verdict is not yet reliable (Phase 4).", 10.2, "middle", MUTED)
f.t(710, 970, "Copies are recognised; telling a real copy from a natural look-alike is the open problem.", 10.2, "middle", TITLE, "bold")
f.save(str(ROOT / 'docs' / 'figures' / 'phase3_siamese_swaptest.svg'))
