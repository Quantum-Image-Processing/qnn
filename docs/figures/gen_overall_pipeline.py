"""Pictorial overview of the nine ST-QNN stages (the Overall System figure).
Run from the repo root:  python docs/figures/gen_overall_pipeline.py
Output: docs/figures/overall_pipeline.svg

Uses CoMoFoD scene 026 (a car copied within a car park) for every picture,
and its ground-truth mask for the two copy-move regions. The output panel is
a schematic; measured ST-QNN results are in the paper's results section.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_zernike_modules import (Fig, INK, MUTED, TITLE, CREAM, CREAMB, GREY, GREYB,  # noqa: E402
                                 BLUE, BLUEB, PINK, TEAL, GREEN, GREENB, AMBER, AMBERB, QBLUE)

PURPLE = "#7b1fa2"
SCENE = '026'
root = ROOT / 'data' / 'comofod'
data = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root
rgb = Image.open(data / f'{SCENE}_F.png').convert('RGB')
gray = np.asarray(Image.open(data / f'{SCENE}_F.png').convert('L').resize((128, 128), Image.LANCZOS),
                  dtype=np.float64) / 255.0
gt = np.asarray(Image.open(data / f'{SCENE}_B.png').convert('L')) > 128
PA, PB = (48, 48), (40, 96)          # two 8x8 patches, one in each ground-truth region
patch = gray[PA[0]:PA[0] + 8, PA[1]:PA[1] + 8]
psi = patch.ravel() / np.linalg.norm(patch)


def uri(im):
    b = io.BytesIO(); im.save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def arr_img(a, size):
    return Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).resize((size, size), Image.NEAREST)


W, H = 1420, 760
f = Fig(W, H, "Overall system of ST-QNN: nine stages in three groups. Classical, once per image: input image, "
              "image pre-processing, patch extraction and feature extraction. Quantum, one circuit call per "
              "admissible pair: quantum feature encoding, Siamese QNN, swap test and spatial offset consistency. "
              "Classical, after all pairs are scored: patch matching and forgery detection.")
f.t(W / 2, 34, "ST-QNN overall system &#8212; nine stages, grouped by where they run", 15, "middle", TITLE, "bold")
f.t(W / 2, 54, f"every picture is CoMoFoD scene {SCENE}, a car copied within the same car park", 11, "middle", MUTED)


def img(x, y, w, h, im):
    f.o.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri(im)}"/>')


def band(y, h, label, fill, stroke, col):
    f.box(20, y, W - 40, h, fill, stroke, rx=8, sw=1.2)
    f.t(36, y + 18, label, 11, "start", col, "bold")


PW, PH, GAP = 318, 150, 22
XS = [40 + i * (PW + GAP) for i in range(4)]


def panel(x, y, n, head, w=PW):
    f.box(x, y, w, PH, "#ffffff", GREYB)
    f.badge(x + 14, y + 14, n, INK, 10)
    f.t(x + 32, y + 19, head, 11.5, "start", INK, "bold")


def arrows(xs, y, w=PW, double_last=False):
    for i in range(len(xs) - 1):
        x1, x2 = xs[i] + w + 3, xs[i + 1] - 3
        if double_last and i == len(xs) - 2:
            f.o.append(f'<line x1="{x1 + 2}" y1="{y}" x2="{x2}" y2="{y}" stroke="{INK}" stroke-width="1.4" '
                       'marker-start="url(#a)" marker-end="url(#a)"/>')
        else:
            f.arrow(x1, y, x2, y)


# ------------------------------------------------ band 1: classical, once per image
Y1 = 72
band(Y1, 196, "CLASSICAL &#183; once per image", "#f4f4f4", "#d0d0d0", MUTED)
py = Y1 + 32
panel(XS[0], py, "1", "Input image")
th = rgb.resize((104, 104), Image.LANCZOS)
img(XS[0] + 20, py + 34, 104, 104, th)
s = 104 / 512
ys, xs = np.where(gt)
from scipy.ndimage import label as cc_label  # noqa: E402
lab, n = cc_label(gt)
for k in range(1, n + 1):
    yy, xx = np.where(lab == k)
    f.box(XS[0] + 20 + xx.min() * s - 1, py + 34 + yy.min() * s - 1, (xx.max() - xx.min()) * s + 2,
          (yy.max() - yy.min()) * s + 2, "none", TITLE, rx=1, sw=1.8)
f.t(XS[0] + 136, py + 62, "JPEG or PNG,", 9.8, "start", MUTED)
f.t(XS[0] + 136, py + 78, "any size, colour", 9.8, "start", MUTED)
f.t(XS[0] + 136, py + 104, "red: the copied car", 9.8, "start", TITLE)
f.t(XS[0] + 136, py + 120, "and its source", 9.8, "start", TITLE)
f.t(XS[0] + 136, py + 136, "(ground truth)", 9.8, "start", MUTED)

panel(XS[1], py, "2", "Image pre-processing")
img(XS[1] + 20, py + 34, 104, 104, arr_img(gray, 416))
f.t(XS[1] + 136, py + 62, "grayscale (BT.601)", 9.8, "start", MUTED)
f.t(XS[1] + 136, py + 80, "Lanczos resize to", 9.8, "start", MUTED)
f.t(XS[1] + 136, py + 96, "<tspan font-weight='bold' fill='" + INK + "'>128 &#215; 128</tspan>", 10.5, "start")
f.t(XS[1] + 136, py + 120, "pixels scaled to [0, 1]", 9.8, "start", MUTED)

panel(XS[2], py, "3", "Patch extraction")
gx, gy, gs = XS[2] + 20, py + 34, 104
img(gx, gy, gs, gs, arr_img(gray, 416))
for k in range(17):
    f.line(gx + k * gs / 16, gy, gx + k * gs / 16, gy + gs, "#ffffff", 0.45)
    f.line(gx, gy + k * gs / 16, gx + gs, gy + k * gs / 16, "#ffffff", 0.45)
for (yy, xx), col in ((PA, QBLUE), (PB, TITLE)):
    f.box(gx + xx * gs / 128, gy + yy * gs / 128, gs / 16, gs / 16, "none", col, rx=0, sw=2)
f.line(gx + (PA[1] + 4) * gs / 128, gy + (PA[0] + 4) * gs / 128, gx + (PB[1] + 4) * gs / 128,
       gy + (PB[0] + 4) * gs / 128, "#ffd54f", 1.4, "3 2")
f.t(XS[2] + 136, py + 62, "16 &#215; 16 tiles of 8 &#215; 8", 9.8, "start", MUTED)
f.t(XS[2] + 136, py + 80, "stride 8, no overlap", 9.8, "start", MUTED)
f.t(XS[2] + 136, py + 104, "<tspan font-weight='bold' fill='" + INK + "'>256 patches</tspan>", 10.5, "start")
f.t(XS[2] + 136, py + 122, "<tspan font-weight='bold' fill='" + INK + "'>32,640 pairs</tspan>", 10.5, "start")

panel(XS[3], py, "4", "Feature extraction")
img(XS[3] + 20, py + 40, 62, 62, arr_img(patch, 248))
f.box(XS[3] + 20, py + 40, 62, 62, "none", QBLUE, rx=0, sw=2)
f.t(XS[3] + 51, py + 118, "8 &#215; 8 patch", 9.5, "middle", MUTED)
f.arrow(XS[3] + 88, py + 71, XS[3] + 108, py + 71)
bx, base = XS[3] + 114, py + 102
for i, v in enumerate(psi):
    h = v / psi.max() * 56
    f.o.append(f'<rect x="{bx + i * 3}" y="{base - h:.1f}" width="2.4" height="{h:.1f}" fill="{CREAMB}"/>')
f.t(bx + 96, py + 118, "64 values", 9.5, "middle", MUTED)
f.t(XS[3] + 90, py + 138, "&#968; = x / &#8214;x&#8214;", 10, "middle", INK, "bold")
f.t(XS[3] + 222, py + 138, "64 = 2&#8310; &#8594; <tspan font-weight='bold'>6 qubits</tspan>", 10, "middle", INK)

arrows(XS, py + PH / 2)

# ------------------------------------------------ band 2: quantum, one call per pair
Y2 = Y1 + 196 + 40
f.arrow(W / 2, Y1 + 196 + 4, W / 2, Y2 - 4, QBLUE, 2)
f.t(W / 2 + 12, Y1 + 196 + 26, "for each admissible pair of patches: one circuit call", 10, "start", QBLUE, "bold")
band(Y2, 196, "QUANTUM CIRCUIT &#183; one call per admissible pair (32,640 per image) &#183; 15 wires", "#eef4fb", BLUEB, QBLUE)
py = Y2 + 32

panel(XS[0], py, "5", "Quantum feature encoding")
for r, ((yy, xx), col, name) in enumerate(((PA, QBLUE, "A"), (PB, TITLE, "B"))):
    ty = py + 36 + r * 56
    img(XS[0] + 16, ty, 44, 44, arr_img(gray[yy:yy + 8, xx:xx + 8], 176))
    f.box(XS[0] + 16, ty, 44, 44, "none", col, rx=0, sw=2)
    f.arrow(XS[0] + 64, ty + 22, XS[0] + 84, ty + 22)
    f.box(XS[0] + 88, ty + 2, 50, 40, AMBER, AMBERB, rx=3, sw=1.4)
    f.t(XS[0] + 113, ty + 19, "AMP", 9, "middle", INK, "bold"); f.t(XS[0] + 113, ty + 31, "EMBED", 9, "middle", INK, "bold")
    for q in range(6):
        f.line(XS[0] + 138, ty + 5 + q * 6.8, XS[0] + 176, ty + 5 + q * 6.8, INK, 1)
    f.t(XS[0] + 184, ty + 18, f"register {name}", 9.8, "start", col, "bold")
    f.t(XS[0] + 184, ty + 32, f"6 qubits, |&#968;<tspan font-size='8'>{name}</tspan>&#10217;", 9.5, "start", MUTED)

panel(XS[1], py, "6", "Siamese QNN")
for r, col in enumerate((QBLUE, TITLE)):
    ty = py + 40 + r * 56
    for q in range(3):
        for g in range(3):
            f.box(XS[1] + 18 + g * 24, ty + q * 13, 21, 11, GREEN, "#3d7a3d", rx=1.5, sw=1)
    for q in range(3):
        f.line(XS[1] + 92, ty + 5 + q * 13, XS[1] + 120, ty + 5 + q * 13, INK, 1)
    for q in range(2):
        f.line(XS[1] + 98 + q * 9, ty + 5 + q * 13, XS[1] + 98 + q * 9, ty + 5 + (q + 1) * 13, INK, 1)
        f.circ(XS[1] + 98 + q * 9, ty + 5 + q * 13, 2, INK)
        f.circ(XS[1] + 98 + q * 9, ty + 5 + (q + 1) * 13, 4, "#ffffff", INK, 1)
    f.t(XS[1] + 128, ty + 20, "&#215;3", 10, "start", INK, "bold")
f.line(XS[1] + 54, py + 80, XS[1] + 54, py + 96, "#3d7a3d", 1.6, "4 3")
f.t(XS[1] + 168, py + 66, "tied weights &#952;:", 9.8, "start", "#3d7a3d", "bold")
f.t(XS[1] + 168, py + 82, "same 54 angles on", 9.8, "start", MUTED)
f.t(XS[1] + 168, py + 96, "both registers;", 9.8, "start", MUTED)
f.t(XS[1] + 168, py + 112, "pixels re-uploaded", 9.8, "start", MUTED)
f.t(XS[1] + 168, py + 126, "in every layer", 9.8, "start", MUTED)

panel(XS[2], py, "7", "Swap test")
ax, ay = XS[2] + 22, py + 52
f.line(ax, ay, ax + 200, ay, QBLUE, 2)
f.t(ax, ay - 16, "ancilla", 9, "start", QBLUE, "bold")
for yy in (ay + 38, ay + 66):
    f.line(ax, yy, ax + 200, yy, INK, 3)
f.box(ax + 22, ay - 11, 22, 22, "#ffffff", QBLUE, rx=2, sw=1.8); f.t(ax + 33, ay + 4, "H", 11, "middle", QBLUE, "bold")
f.line(ax + 80, ay, ax + 80, ay + 66, QBLUE, 1.8); f.circ(ax + 80, ay, 4, QBLUE)
for yy in (ay + 38, ay + 66):
    f.line(ax + 74, yy - 6, ax + 86, yy + 6, QBLUE, 2); f.line(ax + 86, yy - 6, ax + 74, yy + 6, QBLUE, 2)
f.box(ax + 116, ay - 11, 22, 22, "#ffffff", QBLUE, rx=2, sw=1.8); f.t(ax + 127, ay + 4, "H", 11, "middle", QBLUE, "bold")
f.box(ax + 156, ay - 13, 34, 26, QBLUE, "#08316b", rx=2, sw=1.2)
f.o.append(f'<path d="M {ax + 163} {ay + 7} A 10 10 0 0 1 {ax + 183} {ay + 7}" fill="none" stroke="#ffffff" stroke-width="1.6"/>')
f.line(ax + 173, ay + 7, ax + 182, ay - 6, "#ffffff", 1.6)
f.t(ax + 210, ay + 42, "A", 9.5, "start", QBLUE, "bold"); f.t(ax + 210, ay + 70, "B", 9.5, "start", TITLE, "bold")
f.t(XS[2] + PW / 2, py + 138, "&#10216;Z&#10217; &#8776; |&#10216;&#968;<tspan font-size='8'>A</tspan>|&#968;<tspan font-size='8'>B</tspan>&#10217;|&#178; : the similarity score", 10, "middle", INK, "bold")

panel(XS[3], py, "8", "Spatial offset consistency")
gx, gy, gs = XS[3] + 18, py + 34, 88
for k in range(9):
    f.line(gx + k * gs / 8, gy, gx + k * gs / 8, gy + gs, GREYB, 0.5)
    f.line(gx, gy + k * gs / 8, gx + gs, gy + k * gs / 8, GREYB, 0.5)
ca, cb = (6, 3), (5, 6)
f.box(gx + ca[1] * 11, gy + ca[0] * 11, 11, 11, PINK, QBLUE, rx=0, sw=1.6)
f.box(gx + cb[1] * 11, gy + cb[0] * 11, 11, 11, PINK, TITLE, rx=0, sw=1.6)
f.line(gx + ca[1] * 11 + 5.5, gy + ca[0] * 11 + 5.5, gx + cb[1] * 11 + 5.5, gy + cb[0] * 11 + 5.5, TITLE, 1.4, "3 2")
f.t(gx + gs / 2, gy + gs + 14, "&#916; = (dy, dx)", 9.5, "middle", TITLE, "bold")
f.box(XS[3] + 124, py + 46, 108, 26, PURPLE, "#4a0e5c", rx=3, sw=1.2)
f.t(XS[3] + 178, py + 63, "RY(&#966; + &#916;&#183;&#960;)", 9.5, "middle", "#ffffff", "bold")
f.box(XS[3] + 124, py + 82, 108, 26, PURPLE, "#4a0e5c", rx=3, sw=1.2)
f.t(XS[3] + 178, py + 99, "CRY &#8594; ancilla", 9.5, "middle", "#ffffff", "bold")
f.t(XS[3] + 240, py + 60, "2 offset", 9.5, "start", MUTED); f.t(XS[3] + 240, py + 74, "qubits,", 9.5, "start", MUTED)
f.t(XS[3] + 240, py + 88, "4 angles", 9.5, "start", MUTED)
f.t(XS[3] + 178, py + 128, "folded into the same readout", 9.8, "middle", MUTED)

arrows(XS, py + PH / 2, double_last=True)

# ------------------------------------------------ band 3: classical, after all pairs
Y3 = Y2 + 196 + 40
f.arrow(W / 2, Y2 + 196 + 4, W / 2, Y3 - 4, GREENB, 2)
f.t(W / 2 + 12, Y2 + 196 + 26, "after all 32,640 pairs are scored", 10, "start", "#3d7a3d", "bold")
band(Y3, 196, "CLASSICAL &#183; after all pairs are scored", "#f1f8f1", GREENB, "#3d7a3d")
py = Y3 + 32
X3 = [40 + i * (PW + GAP) for i in range(3)]
W3 = [PW, PW, 2 * PW + GAP]

panel(X3[0], py, "9", "Patch matching")
for i, h in enumerate([10, 18, 30, 46, 62, 50, 34, 22, 14]):
    col = TEAL if i >= 6 else GREY
    f.o.append(f'<rect x="{X3[0] + 24 + i * 20}" y="{py + 118 - h}" width="17" height="{h}" fill="{col}" stroke="{GREYB}" stroke-width="0.6"/>')
f.line(X3[0] + 24 + 6 * 20 - 2, py + 44, X3[0] + 24 + 6 * 20 - 2, py + 122, TITLE, 1.5, "4 3")
f.t(X3[0] + 24 + 6 * 20 + 2, py + 42, "&#964; = 0.90", 10, "start", TITLE, "bold")
f.t(X3[0] + 212, py + 74, "keep pairs", 9.8, "start", MUTED)
f.t(X3[0] + 212, py + 90, "with &#10216;Z&#10217; &#8805; &#964;", 9.8, "start", MUTED)
f.t(X3[0] + PW / 2, py + 140, "32,640 circuit scores", 9.5, "middle", MUTED)

panel(X3[1], py, "10", "Forgery detection")
f.o.append(f'<rect x="{X3[1] + 20}" y="{py + 36}" width="104" height="104" fill="#111111"/>')
for (yy, xx) in (PA, PB):
    f.o.append(f'<rect x="{X3[1] + 20 + xx * 104 / 128}" y="{py + 36 + yy * 104 / 128}" width="{104 / 16 * 2}" height="{104 / 16 * 2}" fill="#ffffff"/>')
f.box(X3[1] + 138, py + 44, 166, 28, "#ffffff", GREENB, rx=3)
f.t(X3[1] + 221, py + 63, "M &#8805; M<tspan font-size='8'>0</tspan> = 8 &#8594; <tspan font-weight='bold' fill='" + TITLE + "'>FORGED</tspan>", 10.5)
f.box(X3[1] + 138, py + 78, 166, 28, "#ffffff", GREENB, rx=3)
f.t(X3[1] + 221, py + 97, "M &lt; M<tspan font-size='8'>0</tspan> &#8594; <tspan font-weight='bold'>AUTHENTIC</tspan>", 10.5)
f.t(X3[1] + 221, py + 126, "paint both patches of", 9.5, "middle", MUTED)
f.t(X3[1] + 221, py + 140, "every match (schematic mask)", 9.5, "middle", MUTED)

x0, w0 = X3[2], W3[2]
f.box(x0, py, w0, PH, GREEN, GREENB)
f.badge(x0 + 14, py + 14, "&#10003;", "#3d7a3d", 10)
f.t(x0 + 32, py + 19, "Output to the user", 11.5, "start", INK, "bold")
items = [("verdict", ("FORGED or", "AUTHENTIC"), TITLE), ("heatmap", ("brighter =", "stronger evidence"), INK),
         ("mask", ("128 &#215; 128, both", "halves of the copy"), INK), ("boxes", ("connected components,", "display only"), INK)]
for i, (k, (v1, v2), col) in enumerate(items):
    cx = x0 + 22 + i * 156
    f.box(cx, py + 38, 144, 70, "#ffffff", GREENB, rx=3)
    f.t(cx + 72, py + 60, k, 11, "middle", col, "bold")
    f.t(cx + 72, py + 80, v1, 9, "middle", MUTED)
    f.t(cx + 72, py + 94, v2, 9, "middle", MUTED)
f.t(x0 + w0 / 2, py + 132, "measured ST-QNN results are reported in the results section; this panel shows only what is returned", 9.5, "middle", MUTED)

arrows(X3[:2], py + PH / 2)
f.arrow(X3[1] + PW + 3, py + PH / 2, X3[2] - 3, py + PH / 2)
f.save(str(ROOT / 'docs' / 'figures' / 'overall_pipeline.svg'))
