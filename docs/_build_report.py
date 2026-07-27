#!/usr/bin/env python3
"""Generate the CMFD Phase-2 progress report as a self-contained HTML file
(charts embedded as base64 PNGs). Rendered to PDF by headless Chrome."""
import base64, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- palette (validated light-mode values from the dataviz skill) ----
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK2      = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
BASELINE  = '#c3c2b7'
BLUE      = '#2a78d6'   # categorical slot 1
ORANGE    = '#eb6834'   # categorical slot 2

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': BASELINE,
})

def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(BASELINE)
    ax.spines['bottom'].set_color(BASELINE)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ============ CHART 1: precision-recall gap (the flooding signature) ========
imgs   = ['COVERAGE\n1t', 'COVERAGE\n10t', 'COVERAGE\n100t', 'CoMoFoD\n001_F', 'CoMoFoD\n050_F']
recall = [1.000, 1.000, 1.000, 0.806, 1.000]
prec   = [0.269, 0.055, 0.337, 0.017, 0.111]

fig, ax = plt.subplots(figsize=(7.6, 3.5))
x = range(len(imgs)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], recall, w, label='Recall', color=BLUE, zorder=3)
b2 = ax.bar([i + w/2 for i in x], prec,   w, label='Precision', color=ORANGE, zorder=3)
style(ax)
ax.set_ylim(0, 1.12); ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xticks(list(x)); ax.set_xticklabels(imgs, fontsize=9.5)
ax.set_ylabel('score')
ax.set_title('Recall is near-perfect while Precision collapses — the over-detection signature',
             fontsize=11, color=INK, pad=10, loc='left')
for bars, vals in ((b1, recall), (b2, prec)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8.5, color=INK2)
ax.legend(frameon=False, loc='upper right', fontsize=9.5, ncol=2)
CHART1 = to_b64(fig)

# ============ CHART 2: offset filter de-flooding (pairs kept) ================
imgs2  = ['1t', '10t', '100t']
before = [7984, 2466, 2001]
after  = [229, 43, 86]

fig, ax = plt.subplots(figsize=(7.6, 3.3))
x = range(len(imgs2)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], before, w, label='Before tuning (thr 0.8, votes 3)', color=BLUE, zorder=3)
b2 = ax.bar([i + w/2 for i in x], after,  w, label='After tuning (thr 0.92, votes 8)', color=ORANGE, zorder=3)
style(ax)
ax.set_ylim(0, 8800)
ax.set_xticks(list(x)); ax.set_xticklabels([f'COVERAGE {i}' for i in imgs2], fontsize=9.5)
ax.set_ylabel('patch pairs kept as forged')
ax.set_title('Tightening parameters cuts surviving false matches by ~95%',
             fontsize=11, color=INK, pad=10, loc='left')
for bars, vals in ((b1, before), (b2, after)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v+90, f'{v:,}',
                ha='center', va='bottom', fontsize=8.5, color=INK2)
ax.legend(frameon=False, loc='upper right', fontsize=9)
CHART2 = to_b64(fig)

# ============================ HTML =========================================
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CMFD Phase-2 Progress Report</title>
<style>
  @page { size: A4; margin: 18mm 16mm; }
  * { box-sizing: border-box; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-family: 'DejaVu Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
         color: #0b0b0b; font-size: 10.5pt; line-height: 1.5; margin: 0; }
  h1 { font-size: 22pt; margin: 0 0 2pt; line-height: 1.15; }
  h2 { font-size: 14pt; margin: 22pt 0 6pt; padding-bottom: 3pt;
       border-bottom: 2px solid #2a78d6; color: #10203a; }
  h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #10203a; }
  p { margin: 0 0 7pt; }
  .sub { color: #52514e; font-size: 11pt; margin-top: 4pt; }
  .meta { color: #898781; font-size: 9pt; margin-top: 10pt; }
  .cover { border-left: 5px solid #2a78d6; padding: 4pt 0 4pt 14pt; margin-bottom: 6pt; }
  table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
          font-size: 9.3pt; font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid #e1e0d9; padding: 4pt 7pt; text-align: left; vertical-align: top; }
  th { background: #f2f6fc; color: #10203a; font-weight: 600; }
  td.n, th.n { text-align: right; }
  tr.tot td { font-weight: 700; background: #fafaf8; }
  .good { color: #0a7d0a; font-weight: 600; }
  .bad  { color: #c02626; font-weight: 600; }
  .warn { color: #b5730a; font-weight: 600; }
  .callout { background: #f7f9fc; border: 1px solid #dbe6f5; border-left: 4px solid #2a78d6;
             padding: 8pt 12pt; margin: 8pt 0; border-radius: 3px; }
  .callout.red { background: #fdf6f6; border-color: #f0d5d5; border-left-color: #d03b3b; }
  figure { margin: 8pt 0 12pt; }
  figure img { width: 100%; border: 1px solid #e1e0d9; border-radius: 4px; }
  figcaption { font-size: 8.6pt; color: #898781; margin-top: 4pt; }
  code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt;
         background: #f2f2ef; padding: 0.5pt 3pt; border-radius: 3px; }
  pre { background: #f6f6f4; border: 1px solid #e1e0d9; border-radius: 4px;
        padding: 8pt 10pt; font-family: 'DejaVu Sans Mono', monospace;
        font-size: 8.6pt; line-height: 1.45; overflow-x: auto; white-space: pre-wrap; }
  ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
  li { margin-bottom: 3pt; }
  .pill { display: inline-block; font-size: 8pt; font-weight: 700; padding: 1pt 6pt;
          border-radius: 9px; color: #fff; }
  .pill.done { background: #0a7d0a; } .pill.no { background: #c02626; }
  .pill.half { background: #b5730a; }
  .avoid-break { break-inside: avoid; }
  .pb { break-before: page; }
  footer { margin-top: 16pt; padding-top: 6pt; border-top: 1px solid #e1e0d9;
           color: #898781; font-size: 8pt; }
</style></head><body>

<div class="cover">
  <h1>Copy-Move Forgery Detection using a<br>Siamese Swap-Test Quantum Neural Network</h1>
  <div class="sub">Phase-2 Progress &amp; Evaluation Report</div>
  <div class="meta">Final Year Project &middot; Phase 2 &nbsp;|&nbsp; Date: 26 July 2026 &nbsp;|&nbsp;
    Pipeline: <code>cmfd_swaptest_pipeline.py</code> (v2)</div>
</div>

<h2>1. Executive Summary</h2>
<p>This report documents the current state of a quantum-machine-learning pipeline for
<b>copy-move forgery detection (CMFD)</b> &mdash; detecting regions of an image that have been
duplicated and pasted elsewhere in the same image. The system pairs a classical
patch descriptor with a <b>Siamese swap-test quantum neural network (QNN)</b> that learns a
similarity score between two image patches, followed by a spatial offset-consistency
filter that keeps only genuine duplications.</p>
<div class="callout red">
  <b>Headline finding.</b> The infrastructure (data loading, patch extraction, QNN similarity,
  metrics, multi-dataset evaluation) is <b>sound and runs end-to-end</b>. However, the model
  currently <b>over-detects severely</b> &mdash; it flags 50&ndash;97% of each test image as forged.
  Consequently the previously-remembered &ldquo;~80%&rdquo; figure was <b>recall, not accuracy</b>; true
  pixel accuracy on those samples is ~3&ndash;12%, and localization quality (F1/IoU) is low.
  The remaining work is <b>detection quality</b>, not plumbing.
</div>

<h2>2. Objective &amp; Target Output</h2>
<p>The intended end product: <i>given any input image, decide whether it is forged, and if so,
localize the exact duplicated region as a bounding box or heatmap.</i> That decomposes into
four capabilities, tracked in the gap analysis (&sect;8).</p>

<h2>3. System Architecture</h2>
<p>The architecture is a <b>pairwise similarity model</b>, which is the structurally correct choice
for copy-move (the task is inherently &ldquo;does patch A match patch B elsewhere?&rdquo;). A single-patch
classifier cannot express duplication. An earlier degenerate embedding (fixed amplitude
embedding + data-independent tied unitary, which the swap test provably cancels) was replaced
with a <b>data re-uploading</b> ansatz so the tied weights have a genuine effect on the measured
overlap.</p>
<table class="avoid-break">
  <tr><th style="width:34%">Component</th><th>Specification</th></tr>
  <tr><td>Quantum registers</td><td><b>4 qubits per register</b>, two registers (patch&nbsp;A, patch&nbsp;B) = 8 qubits <b>+ 1 ancilla</b> for the swap test = <b>9 wires total</b></td></tr>
  <tr><td>Trainable parameters</td><td><b>24</b> &mdash; shape (2 layers &times; 4 qubits &times; 3 rotations RX/RY/RZ); the same weight tensor is <b>tied</b> across both registers (Siamese)</td></tr>
  <tr><td>Similarity readout</td><td>Swap test: ancilla &lang;Z&rang; = |&lang;&psi;<sub>A</sub>|&psi;<sub>B</sub>&rang;|&sup2;, a learned similarity in [0,1]</td></tr>
  <tr><td>Patches</td><td><b>32&times;32</b> px, stride <b>8</b>, sliding window &rarr; <b>841 patches</b> on a 256&times;256 image</td></tr>
  <tr><td>Patch descriptor</td><td><b>24-dim</b>: 12 largest-magnitude DCT coefficients + 12-bin intensity histogram, L2-normalized, angle-encoded via data re-uploading</td></tr>
  <tr><td>Candidate pairing</td><td>O(N log N) lexicographic block-matching (Fridrich-style) &mdash; sort by feature, compare within a sliding window, instead of O(N&sup2;) all-pairs</td></tr>
  <tr><td>False-match suppression</td><td>Offset-consistency vote over (dy, dx): genuine copy-move produces many pairs sharing one offset (translation only &mdash; rotation is a documented TODO)</td></tr>
  <tr><td>Binary mask</td><td>GT: grayscale &rarr; NEAREST resize &rarr; threshold &gt;128. Prediction: fill matched-pair patches &rarr; morphological close (dilate&rarr;erode)</td></tr>
  <tr><td>Training</td><td>Self-supervised synthetic copy-move pairs (parameter-shift gradients, MSE on similarity labels)</td></tr>
</table>
<p><b>Is the architecture correct?</b> Conceptually yes for <i>measuring patch similarity</i>, and the
degeneracy bug is fixed. But it is <b>incomplete as a detector</b>: (i) there is no image-level
forged/authentic decision, (ii) the DCT+histogram descriptor is not rotation/scale invariant,
and (iii) it is not yet shown that the quantum similarity outperforms a classical distance on
the same features (an ablation is needed).</p>

<h2 class="pb">4. Datasets &amp; Their Roles</h2>
<p>Training and evaluation must use <b>different</b> data sources &mdash; a recurring point of confusion.
The Siamese similarity <i>must</i> be trained on synthetic pairs because real CMFD datasets provide
no patch-level correspondence ground truth.</p>
<table>
  <tr><th>Dataset</th><th>Role</th><th>Why</th></tr>
  <tr><td>Synthetic (self-supervised)</td><td class="good">Training &mdash; mandatory</td><td>Only source of labeled positive/negative patch pairs; CoMoFoD masks have no correspondence labels</td></tr>
  <tr><td>CoMoFoD (small v2)</td><td class="good">Primary evaluation (headline)</td><td>Real forgeries + real binary masks; loader already matches it (5,000 forged images)</td></tr>
  <tr><td>COVERAGE (100 pairs)</td><td class="warn">Stress test only</td><td>Adversarial by design (similar genuine objects); useful to prove generalization, but not a headline metric</td></tr>
</table>
<p><b>Recommendation:</b> train on synthetic, report on <b>CoMoFoD</b>. Never evaluate on synthetic data
(it only proves the model learned its own augmentation). Keep COVERAGE as a &ldquo;hard case&rdquo; data point.</p>

<h2>5. Evaluation Methodology &mdash; why not accuracy?</h2>
<p>Forged regions occupy only <b>~2&ndash;11% of pixels</b>, so the pixel classes are extremely imbalanced.
Under this imbalance, plain accuracy is <b>actively misleading</b>: a model that predicts &ldquo;nothing is
forged&rdquo; scores ~90&ndash;98% accuracy while detecting <b>zero</b> forgeries.</p>
<table class="avoid-break">
  <tr><th>Metric</th><th>Definition</th><th>Meaning here</th></tr>
  <tr><td>Accuracy</td><td>(TP+TN) / all</td><td class="bad">Misleading &mdash; rewards predicting nothing</td></tr>
  <tr><td>Recall</td><td>TP / (TP+FN)</td><td>Fraction of true forged pixels <i>found</i>. <span class="bad">Gamed by flagging everything</span> &rarr; this is the &ldquo;80%&rdquo; trap</td></tr>
  <tr><td>Precision</td><td>TP / (TP+FP)</td><td>Of the pixels flagged, how many were actually forged</td></tr>
  <tr><td>F1</td><td>2&middot;P&middot;R / (P+R)</td><td class="good">Honest single scalar &mdash; collapses if precision OR recall is bad</td></tr>
  <tr><td>IoU (Jaccard)</td><td>TP / (TP+FP+FN)</td><td class="good">Localization overlap &mdash; directly measures &ldquo;does the predicted region match the true region?&rdquo;</td></tr>
</table>
<div class="callout"><b>Reporting rule for this project:</b> lead with <b>F1</b> and <b>IoU</b>; cite precision+recall as
the pair that explains F1; never headline accuracy. IoU is literally the metric that measures the
stated goal (exact forged area).</div>

<h2 class="pb">6. Results</h2>

<h3>6.1 COVERAGE &mdash; loose parameters (threshold 0.8, votes 3, 384&nbsp;px)</h3>
<table class="avoid-break">
  <tr><th>Image</th><th class="n">Accuracy</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">IoU</th></tr>
  <tr><td>COVERAGE 1t</td><td class="n">0.277</td><td class="n">0.269</td><td class="n">1.000</td><td class="n">0.424</td><td class="n">0.269</td></tr>
  <tr><td>COVERAGE 10t</td><td class="n">0.092</td><td class="n">0.055</td><td class="n">1.000</td><td class="n">0.105</td><td class="n">0.055</td></tr>
  <tr><td>COVERAGE 100t</td><td class="n">0.344</td><td class="n">0.337</td><td class="n">1.000</td><td class="n">0.505</td><td class="n">0.337</td></tr>
  <tr class="tot"><td>Mean</td><td class="n">0.238</td><td class="n">0.220</td><td class="n">1.000</td><td class="n">0.345</td><td class="n">0.221</td></tr>
</table>
<figure><img src="data:image/png;base64,__CHART1__">
<figcaption>Figure 1. Across every test image, recall is ~1.0 while precision is far lower &mdash; the model
finds the forgery only because it flags almost the whole image. (COVERAGE at loose parameters;
CoMoFoD from saved masks, &sect;6.3.)</figcaption></figure>

<h3>6.2 Effect of tightening parameters (COVERAGE, threshold 0.92, votes 8, 256&nbsp;px)</h3>
<table class="avoid-break">
  <tr><th>Image</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">IoU</th></tr>
  <tr><td>COVERAGE 1t</td><td class="n">0.308</td><td class="n">0.994</td><td class="n">0.471</td><td class="n">0.308</td></tr>
  <tr><td>COVERAGE 10t</td><td class="n">0.102</td><td class="n">1.000</td><td class="n">0.185</td><td class="n">0.102</td></tr>
  <tr><td>COVERAGE 100t</td><td class="n">0.259</td><td class="n">0.420</td><td class="n">0.320</td><td class="n">0.191</td></tr>
  <tr class="tot"><td>Mean</td><td class="n">0.223</td><td class="n">0.805</td><td class="n">0.325</td><td class="n">0.200</td></tr>
</table>
<figure><img src="data:image/png;base64,__CHART2__">
<figcaption>Figure 2. The mechanism improves sharply: the offset-consistency filter now removes ~95% of
candidate matches (before &rarr; after: 7,984&rarr;229, 2,466&rarr;43, 2,001&rarr;86). Mean F1 looks flat only
because image 100t swung from &ldquo;flag everything&rdquo; (recall 1.0) to under-detecting (recall 0.42).</figcaption></figure>

<h3>6.3 CoMoFoD &mdash; recomputed from saved prediction masks</h3>
<p>No CoMoFoD metrics were previously persisted; the two masks below were recovered from an
earlier run (loose parameters) and re-scored against ground truth.</p>
<table class="avoid-break">
  <tr><th>Image</th><th class="n">Accuracy</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">IoU</th><th class="n">Forged&nbsp;area</th></tr>
  <tr><td>001_F</td><td class="n">0.030</td><td class="n">0.017</td><td class="n">0.806</td><td class="n">0.032</td><td class="n">0.017</td><td class="n">2.0%</td></tr>
  <tr><td>050_F_BC1</td><td class="n">0.125</td><td class="n">0.111</td><td class="n">1.000</td><td class="n">0.200</td><td class="n">0.111</td><td class="n">10.9%</td></tr>
</table>
<div class="callout red"><b>This resolves the &ldquo;80%&rdquo; question.</b> Image 001_F scored <b>recall 0.806</b> &mdash; the figure
in memory &mdash; but with <b>precision 0.017</b> and <b>accuracy 0.030</b>, because ~97% of the image was
flagged (63,346 false-positive pixels out of 65,536). CoMoFoD floods exactly like COVERAGE, so the
over-detection is a <b>model/threshold problem, not a dataset-specific one</b>.</div>

<h2 class="pb">7. Key Finding &mdash; systemic over-detection</h2>
<p>The predicted mask covers most of each image. The diagnostic signature is consistent:
<b>recall &asymp; 1.0, precision near the forged-area base rate</b>. The offset-consistency filter, which
is supposed to suppress coincidental matches, barely rejects anything at loose thresholds
(e.g. 9,734&rarr;7,984 pairs). On self-similar texture, false matches cluster into offset bins just
like true ones, so they survive.</p>
<p><b>Two compounding causes:</b> (1) the DCT+histogram descriptor is weak &mdash; it cannot separate a
true duplicate from a genuinely similar region; (2) thresholds/resolution were not tuned for the
test conditions. Tightening parameters (&sect;6.2) confirms the mechanism is fixable &mdash; surviving
false matches dropped ~95% &mdash; but precision remains low, pointing at the descriptor as the deeper
bottleneck.</p>

<h2>8. Gap Analysis &mdash; distance to the final goal</h2>
<table class="avoid-break">
  <tr><th style="width:38%">Capability</th><th style="width:14%">Status</th><th>Reality</th></tr>
  <tr><td>1. Input any image + preprocess</td><td><span class="pill done">DONE</span></td><td>Loaders + patch extraction work on any image/dataset</td></tr>
  <tr><td>2. Image-level &ldquo;forged: yes/no&rdquo;</td><td><span class="pill no">NOT BUILT</span></td><td>Detector always returns a mask; nothing decides &ldquo;authentic&rdquo;. No classifier head</td></tr>
  <tr><td>3. Localization output format (box / heatmap)</td><td><span class="pill half">HALF</span></td><td>Produces a binary mask; converting to a box/heatmap is trivial (~10 lines)</td></tr>
  <tr><td>4. Localization accuracy (exact edge/area)</td><td><span class="pill no">NOT MET</span></td><td>Floods 50&ndash;97% of the image (precision 0.05&ndash;0.34) &mdash; the core unsolved problem</td></tr>
</table>
<div class="callout"><b>Overall &asymp; 35% of a usable output.</b> Scaffolding is ~60&ndash;70% built, but the parts that
make it <i>a detector</i> &mdash; discriminating forged-vs-clean and localizing precisely &mdash; are the hard
remaining work. The gap is detection <i>quality</i>, not output formatting.</div>

<h2>9. Recommendations</h2>
<ul>
  <li><b>Datasets:</b> keep training on <b>synthetic</b> pairs (mandatory); report headline metrics on
      <b>CoMoFoD</b>; use <b>COVERAGE</b> only as a stress test.</li>
  <li><b>COVERAGE &mdash; keep it.</b> ~142&nbsp;MB extracted, the loader reads it, and a second dataset
      strengthens the report. Optionally delete only the 142&nbsp;MB download zip
      (<code>OneDrive_1_7-26-2026.zip</code>) to reclaim space.</li>
  <li><b>Metrics:</b> report F1 + IoU (with precision/recall); drop accuracy from headline claims.</li>
</ul>

<h2>10. Roadmap &mdash; next steps</h2>
<ol>
  <li><b>Establish an honest baseline</b> &mdash; run a 20&ndash;50 image CoMoFoD batch with tuned parameters
      and record mean F1/IoU. (Nothing trustworthy exists yet.)</li>
  <li><b>De-flood localization</b> &mdash; tune <code>sim-threshold</code> / <code>min-offset-votes</code>;
      strengthen the offset filter (e.g. RANSAC on offsets); if precision still stalls, replace/augment
      the descriptor with a rotation/scale-invariant feature (e.g. Zernike moments).</li>
  <li><b>Add the image-level decision</b> &mdash; derive forged/authentic from the offset-consistent
      matched area, calibrated on authentic images to bound false positives.</li>
  <li><b>Ablation</b> &mdash; confirm the swap-test QNN beats a classical similarity on the same features;
      otherwise justify the quantum component explicitly.</li>
  <li><b>Final output format</b> &mdash; convert the mask to a bounding box (connected components) or a
      similarity heatmap. Trivial, and correctly done last.</li>
</ol>
<div class="callout"><b>Single most important next action:</b> steps 1&ndash;2 &mdash; get a real CoMoFoD number, then
kill the over-detection. Everything else builds on that.</div>

<h2 class="pb">Appendix A &mdash; exact run configurations</h2>
<pre>COVERAGE, loose (&sect;6.1):
  --forged-glob '*t.tif' --mask-union
  --sim-threshold 0.8  --min-offset-votes 3  --target-long-side 384

COVERAGE, tuned (&sect;6.2):
  --forged-glob '*t.tif' --mask-union
  --sim-threshold 0.92 --min-offset-votes 8  --target-long-side 256

Model checkpoint: checkpoints_full/swaptest_qnn_epoch10.npy
  weights shape (2, 4, 3) = 24 params  |  9 wires (4+4 registers + 1 ancilla)
Patch config: 32x32, stride 8  ->  841 patches @ 256x256</pre>

<h2>Appendix B &mdash; file inventory</h2>
<table class="avoid-break">
  <tr><th>Path</th><th>Purpose</th></tr>
  <tr><td><code>code/cmfd_swaptest_pipeline.py</code></td><td>Core pipeline: QNN, detector, metrics, training</td></tr>
  <tr><td><code>code/grip_dataset.py</code></td><td>Dataset-agnostic loader (auto-discovery, mask-union, aspect-preserving resize)</td></tr>
  <tr><td><code>code/test_grip.py</code></td><td>Evaluation runner (native-res + tuning flags)</td></tr>
  <tr><td><code>data/comofod/</code></td><td>CoMoFoD small v2 (primary evaluation)</td></tr>
  <tr><td><code>data/coverage/</code></td><td>COVERAGE (100 pairs, stress test)</td></tr>
  <tr><td><code>results_coverage/, results_coverage_tuned/</code></td><td>COVERAGE run outputs (masks + CSV)</td></tr>
</table>

<footer>CMFD Phase-2 Progress &amp; Evaluation Report &middot; generated 26 July 2026 &middot;
all metrics computed from actual run outputs and saved prediction masks.</footer>
</body></html>"""

HTML = HTML.replace('__CHART1__', CHART1).replace('__CHART2__', CHART2)
with open('/home/adeveloper/Documents/FYP/Phase-2/cmfd_project/docs/CMFD_Phase2_Report.html', 'w') as f:
    f.write(HTML)
print("wrote docs/CMFD_Phase2_Report.html", len(HTML), "bytes")
