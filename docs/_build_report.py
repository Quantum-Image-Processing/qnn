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

# ==== CHART 1: the ablation - quantum vs classical similarity (200 pairs) ====
methods = ['Classical\ncosine sim.', 'Raw overlap\n|<a|b>|² ', 'Circuit\nUNTRAINED', 'Circuit\nTRAINED']
aucs    = [0.887, 0.887, 0.880, 0.883]
cols    = [MUTED, MUTED, BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(7.6, 3.4))
x = range(len(methods))
bars = ax.bar(x, aucs, 0.55, color=cols, zorder=3)
style(ax)
ax.set_ylim(0.5, 0.95); ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.axhline(0.5, color=BASELINE, lw=1, ls='--', zorder=2)
ax.set_xticks(list(x)); ax.set_xticklabels(methods, fontsize=9.5)
ax.set_ylabel('AUC (positives vs negatives)')
ax.set_title('No quantum advantage: the circuit does not beat a classical dot product',
             fontsize=11, color=INK, pad=10, loc='left')
for r, v in zip(bars, aucs):
    ax.text(r.get_x()+r.get_width()/2, v+0.006, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9, color=INK2)
CHART1 = to_b64(fig)

# ==== CHART 2: detection vs the trivial baseline, 8 distinct CoMoFoD scenes ==
scenes  = ['001', '002', '003', '004', '005', '006', '007', '008']
bestf1  = [0.0322, 0.1063, 0.0373, 0.0223, 0.1396, 0.2620, 0.0441, 0.0068]
trivial = [0.0322, 0.0816, 0.0345, 0.0220, 0.1396, 0.2620, 0.0398, 0.0051]

fig, ax = plt.subplots(figsize=(7.6, 3.4))
x = range(len(scenes)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], bestf1,  w, label='Best achievable F1 (oracle threshold)', color=BLUE, zorder=3)
b2 = ax.bar([i + w/2 for i in x], trivial, w, label='"Mark every pixel forged" baseline', color=BASELINE, zorder=3)
style(ax)
ax.set_ylim(0, 0.30)
ax.set_xticks(list(x)); ax.set_xticklabels([f'scene {s}' for s in scenes], fontsize=9)
ax.set_ylabel('F1')
ax.set_title('Detection is indistinguishable from marking the whole image forged',
             fontsize=11, color=INK, pad=10, loc='left')
ax.legend(frameon=False, loc='upper left', fontsize=9)
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
    Pipeline: <code>cmfd_swaptest_pipeline.py</code> (v3 &mdash; pure QNN)</div>
</div>

<h2>1. Executive Summary</h2>
<p>This report documents the current state of a quantum-machine-learning pipeline for
<b>copy-move forgery detection (CMFD)</b> &mdash; detecting regions of an image that have been
duplicated and pasted elsewhere in the same image. The system pairs a classical
patch descriptor with a <b>Siamese swap-test quantum neural network (QNN)</b> that learns a
similarity score between two image patches, followed by a spatial offset-consistency
filter that keeps only genuine duplications.</p>
<div class="callout red">
  <b>Headline finding.</b> The pipeline is a genuine end-to-end quantum model, and it does not work
  as a detector. Both halves of that sentence are measured, not asserted:
  <ul style="margin:6pt 0 0 0">
    <li>The trained circuit scores <b>AUC 0.883</b> against <b>0.887</b> for a parameter-free classical
        cosine similarity &mdash; <b>no quantum advantage</b>, at ~36&nbsp;ms/pair versus microseconds.</li>
    <li>Training moves AUC by <b>0.003</b> (0.880 untrained &rarr; 0.883 trained). The 58 rotation angles
        influence the output <i>less</i> than the choice of input image does.</li>
    <li>Best achievable detection is <b>F1 0.0813</b> against a <b>0.0771</b> baseline of marking every
        pixel forged &mdash; and that is with an oracle per-image threshold.</li>
  </ul>
</div>
<div class="callout red">
  <b>Root cause.</b> The classical offset-consistency vote removed in v3 was not scaffolding around
  the quantum model &mdash; <b>it was the component performing the detection</b>. Patch similarity alone
  drowns in natural self-similarity; a duplicated region is a small signal among tens of thousands
  of genuinely similar patch pairs. See &sect;7.
</div>
<div class="callout red">
  <b>v3 finding &mdash; the reported non-convergence was a data bug, not a model bug.</b>
  Training loss was previously stuck at ~0.33, above the 0.25 constant-predictor floor, and this
  was attributed to the model. The actual cause was in synthetic pair construction: positives
  were rotated 90/180/270&deg; three times out of four. A swap test compares amplitude vectors, so
  a rot90 duplicate permutes the state into a near-orthogonal one &mdash; <b>measured: an exact
  rot90 copy scores ~0.01 where an exact copy scores ~1.0</b>. Labelling those pairs 1.0 asks the
  circuit to fit a target its translation-only architecture provably cannot represent.
  With rotations excluded, <b>loss drops to ~0.003&ndash;0.10</b>, far below the floor.
  Rotated positives remain available behind <code>--train-rotations</code> and should only be
  re-enabled together with a d&theta; component in the offset register.
</div>
<div class="callout">
  <b>Status of &sect;6&ndash;&sect;10.</b> The results, gap analysis, recommendations and roadmap below were
  produced with the <b>v2 hybrid</b> and have <b>not yet been re-run against v3</b>. They are
  retained as the v2 baseline for comparison. Treat every number in them as v2 until the v3
  evaluation is complete; v3 is expected to score <i>lower</i>, since removing the offset vote
  removed the stage that actually suppressed false matches.
</div>

<h2>2. Objective &amp; Target Output</h2>
<p>The intended end product: <i>given any input image, decide whether it is forged, and if so,
localize the exact duplicated region as a bounding box or heatmap.</i> That decomposes into
four capabilities, tracked in the gap analysis (&sect;8).</p>

<h2>3. System Architecture (v3 &mdash; pure QNN)</h2>
<p>The architecture is a <b>pairwise similarity model</b>, which is the structurally correct choice
for copy-move (the task is inherently &ldquo;does patch A match patch B elsewhere?&rdquo;). A single-patch
classifier cannot express duplication.</p>
<p><b>v3 removes every classical detection stage from the v2 hybrid.</b> v2 was quantum only in its
similarity readout: a classical DCT descriptor decided what the circuit was allowed to see, a
classical sort decided which pairs it was allowed to score, and a classical Hough vote decided
which of its outputs to believe. In v3 raw pixels enter the circuit and a measured expectation
value is the decision.</p>
<table class="avoid-break">
  <tr><th style="width:30%">Stage</th><th>v2 (hybrid)</th><th>v3 (pure)</th></tr>
  <tr><td>Patch representation</td><td>24-dim DCT coefficients + intensity histogram</td><td><b>AmplitudeEmbedding of the raw pixels</b> &mdash; all patch&sup2; values, L2-normalized state preparation only</td></tr>
  <tr><td>Candidate pairing</td><td>O(N log N) lexicographic block matching (Fridrich)</td><td><b>Exhaustive</b> &mdash; every admissible pair scored by the circuit, O(N&sup2;)</td></tr>
  <tr><td>False-match suppression</td><td>Classical (dy, dx) Hough vote</td><td><b>In-circuit offset register</b> + trainable CRY coupling inside the interferometer</td></tr>
  <tr><td>Mask cleanup</td><td>Morphological close (dilate&rarr;erode)</td><td><b>None</b></td></tr>
  <tr><td>Trainable parameters</td><td>circuit rotation angles only</td><td>circuit rotation angles only (<i>unchanged &mdash; v2 was already pure here</i>)</td></tr>
</table>
<table class="avoid-break">
  <tr><th style="width:34%">Component</th><th>v3 specification</th></tr>
  <tr><td>Quantum registers</td><td><b>6 qubits per register</b> (8&times;8 patch = 64 amplitudes), two registers = 12 qubits <b>+ 2 offset qubits + 1 ancilla</b> = <b>15 wires total</b></td></tr>
  <tr><td>Trainable parameters</td><td><b>58</b> &mdash; (3 layers &times; 6 qubits &times; 3 rotations) tied across both registers, plus (2 &times; 2) offset-register angles</td></tr>
  <tr><td>Similarity readout</td><td>Swap test with an offset-conditioned readout: H &rarr; CSWAP &rarr; trainable CRY from the offset wires &rarr; H, measured as ancilla &lang;Z&rang;, clipped to [0,1]</td></tr>
  <tr><td>Patches</td><td><b>8&times;8</b> px, stride <b>16</b> &rarr; 256 patches / ~33k pairs on a 256&times;256 image</td></tr>
  <tr><td>Simulation</td><td><code>lightning.qubit</code> + adjoint gradients (3&ndash;7&times; faster than <code>default.qubit</code>; measured <b>0.036 s per pair</b>)</td></tr>
  <tr><td>Training</td><td>Self-supervised synthetic copy-move pairs, MSE on similarity labels</td></tr>
</table>
<div class="callout red">
  <b>Do not &ldquo;simplify&rdquo; the ansatz.</b> A plain amplitude embedding followed by a
  data-independent tied unitary is provably degenerate under a swap test:
  |&lang;U(w)f<sub>A</sub>|U(w)f<sub>B</sub>&rang;|&sup2; = |&lang;f<sub>A</sub>|f<sub>B</sub>&rang;|&sup2;
  for <i>any</i> U(w), so the gradient w.r.t. w is exactly zero. This was measured, not assumed:
  gradient norm <b>~1e-16</b> in the degenerate form vs <b>~1.2e-2</b> with data re-uploading.
  Re-uploading the pixels into the rotation angles is what keeps the tied weights alive.
</div>
<p><b>Cost of purity.</b> Removing the classical prefilter makes pair scoring O(N&sup2;), so stride is
now the dominant cost dial: ~1 min/image at stride 32, ~20 min at stride 16, ~5 h at stride 8.
The quantum replacement for the prefilter is Grover/amplitude amplification over a pair-index
register, which requires QRAM to hold all patches in superposition &mdash; not simulable at this
scale, so it is recorded as future work rather than substituted with a classical stand-in.</p>
<p><b>Known weakness, stated plainly.</b> The offset register is a <i>weaker</i> mechanism than the
vote it replaced, not an equivalent one. Offset consistency is a <b>global</b> property of the pair
ensemble (&ldquo;many pairs share one offset&rdquo;); a per-pair circuit cannot observe clustering it is
never shown. A faithful quantum version needs quantum counting over all pairs at once. Expect
more false positives on homogeneous regions (sky, walls) than v2 produced.</p>

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

<h2 class="pb">6. Results &mdash; pure QNN</h2>
<p>Three questions were tested separately, because they have different answers:
does the model <i>learn</i>, does the quantum circuit <i>beat a classical equivalent</i>, and does
the system <i>detect copy-move forgery</i>.</p>

<h3>6.1 Does training improve the model? No.</h3>
<p>Two training runs, both on the pure architecture:</p>
<table class="avoid-break">
  <tr><th>Run</th><th>Synthetic positives</th><th>Loss trajectory</th><th>Interpretation</th></tr>
  <tr><td>A (10 epochs &times; 80 img)</td><td>pixel-identical copies</td><td>0.047 at epoch 1, flat thereafter</td><td>Nothing to learn &mdash; a swap test returns ~1.0 for identical states with no trained parameters, so the model starts at the answer</td></tr>
  <tr><td>B (11 epochs &times; 200 img)</td><td>degraded + feathered</td><td>0.158 &rarr; 0.178, no downward trend</td><td>A real loss surface, but the weights cannot descend it</td></tr>
</table>
<p>Run B was tested with both plain gradient descent and Adam at identical settings and seed. The two
trajectories agreed to within 0.002 per epoch and final AUC differed by 0.005 &mdash; so the optimizer
is not the limitation.</p>
<div class="callout red"><b>Why the weights cannot help.</b> Varying the 58 rotation angles across
independent random draws changes a pair's score by &sigma;&nbsp;=&nbsp;0.117. Varying the <i>input data</i>
changes it by &sigma;&nbsp;=&nbsp;0.138. The parameters inject almost as much variation as the image
content does &mdash; they behave closer to noise than to a learned function.</div>

<h3>6.2 Does the quantum circuit beat a classical equivalent? No.</h3>
<p>Measured on 200 synthetic pairs (100 positive, 100 negative), reporting AUC &mdash; the probability
that a true duplicate outscores a non-duplicate, where 0.5 is chance:</p>
<table class="avoid-break">
  <tr><th>Scorer</th><th class="n">AUC</th><th>Trainable parameters</th></tr>
  <tr><td>Classical cosine similarity</td><td class="n">0.887</td><td>0</td></tr>
  <tr><td>Raw state overlap |&lang;f<sub>A</sub>|f<sub>B</sub>&rang;|&sup2;</td><td class="n">0.887</td><td>0</td></tr>
  <tr><td>Swap-test circuit, untrained</td><td class="n">0.880</td><td>58</td></tr>
  <tr class="tot"><td>Swap-test circuit, trained</td><td class="n">0.883</td><td>58</td></tr>
</table>
<figure><img src="data:image/png;base64,__CHART1__">
<figcaption>Figure 1. The trained quantum circuit does not outperform a parameter-free classical dot
product. Training moves AUC by 0.003; the gap to classical cosine similarity is &minus;0.004.</figcaption></figure>
<p><b>This is structural, not a tuning failure.</b> A swap test measures
|&lang;&psi;<sub>A</sub>|&psi;<sub>B</sub>&rang;|&sup2;. On amplitude-encoded raw pixels that quantity
<i>is</i> squared cosine similarity &mdash; fixed and parameter-free. The data re-uploading layers exist
only to escape the provable degeneracy described in &sect;3, and they buy trainability by scrambling
the encoded states, which can only move the measured overlap away from the informative one. The
architecture therefore faces a genuine dilemma: <b>untrainable and accurate, or trainable and
slightly worse.</b></p>

<h3>6.3 Does it detect copy-move forgery? No.</h3>
<p>Eight <b>distinct</b> CoMoFoD scenes, 128&nbsp;px tiled (patch 8, stride 8, full coverage), scored
exhaustively. Both columns favour the detector: the scorer is the best available (&sect;6.2) and the
threshold is chosen <i>per image</i> with knowledge of the ground truth &mdash; an oracle no real
deployment gets.</p>
<table class="avoid-break">
  <tr><th>Scene</th><th class="n">Forged area</th><th class="n">Best achievable F1</th><th class="n">&ldquo;Mark every pixel forged&rdquo; F1</th></tr>
  <tr><td>001_F</td><td class="n">1.6%</td><td class="n">0.0322</td><td class="n">0.0322</td></tr>
  <tr><td>002_F</td><td class="n">4.3%</td><td class="n">0.1063</td><td class="n">0.0816</td></tr>
  <tr><td>003_F</td><td class="n">1.8%</td><td class="n">0.0373</td><td class="n">0.0345</td></tr>
  <tr><td>004_F</td><td class="n">1.1%</td><td class="n">0.0223</td><td class="n">0.0220</td></tr>
  <tr><td>005_F</td><td class="n">7.5%</td><td class="n">0.1396</td><td class="n">0.1396</td></tr>
  <tr><td>006_F</td><td class="n">15.1%</td><td class="n">0.2620</td><td class="n">0.2620</td></tr>
  <tr><td>007_F</td><td class="n">2.0%</td><td class="n">0.0441</td><td class="n">0.0398</td></tr>
  <tr><td>008_F</td><td class="n">0.3%</td><td class="n">0.0068</td><td class="n">0.0051</td></tr>
  <tr class="tot"><td>Mean</td><td class="n">&mdash;</td><td class="n">0.0813</td><td class="n">0.0771</td></tr>
</table>
<figure><img src="data:image/png;base64,__CHART2__">
<figcaption>Figure 2. Best achievable F1 tracks the trivial baseline. On scenes 001, 005 and 006 the
optimal threshold is the lowest one tried, keeping all 32,640 pairs &mdash; literally the
&ldquo;mark everything&rdquo; solution.</figcaption></figure>

<h2 class="pb">7. Key Finding &mdash; the classical vote was doing the detection</h2>
<div class="callout red"><b>Mean best-case F1 is 0.0813 against a trivial baseline of 0.0771.</b>
Stripped of every classical stage, the system does not detect copy-move forgery. It is
statistically indistinguishable from marking the entire image as forged.</div>
<p><b>Mechanism.</b> Patch similarity alone cannot identify duplication. In any natural photograph
tens of thousands of 8&times;8 patch pairs are genuinely similar &mdash; sky, foliage, brickwork, smooth
gradients. A copied region is a small signal inside that flood, and a per-pair similarity score has
no way to tell a duplicate from a coincidental look-alike.</p>
<p>The offset-consistency vote removed in v3 existed precisely to solve this: a genuine duplication
produces many pairs sharing <i>one</i> (dy,&nbsp;dx) offset, whereas coincidental matches scatter
across all offsets. It was not scaffolding around the quantum model &mdash; <b>it was the component
performing the detection.</b></p>
<p>The in-circuit offset register (&sect;3) cannot substitute for it, and the reason is structural
rather than incidental: offset consistency is a <b>global property of the pair ensemble</b>
(&ldquo;many pairs agree on one offset&rdquo;), and a circuit that sees one pair at a time cannot observe
clustering it is never shown. A faithful quantum replacement needs quantum counting or amplitude
estimation across all pairs simultaneously, which requires QRAM and is not simulable at this
scale.</p>

<h2>8. What was established</h2>
<table class="avoid-break">
  <tr><th style="width:44%">Claim</th><th style="width:12%">Verdict</th><th>Evidence</th></tr>
  <tr><td>The pipeline is a pure QNN</td><td><span class="pill done">TRUE</span></td><td>No classical stage in the decision path; only trainable parameters are circuit rotation angles (&sect;3)</td></tr>
  <tr><td>The re-uploading ansatz is non-degenerate</td><td><span class="pill done">TRUE</span></td><td>Gradient norm ~1.2e-2 vs ~1e-16 for embedding + tied unitary</td></tr>
  <tr><td>The quantum model learns from data</td><td><span class="pill no">FALSE</span></td><td>AUC 0.880 untrained &rarr; 0.883 trained; weight leverage below data variance (&sect;6.1)</td></tr>
  <tr><td>The quantum model beats a classical baseline</td><td><span class="pill no">FALSE</span></td><td>0.883 vs 0.887 for cosine similarity, at ~36&nbsp;ms/pair vs microseconds (&sect;6.2)</td></tr>
  <tr><td>The system detects copy-move forgery</td><td><span class="pill no">FALSE</span></td><td>F1 0.0813 vs 0.0771 trivial baseline, with an oracle threshold (&sect;6.3)</td></tr>
</table>

<h2>9. Interpretation</h2>
<p>This is a <b>negative result with an identified mechanism</b>, which is a stronger scientific
outcome than an unverified claim of quantum advantage. The contribution is not a working detector;
it is a controlled demonstration of <i>which</i> component of a quantum-assisted CMFD pipeline was
carrying the detection, established by removing every classical stage and measuring what remains.</p>
<p>Three findings generalise beyond this project:</p>
<ol>
  <li><b>A swap test on amplitude-encoded data is a fixed similarity measure.</b> Any variational
      structure wrapped around it either cancels (degenerate, zero gradient) or perturbs the encoding
      away from the quantity that carried the signal. Trainability and fidelity are in tension by
      construction, not by implementation choice.</li>
  <li><b>Self-supervised pair synthesis silently determines what is learnable.</b> Two separate label-noise
      bugs &mdash; rotated positives, and feathering applied to patch-sized pastes &mdash; each created targets
      the architecture provably could not represent, and each produced a loss curve that looked like
      model failure rather than data failure.</li>
  <li><b>Per-pair scoring cannot express ensemble-level evidence.</b> Copy-move detection depends on
      agreement <i>across</i> matches, which is invisible to any model evaluating one pair in isolation.</li>
</ol>

<h2>10. Future work</h2>
<ol>
  <li><b>Quantum offset consistency</b> &mdash; the honest replacement for the classical vote is quantum
      counting / amplitude estimation over a pair-index register, requiring QRAM to hold all patches
      in superposition. This is the single change that would make a pure quantum detector viable, and
      it is out of reach of statevector simulation at useful image sizes.</li>
  <li><b>Rotation invariance</b> &mdash; add a d&theta; component to the offset register. A rot90 duplicate
      currently scores ~0.01 where an exact copy scores ~1.0, so rotated CoMoFoD forgeries are missed
      entirely.</li>
  <li><b>Larger working resolution</b> &mdash; at 128&nbsp;px a 2%-area forgery spans only ~4 patches, and
      copy-move requires a matched <i>pair</i>. Exhaustive scoring is O(N&sup2;), so 256&nbsp;px costs
      ~5&nbsp;h/image against ~20&nbsp;min at 128&nbsp;px.</li>
  <li><b>Grover-based candidate search</b> &mdash; would restore the O(&radic;N) advantage lost when the
      classical prefilter was removed. Same QRAM prerequisite as (1).</li>
</ol>
<div class="callout"><b>Honest summary.</b> The architecture is correct, complete, and measured. It is
not a forgery detector, and the measurements above identify exactly why.</div>

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
