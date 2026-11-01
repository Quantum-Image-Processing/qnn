"""Compose the CMFD pure-QNN architecture figure: three annotated stages
(image preparation -> quantum comparison -> mask formation), in the visual
language of a conference architecture overview figure.

Regenerate after any circuit change:  python docs/figures/gen_architecture_figure.py
"""
INK, MUTED, TITLE = "#1a1a1a", "#5c5c5c", "#c0392b"
CREAM, CREAMB     = "#fdf6e3", "#d6c68a"
GREY,  GREYB      = "#e8e8e8", "#9a9a9a"
BLUE,  BLUEB      = "#d6e6f5", "#7ba7cf"
PINK              = "#f6c9c0"
TEAL              = "#0f7d88"
GREEN, GREENB     = "#e4f0e4", "#79a879"
W, H = 1330, 1600

o = []; A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'role="img" aria-label="Three-stage architecture. Stage 1 image preparation: an image is '
  'converted to grayscale and resized to 128 by 128, tiled into 256 non-overlapping 8 by 8 '
  'patches, each flattened and L2-normalised into a 64-amplitude state, and expanded into '
  '32,640 patch pairs each carrying a displacement. Stage 2 quantum comparison: the 15-wire '
  'swap-test circuit scores one pair per evaluation. Stage 3 mask formation: scores are '
  'thresholded, both patches of every matched pair are painted into the mask, connected '
  'components give boxes, and the matched-pair count gives the verdict.">')
A('<rect width="100%" height="100%" fill="#ffffff"/>')
A('<defs>'
  f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
  f'<marker id="at" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
  f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{TEAL}"/></marker></defs>')
A(f'<g font-family="Helvetica, Arial, sans-serif" fill="{INK}">')

def box(x,y,w,h,fill,stroke,rx=3,sw=1.3,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def txt(x,y,s,size=11,anchor="middle",fill=INK,weight="normal"):
    A(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>')
def arrow(x1,y1,x2,y2,color=INK,sw=1.4,teal=False,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{"at" if teal else "a"})"/>')
def badge(x,y,n,col=INK):
    A(f'<circle cx="{x}" cy="{y}" r="11" fill="{col}"/>')
    A(f'<text x="{x}" y="{y+4}" font-size="12" text-anchor="middle" fill="#ffffff" font-weight="bold">{n}</text>')

# ══════════════════ STAGE 1 — IMAGE PREPARATION ══════════════════
box(20,52,1260,268,"#ffffff",INK,rx=4,sw=1.5,dash="7 5")
txt(650,42,"Stage 1 : Image Preparation &#8212; every operation performed on pixels",13,"middle",TITLE,"bold")

# (1) input
badge(58,84,"1")
box(44,98,210,150,"#ffffff",GREYB)
box(64,112,170,110,GREY,GREYB)
txt(149,172,"input photo",11,"middle",MUTED)
txt(149,266,"any size, RGB &#8594; <tspan font-weight='bold'>grayscale</tspan>",10.5)
txt(149,282,"LANCZOS resize &#8594; <tspan font-weight='bold'>128 &#215; 128</tspan>",10.5)
txt(149,302,"colour discarded: duplication is",9.8,"middle",MUTED)
txt(149,314,"structural, not chromatic",9.8,"middle",MUTED)
arrow(258,173,296,173)

# (2) tiling  (16x16 grid of 8px patches)
badge(316,84,"2")
box(302,98,210,150,"#ffffff",GREYB)
gx,gy,c=340,104,8.6
for r in range(16):
    for k in range(16):
        f=PINK if (r,k) in ((3,3),(11,11)) else GREY
        A(f'<rect x="{gx+k*c:.1f}" y="{gy+r*c:.1f}" width="{c:.1f}" height="{c:.1f}" fill="{f}" stroke="{GREYB}" stroke-width="0.45"/>')
txt(407,266,"tile: stride = patch = 8 px",10.5)
txt(407,282,"<tspan font-weight='bold'>16 &#215; 16 = 256 patches</tspan>",10.5)
txt(407,302,"non-overlapping, 100% coverage &#8212;",9.8,"middle",MUTED)
txt(407,314,"every pixel in exactly one patch",9.8,"middle",MUTED)
arrow(516,173,554,173)

# (3) patch -> state vector
badge(574,84,"3")
box(560,98,210,150,"#ffffff",GREYB)
for r in range(8):
    for k in range(8):
        A(f'<rect x="{584+k*9}" y="{112+r*9}" width="9" height="9" fill="{PINK}" stroke="{GREYB}" stroke-width="0.45"/>')
txt(628,198,"8&#215;8 patch",9.8,"middle",MUTED)
arrow(662,155,686,155)
for i in range(6):
    A(f'<rect x="{694+i*11}" y="{140}" width="10" height="30" fill="{CREAM}" stroke="{CREAMB}" stroke-width="0.8"/>')
txt(760,160,"&#8230;",11)
txt(729,198,"64 values",9.8,"middle",MUTED)
txt(665,266,"flatten &#8594; <tspan font-weight='bold'>L2-normalise</tspan>  &#968; = x / &#8214;x&#8214;",10.5)
txt(665,282,"64 amplitudes = 2&#8310; &#8594; <tspan font-weight='bold'>6 qubits</tspan>",10.5)
txt(665,302,"normalising removes per-patch brightness &#8212;",9.8,"middle",MUTED)
txt(665,314,"free invariance to CoMoFoD's BC category",9.8,"middle",MUTED)
arrow(774,173,812,173)

# (4) pair formation
badge(832,84,"4")
box(818,98,442,150,"#ffffff",GREYB)
for r in range(8):
    for k in range(8):
        f=PINK if (r,k) in ((1,1),(6,6)) else GREY
        A(f'<rect x="{846+k*11}" y="{110+r*11}" width="11" height="11" fill="{f}" stroke="{GREYB}" stroke-width="0.5"/>')
A(f'<line x1="{846+1.5*11}" y1="{110+1.5*11}" x2="{846+6.5*11}" y2="{110+6.5*11}" stroke="{TITLE}" stroke-width="1.5" stroke-dasharray="4 3"/>')
txt(944,163,"&#916;",11,"middle",TITLE,"bold")
txt(1090,128,"pair every patch with every other",10.5)
txt(1090,144,"patch at least 8 px apart",10.5)
txt(1090,168,"<tspan font-weight='bold'>32,640 pairs per image</tspan>",11)
txt(1090,192,"each pair carries &#916; = (dy, dx) / 128,",9.8,"middle",MUTED)
txt(1090,204,"the pixel displacement between them",9.8,"middle",MUTED)
txt(1090,228,"&#8220;does region A reappear at region B?&#8221;",10,"middle",TITLE)
txt(1039,282,"&#968;&#8320;, &#968;&#8321; &#8712; &#8477;&#8310;&#8308;  +  &#916;   &#8594;  one circuit input",10.5)
txt(1039,302,"this pairing IS the copy-move question;",9.8,"middle",MUTED)
txt(1039,314,"cost is O(N&#178;) in patch count",9.8,"middle",MUTED)

# ══════════════════ STAGE 2 — QUANTUM COMPARISON ══════════════════
box(20,352,1260,868,"#ffffff",INK,rx=4,sw=1.5,dash="7 5")
txt(650,342,"Stage 2 : Quantum Comparison &#8212; one circuit evaluation per pair (15 wires, 32 ms)",13,"middle",TITLE,"bold")

y0,dy,gap=512,36,14
yA=[y0+dy*i for i in range(6)]
yB=[yA[-1]+dy+gap+dy*i for i in range(6)]
yO=[yB[-1]+dy+gap, yB[-1]+2*dy+gap]
yAnc=yO[-1]+dy+gap
XS,XE=205,1180
for i,y in enumerate([*yA,*yB,*yO,yAnc]):
    teal = (y==yAnc)
    A(f'<line x1="{XS}" y1="{y}" x2="{XE}" y2="{y}" stroke="{TEAL if teal else INK}" stroke-width="{1.5 if teal else 1.1}"/>')
    txt(XS-12,y+4,f"|0&#10217;<tspan font-size='9'> q{i}</tspan>",11,"end",TEAL if teal else INK)

def brace(y1,y2,label,sub):
    x=132
    A(f'<path d="M {x+14} {y1-9} L {x+6} {y1-9} L {x+6} {(y1+y2)/2-5} L {x} {(y1+y2)/2} L {x+6} {(y1+y2)/2+5} L {x+6} {y2+9} L {x+14} {y2+9}" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
    txt(x-8,(y1+y2)/2-2,label,11.5,"end",INK,"bold"); txt(x-8,(y1+y2)/2+13,sub,9.8,"end",MUTED)
brace(yA[0],yA[-1],"Reg A","patch &#968;&#8320;")
brace(yB[0],yB[-1],"Reg B","patch &#968;&#8321;")
brace(yO[0],yO[-1],"Offset","&#916;")
txt(124,yAnc+4,"Ancilla",11.5,"end",TEAL,"bold")

# (5) amplitude encoding
badge(248,428,"5")
txt(266,432,"amplitude encoding &#8212; 64 pixel amplitudes &#8594; 6 qubits",10.8,"start",INK)
for ys,l in ((yA,"&#968;&#8320;"),(yB,"&#968;&#8321;")):
    box(222,ys[0]-14,56,ys[-1]-ys[0]+28,CREAM,CREAMB,sw=1.5)
    txt(250,(ys[0]+ys[-1])/2-4,"Amp",11.5,"middle",INK,"bold"); txt(250,(ys[0]+ys[-1])/2+10,"Embed",11.5,"middle",INK,"bold")
    txt(250,ys[-1]+25,l,10.5,"middle",MUTED)

# (6) PQC
GX={"RX":300,"RY":348,"RZ":396}
for ys in (yA,yB):
    for y in ys:
        for g,x in GX.items():
            box(x,y-12,40,24,"#ffffff",INK,rx=3,sw=1.1); txt(x+20,y+4,g,10)
def cnot(xc,yc,yt):
    A(f'<line x1="{xc}" y1="{yc}" x2="{xc}" y2="{yt}" stroke="{INK}" stroke-width="1.2"/>')
    A(f'<circle cx="{xc}" cy="{yc}" r="3.8" fill="{INK}"/>')
    A(f'<circle cx="{xc}" cy="{yt}" r="8" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
    A(f'<line x1="{xc-8}" y1="{yt}" x2="{xc+8}" y2="{yt}" stroke="{INK}" stroke-width="1.2"/>')
    A(f'<line x1="{xc}" y1="{yt-8}" x2="{xc}" y2="{yt+8}" stroke="{INK}" stroke-width="1.2"/>')
for ys in (yA,yB):
    for k in range(6): cnot(460+k*25, ys[k], ys[(k+1)%6])
A(f'<path d="M 294 {yA[0]-22} L 294 {yA[0]-30} L 610 {yA[0]-30} L 610 {yA[0]-22}" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
badge(276,yA[0]-30,"6")
txt(430,yA[0]-38,"PQC &#215;3 &#8212; angle = &#952; + pixel&#183;&#960; (re-uploading) + CNOT ring",10.5,"middle",INK)
A(f'<line x1="452" y1="{yA[-1]+16}" x2="452" y2="{yB[0]-16}" stroke="{TEAL}" stroke-width="1.6" stroke-dasharray="4 3"/>')
A(f'<circle cx="452" cy="{(yA[-1]+yB[0])/2}" r="4" fill="{TEAL}"/>')
txt(470,(yA[-1]+yB[0])/2+4,"shared &#952; (Siamese) &#8212; 54 angles, ONE tensor drives both registers",10.8,"start",TEAL,"bold")

# (7) offset encoding
badge(320,yO[0]-26,"7")
txt(338,yO[0]-22,"displacement &#916; angle-encoded into 2 qubits",10.8,"start",INK)
for y in yO:
    box(300,y-12,136,24,BLUE,BLUEB,rx=3,sw=1.2); txt(368,y+4,"RY(&#966; + &#916;&#183;&#960;)",10)

# (8) swap test
badge(690,428,"8",TEAL)
txt(708,432,"swap test &#8212; the comparison itself",10.8,"start",TEAL,"bold")
box(648,yAnc-15,32,30,"#ffffff",TEAL,rx=3,sw=1.6); txt(664,yAnc+5,"H",13,"middle",TEAL,"bold")
for k in range(6):
    x=716+k*30
    A(f'<line x1="{x}" y1="{yA[k]}" x2="{x}" y2="{yAnc}" stroke="{TEAL}" stroke-width="1.3"/>')
    A(f'<circle cx="{x}" cy="{yAnc}" r="4.5" fill="{TEAL}"/>')
    for yy in (yA[k],yB[k]):
        A(f'<line x1="{x-6}" y1="{yy-6}" x2="{x+6}" y2="{yy+6}" stroke="{TEAL}" stroke-width="1.8"/>')
        A(f'<line x1="{x+6}" y1="{yy-6}" x2="{x-6}" y2="{yy+6}" stroke="{TEAL}" stroke-width="1.8"/>')
txt(915,yA[0]-38,"CSWAP &#215;6 &#8212; ancilla-controlled swap of the WHOLE register, one gate per qubit pair",10.5,"middle",TEAL)
for k,y in enumerate(yO):
    x=920+k*44
    A(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{yAnc-15}" stroke="{INK}" stroke-width="1.2"/>')
    A(f'<circle cx="{x}" cy="{y}" r="4.2" fill="{INK}"/>')
    box(x-16,yAnc-15,32,30,BLUE,BLUEB,rx=3,sw=1.2); txt(x,yAnc+5,"RY",10)
txt(942,yAnc+44,"CRY &#215;2 &#8212; offset modulates the interference fringe",10.5,"middle",MUTED)
box(1016,yAnc-15,32,30,"#ffffff",TEAL,rx=3,sw=1.6); txt(1032,yAnc+5,"H",13,"middle",TEAL,"bold")
box(1076,yAnc-16,42,32,"#e0f0f1",TEAL,rx=3,sw=1.7)
A(f'<path d="M 1085 {yAnc+8} A 12 12 0 0 1 1109 {yAnc+8}" fill="none" stroke="{TEAL}" stroke-width="1.5"/>')
A(f'<line x1="1097" y1="{yAnc+8}" x2="1109" y2="{yAnc-8}" stroke="{TEAL}" stroke-width="1.5"/>')
txt(1097,yAnc-28,"&#10216;Z&#10217; &#8712; [0,1]",10.5,"middle",TEAL,"bold")
txt(650,yAnc+80,"identical patches &#8594; swapping changes nothing &#8594; perfect interference &#8594; &#10216;Z&#10217; = 1"
                "&#160;&#160;&#183;&#160;&#160;orthogonal patches &#8594; no interference &#8594; &#10216;Z&#10217; = 0",10.5,"middle",TEAL)
txt(650,yAnc+98,"&#10216;Z&#10217; = |&#10216;&#968;&#8320;|&#968;&#8321;&#10217;|&#178;   &#8212;   the squared overlap of the two patch states = their similarity",10.5,"middle",INK)

# ══════════════════ STAGE 3 — MASK FORMATION ══════════════════
box(20,1252,1260,306,"#ffffff",INK,rx=4,sw=1.5,dash="7 5")
txt(650,1242,"Stage 3 : Mask Formation &#8212; turning 32,640 scores back into an image",13,"middle",TITLE,"bold")

# (9) threshold
badge(58,1284,"9")
box(44,1298,238,150,"#ffffff",GREYB)
for i,h in enumerate([10,16,26,40,58,44,30,20,14]):
    col = TEAL if i>=6 else GREY
    A(f'<rect x="{66+i*23}" y="{1412-h}" width="19" height="{h}" fill="{col}" stroke="{GREYB}" stroke-width="0.6"/>')
A(f'<line x1="{66+6*23-4}" y1="1330" x2="{66+6*23-4}" y2="1422" stroke="{TITLE}" stroke-width="1.5" stroke-dasharray="4 3"/>')
txt(214,1326,"&#10216;Z&#10217; &#8805; 0.90",10,"middle",TITLE,"bold")
txt(163,1430,"score distribution",9.8,"middle",MUTED)
txt(163,1476,"<tspan font-weight='bold'>threshold</tspan> the 32,640 scores",10.5)
txt(163,1496,"pairs above it = &#8220;matched&#8221;",10.5,"middle",MUTED)
txt(163,1524,"the only decision made from the",9.8,"middle",MUTED)
txt(163,1536,"circuit output",9.8,"middle",MUTED)
arrow(286,1373,324,1373)

# (10) paint mask
badge(344,1284,"10")
box(330,1298,238,150,"#ffffff",GREYB)
A(f'<rect x="378" y="1340" width="142" height="126" fill="#111111"/>')
A(f'<rect x="{378+2*17}" y="{1340+2*15}" width="34" height="30" fill="#ffffff"/>')
A(f'<rect x="{378+5*17}" y="{1340+5*15}" width="34" height="30" fill="#ffffff"/>')
txt(449,1476,"paint <tspan font-weight='bold'>BOTH</tspan> patches of every",10.5)
txt(449,1492,"matched pair into a 128&#215;128 mask",10.5)
txt(449,1520,"copy-move marks the source AND",9.8,"middle",MUTED)
txt(449,1532,"the pasted copy &#8212; never just one",9.8,"middle",MUTED)
arrow(572,1373,610,1373)

# (11) connected components
badge(630,1284,"11")
box(616,1298,238,150,"#ffffff",GREYB)
A(f'<rect x="664" y="1340" width="142" height="126" fill="{GREY}" stroke="{GREYB}"/>')
A(f'<rect x="{664+2*17-4}" y="{1340+2*15-4}" width="42" height="38" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
A(f'<rect x="{664+5*17-4}" y="{1340+5*15-4}" width="42" height="38" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
txt(735,1476,"connected components &#8594; boxes",10.5)
txt(735,1492,"rescaled to the original resolution",10.5,"middle",MUTED)
txt(735,1520,"display only &#8212; runs after the verdict",9.8,"middle",MUTED)
txt(735,1532,"and feeds nothing back into it",9.8,"middle",MUTED)
arrow(858,1373,896,1373)

# (12) verdict
badge(916,1284,"12")
box(902,1298,358,150,GREEN,GREENB)
txt(1081,1340,"matched pairs &#8805; 8   &#8594;   <tspan font-weight='bold'>FORGED</tspan>",12)
txt(1081,1364,"otherwise   &#8594;   <tspan font-weight='bold'>AUTHENTIC</tspan>",12)
txt(1081,1398,"outputs: verdict &#183; confidence &#183; mask &#183; boxes",10.5,"middle",MUTED)
txt(1081,1422,"an aggregation of measurement outcomes,",9.8,"middle",MUTED)
txt(1081,1434,"not a classical matching rule",9.8,"middle",MUTED)
txt(1081,1474,"No classical feature extractor, block matcher, offset vote",10,"middle",TITLE)
txt(1081,1490,"or morphology appears anywhere in stages 1&#8211;3.",10,"middle",TITLE)
txt(1081,1514,"Only trainable parameters: 58 circuit rotation angles.",9.8,"middle",MUTED)

A('</g></svg>')
open('docs/figures/architecture_composed.svg','w').write("\n".join(o))
print("wrote docs/figures/architecture_composed.svg")
