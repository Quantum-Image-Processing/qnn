"""Generate the three CMFD architecture figures.
Run:  python docs/figures/gen_figures.py
Outputs: fig1_circuit.svg, fig2_image_processing.svg, fig3_mask_formation.svg
"""
INK, MUTED, TITLE = "#1a1a1a", "#5c5c5c", "#c0392b"
CREAM, CREAMB = "#fdf6e3", "#d6c68a"
GREY,  GREYB  = "#e8e8e8", "#9a9a9a"
BLUE,  BLUEB  = "#d6e6f5", "#7ba7cf"
PINK, TEAL    = "#f6c9c0", "#0f7d88"
GREEN, GREENB = "#e4f0e4", "#79a879"
LILAC, LILACB = "#e8e2f2", "#9b8bc4"

class Fig:
    def __init__(s, w, h, label):
        s.o=[]; s.w=w; s.h=h
        s.o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
                   f'height="{h}" role="img" aria-label="{label}">')
        s.o.append('<rect width="100%" height="100%" fill="#ffffff"/>')
        s.o.append('<defs>'
            f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
            f'<marker id="at" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{TEAL}"/></marker></defs>')
        s.o.append(f'<g font-family="Helvetica, Arial, sans-serif" fill="{INK}">')
    def box(s,x,y,w,h,fill,stroke,rx=3,sw=1.3,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def t(s,x,y,txt,size=11,anchor="middle",fill=INK,weight="normal"):
        s.o.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{txt}</text>')
    def line(s,x1,y1,x2,y2,color=INK,sw=1.2,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')
    def arrow(s,x1,y1,x2,y2,color=INK,sw=1.4,teal=False):
        s.o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{"at" if teal else "a"})"/>')
    def circ(s,x,y,r,fill,stroke=None,sw=1.2):
        st=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
        s.o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"{st}/>')
    def badge(s,x,y,n,col=INK,r=11):
        s.circ(x,y,r,col); s.t(x,y+4,n,12,"middle","#ffffff","bold")
    def save(s,path):
        s.o.append('</g></svg>'); open(path,'w').write("\n".join(s.o)); print("wrote",path)

# ═══════════════════════════ FIGURE 1 — CIRCUIT ═══════════════════════════
def fig1():
    W,H=1420,1420
    f=Fig(W,H,"Gate-level quantum circuit for patch-pair comparison, with an explanation of every "
              "gate: amplitude embedding, RX RY RZ rotations carrying re-uploaded pixel data, a CNOT "
              "entangling ring, offset encoding, and the swap test of Hadamard, six controlled swaps, "
              "two controlled rotations, Hadamard and Pauli-Z measurement.")
    f.t(710,34,"Figure 1 &#8212; Quantum Circuit: what every gate does",15,"middle",TITLE,"bold")
    f.t(710,54,"15 wires &#183; 158 gates &#183; depth 37 &#183; 58 trainable angles &#183; one evaluation compares ONE patch pair",11,"middle",MUTED)

    y0,dy,gp=130,32,16
    yA=[y0+dy*i for i in range(6)]
    yB=[yA[-1]+dy+gp+dy*i for i in range(6)]
    yO=[yB[-1]+dy+gp, yB[-1]+2*dy+gp]
    yAnc=yO[-1]+dy+gp
    XS,XE=230,1250
    for i,y in enumerate([*yA,*yB,*yO,yAnc]):
        teal=(y==yAnc)
        f.line(XS,y,XE,y,TEAL if teal else INK,1.5 if teal else 1.05)
        f.t(XS-12,y+4,f"|0&#10217;<tspan font-size='9'> q{i}</tspan>",10.5,"end",TEAL if teal else INK)
    def brace(y1,y2,lab,sub):
        x=150
        f.o.append(f'<path d="M {x+14} {y1-8} L {x+6} {y1-8} L {x+6} {(y1+y2)/2-5} L {x} {(y1+y2)/2} '
                   f'L {x+6} {(y1+y2)/2+5} L {x+6} {y2+8} L {x+14} {y2+8}" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
        f.t(x-8,(y1+y2)/2-2,lab,11,"end",INK,"bold"); f.t(x-8,(y1+y2)/2+12,sub,9.5,"end",MUTED)
    brace(yA[0],yA[-1],"Register A","patch &#968;&#8320; (64 px)")
    brace(yB[0],yB[-1],"Register B","patch &#968;&#8321; (64 px)")
    brace(yO[0],yO[-1],"Offset","&#916; = (dy,dx)")
    f.t(142,yAnc+4,"Ancilla",11,"end",TEAL,"bold")

    # A: amplitude embedding
    for ys,l in ((yA,"&#968;&#8320;"),(yB,"&#968;&#8321;")):
        f.box(248,ys[0]-13,58,ys[-1]-ys[0]+26,CREAM,CREAMB,sw=1.5)
        f.t(277,(ys[0]+ys[-1])/2-3,"Amp",11,"middle",INK,"bold"); f.t(277,(ys[0]+ys[-1])/2+11,"Embed",11,"middle",INK,"bold")
    f.badge(277,yA[0]-30,"A")
    # B: rotations
    GX={"RX":330,"RY":376,"RZ":422}
    for ys in (yA,yB):
        for y in ys:
            for g,x in GX.items():
                f.box(x,y-11,38,22,"#ffffff",INK,rx=3,sw=1.1); f.t(x+19,y+4,g,9.8)
    f.badge(395,yA[0]-30,"B")
    # C: CNOT ring
    def cnot(xc,yc,yt):
        f.line(xc,yc,xc,yt,INK,1.2); f.circ(xc,yc,3.6,INK)
        f.circ(xc,yt,7.5,"#ffffff",INK,1.2)
        f.line(xc-7.5,yt,xc+7.5,yt,INK,1.2); f.line(xc,yt-7.5,xc,yt+7.5,INK,1.2)
    for ys in (yA,yB):
        for k in range(6): cnot(492+k*24, ys[k], ys[(k+1)%6])
    f.badge(552,yA[0]-30,"C")
    f.o.append(f'<path d="M 324 {yA[0]-19} L 324 {yA[0]-25} L 640 {yA[0]-25} L 640 {yA[0]-19}" fill="none" stroke="{MUTED}" stroke-width="1.1"/>')
    f.t(482,yA[0]-46,"one data re-uploading layer &#8212; repeated &#215;3",10.5,"middle",MUTED)
    # tie
    f.line(482,yA[-1]+14,482,yB[0]-14,TEAL,1.6,"4 3"); f.circ(482,(yA[-1]+yB[0])/2,4,TEAL)
    f.t(500,(yA[-1]+yB[0])/2+4,"shared &#952; &#8212; ONE weight tensor drives both registers (Siamese)",10.5,"start",TEAL,"bold")
    # D: offset
    for y in yO:
        f.box(330,y-11,130,22,BLUE,BLUEB,rx=3,sw=1.2); f.t(395,y+4,"RY(&#966; + &#916;&#183;&#960;)",9.8)
    f.badge(395,yO[0]-26,"D")
    # E-I: swap test
    f.box(690,yAnc-15,32,30,"#ffffff",TEAL,rx=3,sw=1.6); f.t(706,yAnc+5,"H",13,"middle",TEAL,"bold")
    f.badge(706,yAnc-34,"E",TEAL)
    for k in range(6):
        x=760+k*30
        f.line(x,yA[k],x,yAnc,TEAL,1.3); f.circ(x,yAnc,4.5,TEAL)
        for yy in (yA[k],yB[k]):
            f.line(x-6,yy-6,x+6,yy+6,TEAL,1.8); f.line(x+6,yy-6,x-6,yy+6,TEAL,1.8)
    f.badge(835,yA[0]-30,"F",TEAL)
    for k,y in enumerate(yO):
        x=970+k*44
        f.line(x,y,x,yAnc-15,INK,1.2); f.circ(x,y,4.2,INK)
        f.box(x-16,yAnc-15,32,30,BLUE,BLUEB,rx=3,sw=1.2); f.t(x,yAnc+5,"RY",9.8)
    f.badge(992,yO[0]-26,"G")
    f.box(1066,yAnc-15,32,30,"#ffffff",TEAL,rx=3,sw=1.6); f.t(1082,yAnc+5,"H",13,"middle",TEAL,"bold")
    f.badge(1082,yAnc-34,"H",TEAL)
    f.box(1126,yAnc-16,44,32,"#e0f0f1",TEAL,rx=3,sw=1.7)
    f.o.append(f'<path d="M 1136 {yAnc+8} A 12 12 0 0 1 1160 {yAnc+8}" fill="none" stroke="{TEAL}" stroke-width="1.5"/>')
    f.line(1148,yAnc+8,1160,yAnc-8,TEAL,1.5)
    f.badge(1148,yAnc-34,"I",TEAL)
    f.o.append(f'<path d="M 682 {yAnc+30} L 682 {yAnc+38} L 1178 {yAnc+38} L 1178 {yAnc+30}" fill="none" stroke="{TEAL}" stroke-width="1.2"/>')
    f.t(930,yAnc+54,"S W A P   T E S T",11,"middle",TEAL,"bold")

    # ─── legend / per-gate explanation ───
    Y=yAnc+90
    f.t(710,Y,"What each gate does",13,"middle",TITLE,"bold")
    rows=[
      ("A","AmpEmbed","Amplitude Embedding",
       ["Writes the 64 normalised pixel values into the","64 amplitudes of a 6-qubit state."],
       ["This is how an image patch enters the quantum computer.","64 numbers fit in 6 qubits &#8212; log&#8322; compression."]),
      ("B","RX RY RZ","Single-qubit rotations",
       ["Rotate a qubit about the x, y and z axes by","angle &#952;&#7522; + pixel&#7522;&#183;&#960;."],
       ["The trainable part. The pixel is re-uploaded INTO the angle;","without that the swap test cancels &#952; and the gradient is zero."]),
      ("C","CNOT","Controlled-NOT",
       ["Flips the target qubit when the control qubit","is |1&#10217;. Wired as a 6-qubit ring."],
       ["Entangles neighbouring qubits so a layer can express","correlations across the patch, not just per-pixel effects."]),
      ("D","RY(&#966;+&#916;&#183;&#960;)","Offset encoding",
       ["Rotates each offset qubit by an angle set by","the pair's displacement &#916;."],
       ["Puts WHERE the two patches sit relative to each other","into the circuit, alongside what they look like."]),
      ("E","H","Hadamard (open)",
       ["Puts the ancilla into an equal superposition","of |0&#10217; and |1&#10217;."],
       ["Opens the interferometer: &#8220;swap the registers&#8221; and","&#8220;leave them alone&#8221; now happen simultaneously."]),
      ("F","CSWAP &#215;6","Controlled-SWAP (Fredkin)",
       ["Swaps qubit k of Register A with qubit k of","Register B, but only when the ancilla is |1&#10217;."],
       ["Six gates swap the WHOLE 6-qubit state &#8212; this is the","comparison. Fewer than six compares only a fragment,","and measurably inverts the answer."]),
      ("G","CRY &#215;2","Controlled-RY",
       ["Rotates the ancilla by a trainable angle when","an offset qubit is |1&#10217;."],
       ["Lets the displacement modulate the interference, so the","score depends on geometry as well as appearance."]),
      ("H","H","Hadamard (close)",
       ["Second Hadamard on the ancilla."],
       ["Closes the interferometer, turning the phase difference","between swapped and unswapped branches into probability."]),
      ("I","&#10216;Z&#10217;","Pauli-Z measurement",
       ["Measures the ancilla: &#10216;Z&#10217; = P(0) &#8722; P(1)."],
       ["Equals |&#10216;&#968;&#8320;|&#968;&#8321;&#10217;|&#178;, the squared overlap of the two patch","states &#8212; the similarity score, in [0,1]."]),
    ]
    ry=Y+22
    f.box(40,ry,1340,26,GREY,GREYB,rx=2,sw=1)
    for x,h in ((70,"#"),(112,"symbol"),(228,"name"),(420,"operation"),(830,"why it is here")):
        f.t(x,ry+17,h,10,"start",MUTED,"bold")
    ry+=26
    for i,(b,sym,name,op,why) in enumerate(rows):
        nl=max(len(op),len(why)); rh=16*nl+18
        f.box(40,ry,1340,rh,"#ffffff" if i%2 else "#fbfbfa",GREYB,rx=0,sw=0.7)
        f.badge(70,ry+rh/2,b,TEAL if b in "EFGHI" else INK,9.5)
        f.t(112,ry+rh/2+4,sym,10.5,"start",INK,"bold")
        f.t(228,ry+rh/2+4,name,10.5,"start",INK)
        y1=ry+rh/2-(len(op)-1)*8+4
        for j,l in enumerate(op): f.t(420,y1+j*16,l,9.6,"start",MUTED)
        y2=ry+rh/2-(len(why)-1)*8+4
        for j,l in enumerate(why): f.t(830,y2+j*16,l,9.6,"start",MUTED)
        ry+=rh
    f.h=ry+40
    f.o[0]=f.o[0].replace(f'viewBox="0 0 {W} {H}"',f'viewBox="0 0 {W} {int(ry+40)}"').replace(f'height="{H}"',f'height="{int(ry+40)}"')
    f.save('docs/figures/fig1_circuit.svg')

# ══════════════════════ FIGURE 2 — IMAGE PROCESSING ══════════════════════
def fig2():
    W,H=1420,760
    f=Fig(W,H,"Image processing pipeline: an input photograph is converted to grayscale, resized to "
              "128 by 128, tiled into 256 non-overlapping 8 by 8 patches, each patch flattened and "
              "L2-normalised into a 64-amplitude state, and all 32,640 admissible patch pairs formed "
              "with their displacement.")
    f.t(710,34,"Figure 2 &#8212; Image Processing: photograph to quantum states",15,"middle",TITLE,"bold")
    f.t(710,54,"every operation performed on pixels before anything quantum happens",11,"middle",MUTED)

    PY_=100; PH=230
    def panel(x,w,n,head):
        f.box(x,PY_,w,PH,"#ffffff",GREYB)
        f.badge(x+16,PY_-4,n)
        f.t(x+w/2,PY_-14,head,11.5,"middle",INK,"bold")

    # 1 input
    panel(36,236,"1","input photograph")
    f.box(72,124,164,120,GREY,GREYB); f.t(154,190,"any size, RGB",10.5,"middle",MUTED)
    f.t(154,272,"read as uploaded &#8212; no",9.8,"middle",MUTED)
    f.t(154,284,"assumption about size or format",9.8,"middle",MUTED)
    f.arrow(276,215,314,215)

    # 2 grayscale + resize
    panel(320,236,"2","grayscale + resize")
    f.box(356,124,164,84,GREY,GREYB)
    f.t(438,170,"L channel only",10.5,"middle",MUTED)
    f.t(438,228,"LANCZOS &#8594; <tspan font-weight='bold'>128 &#215; 128</tspan>",10.5)
    f.t(438,272,"colour is discarded: a duplicated",9.8,"middle",MUTED)
    f.t(438,284,"region shares structure, not hue",9.8,"middle",MUTED)
    f.arrow(560,215,598,215)

    # 3 tile
    panel(604,236,"3","tile into patches")
    gx,gy,c=640,120,9
    for r in range(16):
        for k in range(16):
            fill=PINK if (r,k) in ((3,3),(11,11)) else GREY
            f.o.append(f'<rect x="{gx+k*c}" y="{gy+r*c}" width="{c}" height="{c}" fill="{fill}" stroke="{GREYB}" stroke-width="0.4"/>')
    f.t(722,282,"stride = patch = 8 px &#8594; <tspan font-weight='bold'>16&#215;16 = 256</tspan>",10.5)
    f.t(722,300,"non-overlapping, 100% coverage",9.8,"middle",MUTED)
    f.arrow(844,215,882,215)

    # 4 flatten + normalise
    panel(888,236,"4","flatten + normalise")
    for r in range(8):
        for k in range(8):
            f.o.append(f'<rect x="{908+k*10}" y="{120+r*10}" width="10" height="10" fill="{PINK}" stroke="{GREYB}" stroke-width="0.4"/>')
    f.t(948,214,"8&#215;8",9.5,"middle",MUTED)
    f.arrow(994,160,1018,160)
    for i in range(7):
        f.box(1026+i*11,146,10,28,CREAM,CREAMB,rx=1,sw=0.8)
    f.t(1110,164,"&#8230;",11)
    f.t(1064,214,"64 values",9.5,"middle",MUTED)
    f.t(1006,258,"&#968; = x / &#8214;x&#8214;  (L2 norm)",11,"middle",INK,"bold")
    f.t(1006,282,"64 amplitudes = 2&#8310; &#8594; fits exactly in <tspan font-weight='bold'>6 qubits</tspan>",9.8,"middle",MUTED)
    f.t(1006,300,"normalising also removes per-patch brightness",9.8,"middle",MUTED)
    f.arrow(1128,215,1166,215)

    # 5 pairs
    panel(1172,212,"5","form pairs")
    for r in range(8):
        for k in range(8):
            fill=PINK if (r,k) in ((1,1),(6,6)) else GREY
            f.o.append(f'<rect x="{1204+k*11}" y="{122+r*11}" width="11" height="11" fill="{fill}" stroke="{GREYB}" stroke-width="0.5"/>')
    f.line(1204+1.5*11,122+1.5*11,1204+6.5*11,122+6.5*11,TITLE,1.5,"4 3")
    f.t(1302,175,"&#916;",11,"middle",TITLE,"bold")
    f.t(1278,282,"all patches &#8805; 8 px apart",10.5)
    f.t(1278,300,"<tspan font-weight='bold'>32,640 pairs</tspan> + &#916; each",10.5)

    # summary strip
    f.box(36,378,1348,120,CREAM,CREAMB,sw=1.4)
    f.t(710,404,"What leaves this stage",12.5,"middle",INK,"bold")
    f.t(710,430,"&#968;&#8320; , &#968;&#8321; &#8712; &#8477;&#8310;&#8308;  (two normalised patch vectors)   +   &#916; = (dy, dx) / 128  (their displacement)   &#8212;   repeated for all 32,640 pairs",11.5)
    f.t(710,458,"No descriptor is computed. No DCT, no histogram, no keypoints, no edges. The circuit receives raw normalised pixels and nothing else,",10.2,"middle",MUTED)
    f.t(710,474,"which is what makes the model &#8220;pure&#8221;: no classical algorithm has decided in advance which parts of the patch matter.",10.2,"middle",MUTED)

    # cost note
    f.box(36,522,1348,86,GREY,GREYB)
    f.t(710,548,"Why the pair count dominates everything",12,"middle",INK,"bold")
    f.t(710,572,"N patches produce N(N&#8722;1)/2 pairs, and each pair costs one circuit evaluation (32 ms). 256 patches &#8594; 32,640 pairs &#8594; &#8776; 17 min per image.",10.2,"middle",MUTED)
    f.t(710,590,"Halving the patch size quadruples N and multiplies the pair count by &#8776;16 &#8212; so working resolution, not gate count, sets the runtime.",10.2,"middle",MUTED)

    f.t(710,650,"Stride is set EQUAL to patch size so the tiles cover the image completely.",11,"middle",TITLE,"bold")
    f.t(710,670,"With a larger stride the tiles leave gaps: at 256 px with stride 32, only 3 of 64 patches fell inside a 2%-area forged region,",10.2,"middle",MUTED)
    f.t(710,686,"and a forged image and its authentic original produced byte-identical output because the forgery was never sampled.",10.2,"middle",MUTED)
    f.save('docs/figures/fig2_image_processing.svg')

# ═══════════════════════ FIGURE 3 — MASK FORMATION ═══════════════════════
def fig3():
    W,H=1420,720
    f=Fig(W,H,"Mask formation: 32,640 circuit scores are thresholded at 0.90, both patches of every "
              "matched pair are painted into a 128 by 128 mask, the mask is rescaled to the original "
              "resolution, connected components give bounding boxes, and the matched-pair count gives "
              "the forged or authentic verdict.")
    f.t(710,34,"Figure 3 &#8212; Mask Formation: scores back into an image",15,"middle",TITLE,"bold")
    f.t(710,54,"how 32,640 measured numbers become a localised forgery mask",11,"middle",MUTED)

    PY_=100; PH=250
    def panel(x,w,n,head,fill="#ffffff",stroke=GREYB):
        f.box(x,PY_,w,PH,fill,stroke)
        f.badge(x+16,PY_-4,n); f.t(x+w/2,PY_-14,head,11.5,"middle",INK,"bold")

    # 1 threshold
    panel(36,286,"1","threshold the scores")
    for i,h in enumerate([12,20,32,50,72,54,36,24,16]):
        col=TEAL if i>=6 else GREY
        f.o.append(f'<rect x="{78+i*24}" y="{242-h}" width="20" height="{h}" fill="{col}" stroke="{GREYB}" stroke-width="0.6"/>')
    f.line(78+6*24-4,132,78+6*24-4,252,TITLE,1.5,"4 3")
    f.t(232,128,"&#10216;Z&#10217; &#8805; 0.90",10.5,"middle",TITLE,"bold")
    f.t(179,266,"32,640 circuit scores",9.8,"middle",MUTED)
    f.t(179,296,"keep pairs scoring &#8805; threshold",10.5)
    f.t(179,320,"the ONLY decision taken from the",9.8,"middle",MUTED)
    f.t(179,334,"circuit output &#8212; everything after",9.8,"middle",MUTED)
    f.t(179,348,"this is bookkeeping",9.8,"middle",MUTED)
    f.arrow(326,215,364,215)

    # 2 paint mask
    panel(370,286,"2","paint both patches")
    f.o.append(f'<rect x="424" y="120" width="178" height="130" fill="#111111"/>')
    f.o.append(f'<rect x="{424+2*20}" y="{120+2*16}" width="40" height="32" fill="#ffffff"/>')
    f.o.append(f'<rect x="{424+5*20}" y="{120+5*16}" width="40" height="32" fill="#ffffff"/>')
    f.line(424+2*20+40,120+2*16+16,424+5*20,120+5*16+16,TITLE,1.3,"4 3")
    f.t(513,266,"128 &#215; 128 binary mask",9.8,"middle",MUTED)
    f.t(513,296,"mark <tspan font-weight='bold'>BOTH</tspan> patches of each matched pair",10.5)
    f.t(513,320,"a copy-move has two halves &#8212; the source",9.8,"middle",MUTED)
    f.t(513,334,"region and the pasted copy. Marking only",9.8,"middle",MUTED)
    f.t(513,348,"one would be half an answer.",9.8,"middle",MUTED)
    f.arrow(664,215,702,215)

    # 3 rescale + components
    panel(708,286,"3","rescale &#8594; bounding boxes")
    f.o.append(f'<rect x="762" y="120" width="178" height="130" fill="{GREY}" stroke="{GREYB}"/>')
    f.o.append(f'<rect x="{762+2*20-5}" y="{120+2*16-5}" width="50" height="42" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
    f.o.append(f'<rect x="{762+5*20-5}" y="{120+5*16-5}" width="50" height="42" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
    f.t(851,266,"original resolution",9.8,"middle",MUTED)
    f.t(851,296,"connected components &#8594; boxes",10.5)
    f.t(851,320,"display formatting only: it runs after",9.8,"middle",MUTED)
    f.t(851,334,"the verdict and feeds nothing back into",9.8,"middle",MUTED)
    f.t(851,348,"it, the mask, or any reported metric.",9.8,"middle",MUTED)
    f.arrow(1002,215,1040,215)

    # 4 verdict
    panel(1046,338,"4","verdict",GREEN,GREENB)
    f.t(1215,148,"count the matched pairs",10.5,"middle",MUTED)
    f.box(1080,164,270,40,"#ffffff",GREENB,rx=3)
    f.t(1215,190,"&#8805; 8 pairs  &#8594;  <tspan font-weight='bold' fill='"+TITLE+"'>FORGED</tspan>",12)
    f.box(1080,212,270,40,"#ffffff",GREENB,rx=3)
    f.t(1215,238,"&lt; 8 pairs  &#8594;  <tspan font-weight='bold'>AUTHENTIC</tspan>",12)
    f.t(1215,286,"outputs: verdict &#183; confidence &#183; mask &#183; boxes",10.2,"middle",MUTED)
    f.t(1215,314,"an aggregation of measurement outcomes,",9.8,"middle",MUTED)
    f.t(1215,328,"not a classical matching rule &#8212; it uses no",9.8,"middle",MUTED)
    f.t(1215,342,"patch content and no geometry",9.8,"middle",MUTED)

    # what was removed
    f.box(36,410,1348,120,GREY,GREYB)
    f.t(710,436,"What this stage does NOT do",12.5,"middle",TITLE,"bold")
    removed=["DCT / histogram descriptor","lexicographic block matching","(dy, dx) Hough vote","binary dilation / erosion","RANSAC"]
    x=110
    for r in removed:
        w=len(r)*6.0+26
        f.box(x,456,w,28,"#ffffff",GREYB,rx=2,sw=1)
        f.t(x+w/2,474,r,9.8,"middle",MUTED)
        f.line(x+10,470,x+w-10,470,TITLE,1.2)
        x+=w+14
    f.t(710,508,"None of these appears anywhere in the pipeline. The mask is built directly from thresholded circuit outputs.",10.2,"middle",MUTED)

    # honest note
    f.box(36,552,1348,120,"#ffffff",TITLE,sw=1.4)
    f.t(710,578,"Known limitation of removing the offset vote",12.5,"middle",TITLE,"bold")
    f.t(710,604,"Offset consistency &#8212; &#8220;many matched pairs share ONE displacement&#8221; &#8212; is a property of the pair ensemble, not of any single pair.",10.4,"middle",INK)
    f.t(710,622,"A circuit that sees one pair at a time cannot observe clustering it is never shown, so the offset register in Figure 1 is a weaker",10.4,"middle",INK)
    f.t(710,640,"substitute than the classical vote it replaced. A faithful quantum version needs quantum counting over all pairs at once (QRAM).",10.4,"middle",INK)
    f.t(710,660,"Expect more false positives on homogeneous regions &#8212; sky, walls, smooth concrete &#8212; than a vote-based pipeline would give.",10.2,"middle",MUTED)
    f.save('docs/figures/fig3_mask_formation.svg')

fig1(); fig2(); fig3()
