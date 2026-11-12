"""Generate phase figures 1, 2 and 4 for the 9-wire Zernike swap-test QNN.
Run from the repo root:  python docs/figures/gen_zernike_modules.py
Outputs: phase1_preprocessing.svg, phase2_qnn_circuit.svg, phase4_batch_output.svg (phase 3: gen_siamese_swaptest.py)

Every number drawn here comes from code/cmfd_zernike_swaptest.py, train_zernike.py,
inference_zernike.py, models/zernike_9w_calibration.json, or a measured run.
Same visual style as gen_figures.py.
"""
INK, MUTED, TITLE = "#1a1a1a", "#5c5c5c", "#c0392b"
CREAM, CREAMB = "#fdf6e3", "#d6c68a"
GREY,  GREYB  = "#e8e8e8", "#9a9a9a"
BLUE,  BLUEB  = "#d6e6f5", "#7ba7cf"
PINK, TEAL    = "#f6c9c0", "#0f7d88"
GREEN, GREENB = "#e4f0e4", "#79a879"
AMBER, AMBERB = "#ffb300", "#8a5c00"
QBLUE = "#0d47a1"


class Fig:
    def __init__(s, w, h, label):
        s.o = []; s.w = w; s.h = h
        s.o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
                   f'height="{h}" role="img" aria-label="{label}">')
        s.o.append('<rect width="100%" height="100%" fill="#ffffff"/>')
        s.o.append('<defs>'
            f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>')
        s.o.append(f'<g font-family="Helvetica, Arial, sans-serif" fill="{INK}">')
    def box(s,x,y,w,h,fill,stroke,rx=3,sw=1.3,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def t(s,x,y,txt,size=11,anchor="middle",fill=INK,weight="normal"):
        s.o.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{txt}</text>')
    def line(s,x1,y1,x2,y2,color=INK,sw=1.2,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')
    def arrow(s,x1,y1,x2,y2,color=INK,sw=1.4):
        s.o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"/>')
    def circ(s,x,y,r,fill,stroke=None,sw=1.2):
        st=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
        s.o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"{st}/>')
    def badge(s,x,y,n,col=INK,r=11):
        s.circ(x,y,r,col); s.t(x,y+4,n,12,"middle","#ffffff","bold")
    def save(s,path):
        s.o.append('</g></svg>'); open(path,'w').write("\n".join(s.o)); print("wrote",path)

    def header(s, title, sub, active):
        """Title, subtitle and the four-phase roadmap with this phase highlighted."""
        s.t(s.w/2,34,title,15,"middle",TITLE,"bold")
        s.t(s.w/2,54,sub,11,"middle",MUTED)
        names = ["Phase 1  Ingestion + pre-processing", "Phase 2  QNN circuit", "Phase 3  Swap test + Siamese QNN", "Phase 4  Batch extraction + output"]
        bw, gap = 300, 36
        x0 = s.w/2 - (4*bw + 3*gap)/2
        for i, n in enumerate(names):
            x = x0 + i*(bw+gap)
            on = (i+1 == active)
            s.box(x,70,bw,28,TITLE if on else "#ffffff",TITLE if on else GREYB,rx=14,sw=1.3)
            s.t(x+bw/2,89,n,11,"middle","#ffffff" if on else MUTED,"bold" if on else "normal")
            if i < 3:
                s.arrow(x+bw+6,84,x+bw+gap-6,84,GREYB)

    def panel(s,x,y,w,h,n,head,fill="#ffffff",stroke=GREYB):
        s.box(x,y,w,h,fill,stroke)
        s.badge(x+16,y-4,n); s.t(x+w/2,y-14,head,11.5,"middle",INK,"bold")


# ═════════════════════ MODULE 1 — PRE-PROCESSING ═════════════════════
def module1():
    W,H=1420,650
    f=Fig(W,H,"Phase 1, ingestion and pre-processing: the photograph is converted to grayscale, resized to 128 by 128, "
              "tiled into 256 patches of 8 by 8, each patch is reduced to 12 rotation-invariant Zernike "
              "magnitudes, padded to 16 and normalised into a 4-qubit amplitude state.")
    f.header("Phase 1 &#8212; Ingestion and pre-processing: photograph to 4-qubit Zernike states",
             "classical data preparation &#183; fixed, nothing here is trained &#183; runs once per image", 1)
    PY,PH=150,230
    Y=lambda v: v+50          # same in-panel layout as Figure 2, shifted down

    f.panel(36,PY,236,PH,"1","input photograph")
    f.box(72,Y(124),164,120,GREY,GREYB); f.t(154,Y(190),"any size, RGB",10.5,"middle",MUTED)
    f.t(154,Y(272),"JPEG or PNG, read as uploaded",9.8,"middle",MUTED)
    f.t(154,Y(284),"no assumption about size",9.8,"middle",MUTED)
    f.arrow(276,Y(215),314,Y(215))

    f.panel(320,PY,236,PH,"2","grayscale + resize")
    f.box(356,Y(124),164,84,GREY,GREYB); f.t(438,Y(170),"L channel only",10.5,"middle",MUTED)
    f.t(438,Y(228),"LANCZOS &#8594; <tspan font-weight='bold'>128 &#215; 128</tspan>",10.5)
    f.t(438,Y(272),"pixel values scaled to [0, 1]",9.8,"middle",MUTED)
    f.arrow(560,Y(215),598,Y(215))

    f.panel(604,PY,236,PH,"3","tile into patches")
    gx,gy,c=640,Y(120),9
    for r in range(16):
        for k in range(16):
            fill=PINK if (r,k) in ((3,3),(11,11)) else GREY
            f.o.append(f'<rect x="{gx+k*c}" y="{gy+r*c}" width="{c}" height="{c}" fill="{fill}" stroke="{GREYB}" stroke-width="0.4"/>')
    f.t(722,Y(282),"8 &#215; 8, stride 8 &#8594; <tspan font-weight='bold'>16&#215;16 = 256</tspan>",10.5)
    f.t(722,Y(300),"non-overlapping, 100% coverage",9.8,"middle",MUTED)
    f.arrow(844,Y(215),882,Y(215))

    # 4 Zernike: patch with inscribed unit disc and polar sample grid
    f.panel(888,PY,236,PH,"4","Zernike moments")
    px,py,cs=904,Y(124),10
    for r in range(8):
        for k in range(8):
            f.o.append(f'<rect x="{px+k*cs}" y="{py+r*cs}" width="{cs}" height="{cs}" fill="{PINK}" stroke="{GREYB}" stroke-width="0.4"/>')
    import math
    cx,cy,R=px+40,py+40,40
    f.circ(cx,cy,R,"none",TITLE,1.4)
    for a in range(8):
        th=2*math.pi*a/8
        f.line(cx,cy,cx+R*math.cos(th),cy+R*math.sin(th),TITLE,0.6)
    for rr in (0.35,0.7):
        for a in range(16):
            th=2*math.pi*a/16
            f.circ(cx+R*rr*math.cos(th),cy+R*rr*math.sin(th),1.3,TITLE)
    f.arrow(990,Y(164),1012,Y(164))
    hs=[30,22,26,12,18,9,15,11,7,10,6,5]
    for i,h in enumerate(hs):
        f.box(1018+i*8,Y(184)-h,7,h,CREAM,CREAMB,rx=0.5,sw=0.7)
    f.t(1066,Y(198),"12 values",9.5,"middle",MUTED)
    f.t(944,Y(222),"13 radii &#215; 32 angles",9.5,"middle",MUTED)
    f.t(1006,Y(246),"keep |Z(n,m)|, orders n &#8804; 5",11,"middle",INK,"bold")
    f.t(1006,Y(268),"rotating a patch only changes the",9.8,"middle",MUTED)
    f.t(1006,Y(282),"phase, so the magnitudes ignore it",9.8,"middle",MUTED)
    f.t(1006,Y(300),"one fixed 12 &#215; 64 matrix, not trained",9.8,"middle",MUTED)
    f.arrow(1128,Y(215),1166,Y(215))

    # 5 pad + normalise
    f.panel(1172,PY,212,PH,"5","pad + normalise")
    for i in range(16):
        r,k=divmod(i,4)
        f.box(1238+k*18,Y(124)+r*18,16,16,CREAM if i<12 else GREY,CREAMB if i<12 else GREYB,rx=1,sw=0.8)
        if i>=12: f.t(1238+k*18+8,Y(124)+r*18+12,"0",9,"middle",MUTED)
    f.t(1278,Y(214),"12 moments + 4 zeros",9.5,"middle",MUTED)
    f.t(1278,Y(246),"&#968; = z / &#8214;z&#8214;",11,"middle",INK,"bold")
    f.t(1278,Y(268),"16 = 2&#8308; amplitudes &#8594; <tspan font-weight='bold'>4 qubits</tspan>",9.8,"middle",MUTED)
    f.t(1278,Y(282),"a blank patch gets the",9.8,"middle",MUTED)
    f.t(1278,Y(296),"uniform state",9.8,"middle",MUTED)

    f.box(36,412,1348,100,CREAM,CREAMB,sw=1.4)
    f.t(710,438,"What leaves this phase",12.5,"middle",INK,"bold")
    f.t(710,462,"256 vectors &#968;&#7522; &#8712; &#8477;&#185;&#8310;, one per patch, each ready to load onto a 4-qubit register in Phase 2",11.5)
    f.t(710,486,"The earlier v3 model sent 64 raw pixels to 6 qubits per patch. Sending 12 Zernike magnitudes to 4 qubits is what cuts the circuit from 15 wires to 9.",10.2,"middle",MUTED)

    f.box(36,530,1348,96,GREY,GREYB)
    f.t(710,556,"Rotation handling comes from this phase, not from the circuit",12,"middle",INK,"bold")
    f.t(710,580,"A patch rotated by 45&#176; finds its true source among all patches 94.5% of the time with Zernike vectors, against 3.0% with raw pixels (200 queries, 20 test images).",10.2,"middle",MUTED)
    f.t(710,598,"Adding the trained circuit keeps this (93.0%), and so does removing its CNOTs (90.0%): entanglement does not create it.",10.2,"middle",MUTED)
    f.save('docs/figures/phase1_preprocessing.svg')


# ═════════════════════ MODULE 2 — QUANTUM CIRCUIT ═════════════════════
def module2():
    W,H=1420,720
    f=Fig(W,H,"Phase 2, QNN circuit: two 4-qubit registers are amplitude-encoded with the Zernike states "
              "of patches i and j, processed by three tied data re-uploading layers sharing 36 trained "
              "angles, and compared by a swap test on one ancilla, 9 wires in total.")
    f.header("Phase 2 &#8212; QNN circuit: 9-wire Siamese circuit with data re-uploading",
             "9 wires &#183; 104 gates &#183; 36 trained rotation angles &#183; one evaluation scores ONE patch pair", 2)

    yA=[170,204,238,272]; yB=[338,372,406,440]; yAnc=500
    XS,XE=196,880
    for i,y in enumerate([*yA,*yB,yAnc]):
        anc=(y==yAnc)
        f.line(XS,y,XE,y,QBLUE if anc else INK,2.4 if anc else 1.6)
        f.t(XS-8,y+4,"|0&#10217;",11.5,"end",QBLUE if anc else INK,"bold")
        f.t(XS-36,y+4,f"q{i}",9.5,"end",MUTED)

    def brace(y1,y2,lab,sub,col):
        x=140
        f.o.append(f'<path d="M {x+12} {y1-9} L {x+4} {y1-9} L {x+4} {(y1+y2)/2-5} L {x-2} {(y1+y2)/2} '
                   f'L {x+4} {(y1+y2)/2+5} L {x+4} {y2+9} L {x+12} {y2+9}" fill="none" stroke="{col}" stroke-width="2"/>')
        f.t(x-8,(y1+y2)/2-2,lab,11.5,"end",col,"bold"); f.t(x-8,(y1+y2)/2+13,sub,9.5,"end",MUTED)
    brace(yA[0],yA[-1],"REGISTER A","patch i (16 amp.)","#a16800")
    brace(yB[0],yB[-1],"REGISTER B","patch j (16 amp.)","#a16800")
    f.t(132,yAnc+4,"ANCILLA",11.5,"end",QBLUE,"bold")

    for ys in (yA,yB):
        f.box(208,ys[0]-14,52,ys[-1]-ys[0]+28,AMBER,AMBERB,sw=2)
        f.t(234,(ys[0]+ys[-1])/2-3,"AMP",10.5,"middle",INK,"bold")
        f.t(234,(ys[0]+ys[-1])/2+11,"EMBED",10.5,"middle",INK,"bold")
        for y in ys:
            for g,x in (("RX",278),("RY",316),("RZ",354)):
                f.box(x,y-11,34,22,GREEN,"#3d7a3d",rx=3,sw=1.6); f.t(x+17,y+4,g,10,"middle",INK,"bold")
        for k in range(4):
            xc=410+k*22; yc,yt=ys[k],ys[(k+1)%4]
            f.line(xc,yc,xc,yt,INK,1.6); f.circ(xc,yc,3.5,INK)
            f.circ(xc,yt,7,"#ffffff",INK,1.6); f.line(xc-7,yt,xc+7,yt,INK,1.4); f.line(xc,yt-7,xc,yt+7,INK,1.4)

    f.o.append(f'<path d="M 274 {yA[0]-24} L 274 {yA[0]-30} L 486 {yA[0]-30} L 486 {yA[0]-24}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
    f.t(380,yA[0]-38,"RE-UPLOADING LAYER &#215;3: angle = w + 2&#960;&#183;&#968;&#8342; , then CNOT ring",11,"middle",INK,"bold")

    f.line(333,yA[-1]+14,333,yB[0]-14,QBLUE,2.2,"6 4"); f.circ(333,(yA[-1]+yB[0])/2,4,QBLUE)
    f.t(344,(yA[-1]+yB[0])/2+4,"same w on both registers (Siamese)",10.5,"start",QBLUE,"bold")

    f.box(540,yAnc-14,28,28,"#ffffff",QBLUE,rx=3,sw=2.2); f.t(554,yAnc+5,"H",13,"middle",QBLUE,"bold")
    for k in range(4):
        x=600+k*30
        f.line(x,yA[k],x,yAnc,QBLUE,1.8); f.circ(x,yAnc,4,QBLUE)
        for yy in (yA[k],yB[k]):
            f.line(x-6,yy-6,x+6,yy+6,QBLUE,2); f.line(x+6,yy-6,x-6,yy+6,QBLUE,2)
    f.t(645,yA[0]-38,"CSWAP &#215; 4",12,"middle",QBLUE,"bold")
    f.t(645,yA[0]-22,"swap register A &#8596; B",9.5,"middle",MUTED)
    f.box(746,yAnc-14,28,28,"#ffffff",QBLUE,rx=3,sw=2.2); f.t(760,yAnc+5,"H",13,"middle",QBLUE,"bold")
    f.box(800,yAnc-16,44,32,QBLUE,"#08316b",rx=3,sw=1.6)
    f.o.append(f'<path d="M 810 {yAnc+9} A 12 12 0 0 1 834 {yAnc+9}" fill="none" stroke="#ffffff" stroke-width="2"/>')
    f.line(822,yAnc+9,834,yAnc-8,"#ffffff",2)
    f.t(822,yAnc-26,"&#10216;Z&#10217; &#8712; [0,1]",11,"middle",QBLUE,"bold")
    f.o.append(f'<path d="M 534 {yAnc+30} L 534 {yAnc+38} L 850 {yAnc+38} L 850 {yAnc+30}" fill="none" stroke="{QBLUE}" stroke-width="1.6"/>')
    f.t(692,yAnc+54,"SWAP TEST",11.5,"middle",QBLUE,"bold")

    # right: what is trained, how it was trained
    RX,RW=920,464
    f.panel(RX,150,RW,150,"A","TRAINED: the green gates",GREEN,GREENB)
    f.t(RX+RW/2,192,"36 trained rotation angles   w &#8712; &#8477;&#179;&#739;&#8308;&#739;&#179;",13,"middle",INK,"bold")
    f.t(RX+RW/2,220,"the RX, RY, RZ angles: 3 layers &#215; 4 qubits &#215; 3 axes",10.5,"middle",INK)
    f.t(RX+RW/2,240,"one tensor shared by both registers, learned by Adam",10.5,"middle",INK)
    f.t(RX+RW/2,270,"every trained parameter is a gate angle; no classical layer",10.5,"middle",MUTED)

    f.panel(RX,334,RW,210,"B","how it was trained")
    rows=[("data","140 CoMoFoD training scenes, authentic images only"),
          ("pairs","per image: 6 copied + 6 non-matching patch pairs"),
          ("","copies rotated by 90&#176;&#183;k, with brightness, noise, JPEG, blended paste"),
          ("loss","squared error of &#10216;Z&#10217; against 1 (copy) or 0 (other)"),
          ("optimiser","Adam, learning rate 0.1 &#8594; 0.01 (cosine), adjoint gradients"),
          ("length","10 epochs &#215; 80 images = 800 updates, 132 s on one CPU thread"),
          ("kept","epoch 9, best of 20 separate validation scenes"),
          ("learned","validation pair AUC 0.584 &#8594; 0.676")]
    for i,(k,v) in enumerate(rows):
        y=362+i*22
        f.t(RX+86,y,k,10,"end",INK,"bold"); f.t(RX+96,y,v,10,"start",INK if k!="learned" else TEAL,"bold" if k=="learned" else "normal")

    f.box(36,584,1348,112,"#f5f5f5",GREYB,sw=1.4)
    f.t(710,610,"&#10216;Z&#10217; = |&#10216;&#966;&#7522;|&#966;&#11388;&#10217;|&#178;  &#8212;  the measured overlap of the two processed patch states = their similarity score",12.5,"middle",INK,"bold")
    f.t(600,634,"identical patches &#8594; &#10216;Z&#10217; = 1",10.8,"middle",QBLUE,"bold")
    f.t(820,634,"unrelated patches &#8594; &#10216;Z&#10217; near 0",10.8,"middle",TITLE,"bold")
    f.t(710,662,"Change from v3:  15 &#8594; 9 wires  &#183;  6 &#8594; 4 qubits per register  &#183;  2-qubit offset register removed (its trained gain was almost constant, 0.992&#8211;0.998)",10.4,"middle",INK)
    f.t(710,680,"58 &#8594; 36 trained angles  &#183;  158 &#8594; 104 gates  &#183;  rotated copies added to training  &#183;  fixed validation scenes added",10.4,"middle",INK)
    f.save('docs/figures/phase2_qnn_circuit.svg')


# ═════════════════════ MODULE 3 — BATCH EVALUATION ═════════════════════
def module3():
    W,H=1420,700
    f=Fig(W,H,"Phase 4, batch extraction and output: the register circuit is run once per patch in one batch, all 32,640 "
              "pair scores are formed at once, pairs scoring at least 0.6 are kept, 8,192 or more kept pairs "
              "gives a forged verdict, and both patches of each match are painted into the mask.")
    f.header("Phase 4 &#8212; Batch extraction and output: 32,640 pair scores to a verdict",
             "every pair in the image is scored, then one count decides forged or authentic", 4)
    PY,PH=150,230
    Y=lambda v: v+50

    f.panel(36,PY,236,PH,"1","one state per patch")
    for i in range(6):
        f.box(64+i*8,Y(124)+i*8,56,46,CREAM,CREAMB,rx=2,sw=0.9)
    f.t(132,Y(196),"256 &#968;&#7522;",10,"middle",MUTED)
    f.arrow(174,Y(170),200,Y(170))
    for i in range(6):
        f.box(206+i*5,Y(130)+i*6,34,34,BLUE,BLUEB,rx=2,sw=0.9)
    f.t(240,Y(212),"256 &#966;&#7522;",10,"middle",MUTED)
    f.t(154,Y(246),"&#966;&#7522; = embed + 3 layers on &#968;&#7522;",11,"middle",INK,"bold")
    f.t(154,Y(268),"all 256 patches in one batch",9.8,"middle",MUTED)
    f.t(154,Y(282),"(4 qubits each, same trained w)",9.8,"middle",MUTED)
    f.arrow(276,Y(215),314,Y(215))

    f.panel(320,PY,236,PH,"2","all pairs at once")
    gx,gy,c=382,Y(120),7
    for r in range(16):
        for k in range(16):
            if k>r:
                v=((r*37+k*91+r*k*13)%100)/100
                col=TEAL if v>0.75 else (BLUEB if v>0.4 else BLUE)
            else:
                col="#ffffff"
            f.o.append(f'<rect x="{gx+k*c}" y="{gy+r*c}" width="{c}" height="{c}" fill="{col}" stroke="{GREYB}" stroke-width="0.3"/>')
    f.t(438,Y(246),"S&#7522;&#11388; = |&#10216;&#966;&#7522;|&#966;&#11388;&#10217;|&#178;  for all i &lt; j",11,"middle",INK,"bold")
    f.t(438,Y(268),"<tspan font-weight='bold'>32,640 pairs</tspan>, each equal to its",9.8,"middle",MUTED)
    f.t(438,Y(282),"9-wire swap test (difference &lt; 10&#8315;&#185;&#8309;)",9.8,"middle",MUTED)
    f.arrow(560,Y(215),598,Y(215))

    f.panel(604,PY,236,PH,"3","threshold the scores")
    for i,h in enumerate([18,34,58,72,60,44,30,20,12]):
        col=TEAL if i>=5 else GREY
        f.o.append(f'<rect x="{634+i*20}" y="{Y(222)-h}" width="17" height="{h}" fill="{col}" stroke="{GREYB}" stroke-width="0.6"/>')
    f.line(634+5*20-2,Y(126),634+5*20-2,Y(228),TITLE,1.5,"4 3")
    f.t(732,Y(122)-4,"S &#8805; 0.6",10.5,"middle",TITLE,"bold")
    f.t(722,Y(246),"keep pairs with S &#8805; 0.6",11,"middle",INK,"bold")
    f.t(722,Y(268),"the only decision taken from",9.8,"middle",MUTED)
    f.t(722,Y(282),"the circuit outputs",9.8,"middle",MUTED)
    f.arrow(844,Y(215),882,Y(215))

    f.panel(888,PY,236,PH,"4","verdict",GREEN,GREENB)
    f.t(1006,Y(140),"count the kept pairs",10.5,"middle",MUTED)
    f.box(906,Y(152),200,34,"#ffffff",GREENB)
    f.t(1006,Y(174),"&#8805; 8,192  &#8594;  <tspan font-weight='bold' fill='"+TITLE+"'>FORGED</tspan>",11.5)
    f.box(906,Y(194),200,34,"#ffffff",GREENB)
    f.t(1006,Y(216),"&lt; 8,192  &#8594;  <tspan font-weight='bold'>AUTHENTIC</tspan>",11.5)
    f.t(1006,Y(254),"0.6 and 8,192 chosen on 20",9.8,"middle",MUTED)
    f.t(1006,Y(268),"validation scenes, never on",9.8,"middle",MUTED)
    f.t(1006,Y(282),"the test scenes",9.8,"middle",MUTED)
    f.arrow(1128,Y(215),1166,Y(215))

    f.panel(1172,PY,212,PH,"5","mask + boxes")
    f.o.append(f'<rect x="1208" y="{Y(120)}" width="140" height="100" fill="#111111"/>')
    f.o.append(f'<rect x="1230" y="{Y(138)}" width="30" height="24" fill="#ffffff"/>')
    f.o.append(f'<rect x="1300" y="{Y(180)}" width="30" height="24" fill="#ffffff"/>')
    f.o.append(f'<rect x="1226" y="{Y(134)}" width="38" height="32" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
    f.o.append(f'<rect x="1296" y="{Y(176)}" width="38" height="32" fill="none" stroke="{TITLE}" stroke-width="1.8"/>')
    f.line(1260,Y(150),1300,Y(192),TITLE,1.2,"4 3")
    f.t(1278,Y(246),"paint BOTH patches",11,"middle",INK,"bold")
    f.t(1278,Y(268),"of every kept pair; boxes",9.8,"middle",MUTED)
    f.t(1278,Y(282),"are for display only",9.8,"middle",MUTED)

    # latency, measured on one 128 x 128 image
    f.box(36,412,1348,150,CREAM,CREAMB,sw=1.4)
    f.t(710,438,"Latency per 128 &#215; 128 image (measured)",12.5,"middle",INK,"bold")
    bars=[("v3, 15 wires, one circuit per pair",14.76,"14.8 min (about 20 min reported in the dashboard)",GREYB),
          ("9 wires, one circuit per pair",3.5,"3.3&#8211;3.7 min",BLUEB),
          ("9 wires, batch evaluation above",0.015,"0.7&#8211;0.9 s",TEAL)]
    for i,(lab,mins,val,col) in enumerate(bars):
        y=462+i*30
        f.t(330,y+13,lab,10.5,"end",INK,"bold" if i==2 else "normal")
        w=max(3,mins/14.76*640)
        f.box(342,y,w,18,col,col,rx=2,sw=0.5)
        f.t(342+w+10,y+13,val,10.5,"start",INK,"bold")

    f.box(36,580,1348,96,"#ffffff",TITLE,sw=1.4)
    f.t(710,606,"What this phase does not fix",12.5,"middle",TITLE,"bold")
    f.t(710,630,"Each pair is still scored on its own, and with 32,640 pairs per image natural look-alikes outnumber real copies. On 40 held-out scenes",10.4,"middle",INK)
    f.t(710,648,"the verdict is right for 53.8% of images (AUC 0.50). The batch evaluation is exact in simulation only: on quantum hardware every pair needs its own runs.",10.4,"middle",INK)
    f.save('docs/figures/phase4_batch_output.svg')


if __name__ == "__main__":
    module1(); module2(); module3()
