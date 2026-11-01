"""High-contrast, large-print version of the swap-test circuit for projector /
e-board presentation. Thick strokes, big bold fonts, saturated (non-pastel)
colors so it survives ambient room light and viewing from the back of a room.

Run:  python docs/figures/gen_circuit_presentation.py
"""

BG      = "#ffffff"
INK     = "#000000"      # wires, gate borders, default text — pure black, max contrast
MUTED   = "#333333"      # secondary captions — still dark, still readable at distance
TITLE   = "#c1121f"      # headline / red accents
AMBER   = "#ffb300"      # Register data (AmpEmbed) — warm, high-visibility
AMBER_TXT = "#000000"
PURPLE  = "#7b1fa2"      # Offset / geometry register
PURPLE_TXT= "#ffffff"
BLUE    = "#0d47a1"      # Swap-test / ancilla / measurement — the "quantum answer" path
BLUE_TXT  = "#ffffff"

W, H = 1600, 1180
o = []; A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'role="img" aria-label="High-contrast presentation circuit: two 6-qubit patch registers '
  'in amber, a 2-qubit offset register in purple, and the swap test measured on the ancilla '
  'in blue, for a single patch-pair similarity evaluation.">')
A(f'<rect width="100%" height="100%" fill="{BG}"/>')
A('<defs>'
  f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>')
A(f'<g font-family="Arial, Helvetica, sans-serif" fill="{INK}">')

def box(x,y,w,h,fill,stroke,rx=5,sw=3.2):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def t(x,y,s,size=16,anchor="middle",fill=INK,weight="bold"):
    A(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>')
def line(x1,y1,x2,y2,color=INK,sw=3.2,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')
def circ(x,y,r,fill,stroke=None,sw=3):
    st=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    A(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"{st}/>')

# ---- title ----
t(W/2,44,"Pure Swap-Test Siamese QNN &#8212; Patch-Pair Comparison Circuit",26,"middle",TITLE,"bold")
t(W/2,72,"15 wires &#183; 158 gates &#183; 58 trainable angles &#183; one evaluation scores ONE patch pair",15,"middle",MUTED,"normal")

# ---- wire layout ----
y0,dy,gap=140,44,26
yA=[y0+dy*i for i in range(6)]
yB=[yA[-1]+dy+gap+dy*i for i in range(6)]
yO=[yB[-1]+dy+gap, yB[-1]+2*dy+gap]
yAnc=yO[-1]+dy+gap
XS,XE=260,1460
for i,y in enumerate([*yA,*yB,*yO,yAnc]):
    is_anc = (y==yAnc)
    line(XS,y,XE,y, BLUE if is_anc else INK, 4.5 if is_anc else 3.2)
    t(XS-16,y+6,f"|0&#10217;",15,"end", BLUE if is_anc else INK)
    t(XS-52,y+6,f"q{i}",13,"end", MUTED, "normal")

def brace(y1,y2,lab,sub,col):
    x=150
    A(f'<path d="M {x+18} {y1-11} L {x+6} {y1-11} L {x+6} {(y1+y2)/2-7} L {x-2} {(y1+y2)/2} '
      f'L {x+6} {(y1+y2)/2+7} L {x+6} {y2+11} L {x+18} {y2+11}" fill="none" stroke="{col}" stroke-width="3.5"/>')
    t(x-10,(y1+y2)/2-4,lab,16,"end",col,"bold"); t(x-10,(y1+y2)/2+17,sub,13,"end",MUTED,"normal")
brace(yA[0],yA[-1],"REGISTER A","patch &#968;&#8320; (64 px)",AMBER.replace('#ffb300','#a16800'))
brace(yB[0],yB[-1],"REGISTER B","patch &#968;&#8321; (64 px)",AMBER.replace('#ffb300','#a16800'))
brace(yO[0],yO[-1],"OFFSET","&#916;=(dy,dx)",PURPLE)
t(96,yAnc+6,"ANCILLA",16,"end",BLUE,"bold")

# ---- amplitude embedding ----
for ys in (yA,yB):
    box(272,ys[0]-16,72,ys[-1]-ys[0]+32,AMBER,"#8a5c00",sw=3.5)
    t(308,(ys[0]+ys[-1])/2-6,"AMP",15,"middle",AMBER_TXT,"bold")
    t(308,(ys[0]+ys[-1])/2+14,"EMBED",15,"middle",AMBER_TXT,"bold")

# ---- rotation layer ----
GX={"RX":392,"RY":448,"RZ":504}
for ys in (yA,yB):
    for y in ys:
        for g,x in GX.items():
            box(x,y-15,48,30,"#ffffff",INK,rx=4,sw=3)
            t(x+24,y+6,g,14,"middle",INK,"bold")

def cnot(xc,yc,yt):
    line(xc,yc,xc,yt,INK,3.2)
    circ(xc,yc,5.5,INK)
    circ(xc,yt,11,"#ffffff",INK,3.2)
    line(xc-11,yt,xc+11,yt,INK,3.2); line(xc,yt-11,xc,yt+11,INK,3.2)
for ys in (yA,yB):
    for k in range(6): cnot(576+k*30, ys[k], ys[(k+1)%6])

A(f'<path d="M 386 {yA[0]-30} L 386 {yA[0]-38} L 750 {yA[0]-38} L 750 {yA[0]-30}" fill="none" stroke="{MUTED}" stroke-width="2.5"/>')
t(568,yA[0]-48,"DATA RE-UPLOADING LAYER  &#215;3   (&#952; = w + pixel &#183; &#960;)   +   CNOT RING",16,"middle",INK,"bold")

# shared-theta tie
line(568,yA[-1]+18,568,yB[0]-18,BLUE,4,"9 5")
circ(568,(yA[-1]+yB[0])/2,6,BLUE)
t(592,(yA[-1]+yB[0])/2+6,"SHARED &#952; &#8212; ONE weight tensor drives BOTH registers (Siamese)",16,"start",BLUE,"bold")

# ---- offset encoding ----
for y in yO:
    box(392,y-16,164,32,PURPLE,"#4a0e5c",sw=3.2)
    t(474,y+6,"RY(&#966;+&#916;&#183;&#960;)",14,"middle",PURPLE_TXT,"bold")

# ---- swap test ----
box(824,yAnc-19,42,38,"#ffffff",BLUE,rx=5,sw=4.5)
t(845,yAnc+8,"H",20,"middle",BLUE,"bold")

for k in range(6):
    x=910+k*34
    line(x,yA[k],x,yAnc,BLUE,3.6)
    circ(x,yAnc,6.5,BLUE)
    for yy in (yA[k],yB[k]):
        line(x-9,yy-9,x+9,yy+9,BLUE,3.4); line(x+9,yy-9,x-9,yy+9,BLUE,3.4)
t(1080,yA[0]-38,"CSWAP &#215; 6",18,"middle",BLUE,"bold")
t(1080,yA[0]-16,"ancilla-controlled swap of the WHOLE register",13.5,"middle",MUTED,"normal")

for k,y in enumerate(yO):
    x=1180+k*58
    line(x,y,x,yAnc-19,INK,3.2); circ(x,y,6,INK)
    box(x-22,yAnc-19,44,38,PURPLE,"#4a0e5c",rx=5,sw=3.2); t(x,yAnc+8,"RY",14,"middle",PURPLE_TXT,"bold")
t(1210,yO[0]-34,"CRY &#215; 2",17,"middle",PURPLE,"bold")
t(1210,yO[0]-14,"offset modulates the interference fringe",13,"middle",MUTED,"normal")

box(1318,yAnc-19,42,38,"#ffffff",BLUE,rx=5,sw=4.5)
t(1339,yAnc+8,"H",20,"middle",BLUE,"bold")

box(1384,yAnc-21,60,42,BLUE,"#08316b",rx=5,sw=3.5)
A(f'<path d="M 1397 {yAnc+11} A 17 17 0 0 1 1431 {yAnc+11}" fill="none" stroke="{BLUE_TXT}" stroke-width="3"/>')
line(1414,yAnc+11,1431,yAnc-11,BLUE_TXT,3)
t(1414,yAnc-34,"&#10216;Z&#10217; &#8712; [0,1]",16,"middle",BLUE,"bold")

A(f'<path d="M 816 {yAnc+52} L 816 {yAnc+62} L 1444 {yAnc+62} L 1444 {yAnc+52}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>')
t(1130,yAnc+82,"SWAP TEST",17,"middle",BLUE,"bold")

# ---- bottom explanation strip ----
BY = yAnc + 120
box(60,BY,1480,170,"#f5f5f5","#000000",rx=6,sw=3)
t(800,BY+34,"&#10216;Z&#10217; = |&#10216;&#968;&#8320;|&#968;&#8321;&#10217;|&#178;  &#8212;  the measured overlap of the two patch states  =  their similarity score",19,"middle",INK,"bold")
t(800,BY+68,"IDENTICAL patches &#8594; swapping changes nothing &#8594; perfect interference &#8594; &#10216;Z&#10217; = 1",16,"middle",BLUE,"bold")
t(800,BY+96,"ORTHOGONAL patches &#8594; no interference &#8594; &#10216;Z&#10217; = 0",16,"middle",TITLE,"bold")
t(800,BY+128,"Every trainable parameter above is a circuit rotation angle &#8212; no classical layer anywhere in the model.",14.5,"middle",MUTED,"normal")

A('</g></svg>')
open('docs/figures/circuit_presentation.svg','w').write("\n".join(o))
print("wrote docs/figures/circuit_presentation.svg")
