import numpy as np, sys
from PIL import Image
D='data/comofod/CoMoFoD_small_v2/'
def lap(a):
    return 4*a-np.roll(a,1,0)-np.roll(a,-1,0)-np.roll(a,1,1)-np.roll(a,-1,1)
def peak(img,S,minshift):
    a=lap(img); a=a-a.mean()
    R=np.fft.ifft2(np.abs(np.fft.fft2(a))**2).real/max((a**2).sum(),1e-12)
    k=np.minimum(np.arange(S),S-np.arange(S))
    ok=k[:,None]**2+k[None,:]**2>=minshift**2
    v=R[ok]; med=np.median(v); mad=np.median(np.abs(v-med))+1e-12
    return float(v.max()),float((v.max()-med)/mad)
def auc(p,n):
    p,n=np.asarray(p),np.asarray(n); return float(np.mean([(x>n).mean()+.5*(x==n).mean() for x in p]))
ids=[f'{i:03d}' for i in range(1,201)]
G=[('translation',0,40),('rotation',40,80),('scaling',80,120),('distortion',120,160),('combination',160,200),('all 200',0,200)]
for S in (128,256,512):
    ms=S//8
    r={}
    for s in ids:
        l=lambda n:np.asarray(Image.open(f'{D}{s}_{n}.png').convert('L').resize((S,S),Image.LANCZOS),dtype=np.float64)/255.
        r[s]=(peak(l('F'),S,ms),peak(l('O'),S,ms))
    wires=int(2*np.log2(S))+1
    print(f'{S}px ({wires} wires)')
    for name,a,b in G:
        sub=ids[a:b]
        pf=[r[s][0][0] for s in sub]; po=[r[s][1][0] for s in sub]
        print(f'  {name:12s} AUC peak {auc(pf,po):.3f} | AUC z {auc([r[s][0][1] for s in sub],[r[s][1][1] for s in sub]):.3f} | median peak forged {np.median(pf):.4f} authentic {np.median(po):.4f}',flush=True)
