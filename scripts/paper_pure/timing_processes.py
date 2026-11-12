import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np, multiprocessing as mp
sys.path.insert(0,'code')
from PIL import Image
from cmfd_swaptest_pipeline import SwapTestSiameseQNN, extract_all_patches, generate_all_pairs, encode_offset
CK='checkpoints_pure/swaptest_qnn_epoch10.npz'
img=np.asarray(Image.open('data/comofod/CoMoFoD_small_v2/001_F.png').convert('L').resize((128,128),Image.LANCZOS),dtype=np.float32)/255.
C,ST=extract_all_patches(img,8,8); P=generate_all_pairs(C,min_spatial_dist=8)
Q=None
def init():
    global Q
    Q=SwapTestSiameseQNN(patch_size=8,n_layers=3,n_offset_qubits=0,img_size=128,seed=1); Q.load(CK)
def job(ch):
    o=np.array([encode_offset(C[j][0]-C[i][0],C[j][1]-C[i][1],128) for i,j in ch])
    return Q.similarity(ST[[i for i,j in ch]],ST[[j for i,j in ch]],o)
N=256
chunks=[P[k*N:(k+1)*N] for k in range(16)]
total=N*16
init(); job(chunks[0][:8])
t=time.perf_counter(); s1=np.concatenate([job(c) for c in chunks]); d1=time.perf_counter()-t
print(f'1 proc: {d1/total*1e3:.2f} ms/pair -> {d1/total*len(P)/60:.1f} min/image')
for w in (4,8):
    ctx=mp.get_context('fork')
    with ctx.Pool(w,initializer=init) as pool:
        list(pool.map(job,[chunks[0][:8]]*w))
        t=time.perf_counter(); r=pool.map(job,chunks); d=time.perf_counter()-t
    s=np.concatenate(r)
    print(f'{w} procs: {d/total*1e3:.2f} ms/pair -> {d/total*len(P)/60:.1f} min/image; speedup x{d1/d:.1f}; max|diff|={np.abs(s-s1).max():.1e}')
