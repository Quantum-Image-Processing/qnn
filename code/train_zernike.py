"""
Train the 9-wire Zernike swap-test QNN and check that it actually learned.
==========================================================================
Only the 36 circuit rotation angles are trained, on self-supervised synthetic
pairs from authentic training images. Scenes are split three ways: training,
validation (a fixed set of pairs, used to pick the best weights and to check
learning), and test (written to split.json, never used here).

Example:
    cd code
    OMP_NUM_THREADS=1 python train_zernike.py --dataset-dir ../data/comofod \
        --epochs 10 --images-per-epoch 80 --out ../models/zernike_9w.npz

The learning rate starts at --lr and follows a cosine curve down to
--lr * --lr-final-frac. Trials at 0.3 did not learn and 0.01 was too slow.
OMP_NUM_THREADS=1 was faster than all threads at this circuit size.

Learning here means ranking copied pairs above non-copied pairs. It is not the
same as detecting forgeries in an image, which evaluate_zernike.py measures;
see the note at the top of cmfd_zernike_swaptest.py.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

import cmfd_swaptest_pipeline as v3
from cmfd_zernike_swaptest import ZernikeSwapQNN, zernike_state
from train_split import scene_id

# The bar for saying training learned something. A project choice, not a standard.
MIN_AUC_GAIN = 0.03


def pair_arrays(batch):
    return (np.array([b[0] for b in batch]), np.array([b[1] for b in batch]),
            np.array([b[3] for b in batch]))


def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def validate(qnn, a, b, labels):
    s = np.asarray(qnn.score(a, b), dtype=float)
    return dict(loss=float(np.mean((s - labels) ** 2)), auc=auc(s, labels),
                copied=float(s[labels == 1].mean()), other=float(s[labels == 0].mean()))


def main():
    ap = argparse.ArgumentParser(description="Trainer for the 9-wire Zernike swap-test QNN")
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--val-scenes', type=int, default=20)
    ap.add_argument('--val-pairs-per-image', type=int, default=20)
    ap.add_argument('--n-layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--images-per-epoch', type=int, default=80)
    ap.add_argument('--pairs-per-image', type=int, default=12)
    ap.add_argument('--img-size', type=int, default=128,
                    help='Same resolution the detector scores at.')
    ap.add_argument('--no-rotations', action='store_true',
                    help='Leave out rotated positives. They are on by default '
                         'because Zernike magnitudes can represent them.')
    ap.add_argument('--lr', type=float, default=0.1, help='Starting learning rate.')
    ap.add_argument('--lr-final-frac', type=float, default=0.1,
                    help='Final learning rate as a fraction of --lr. 1 keeps it constant.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='../models/zernike_9w.npz', help='Best weights by validation AUC.')
    ap.add_argument('--checkpoint-dir', default='../models/zernike_checkpoints')
    args = ap.parse_args()

    ds = v3.CoMoFoDDataset(args.dataset_dir, img_size=args.img_size, grayscale=True)
    # v3's pair generator encodes patches through this one function, so replacing
    # it in this process is all it needs. Nothing on disk changes.
    v3.prepare_patch_state = zernike_state

    # ---- split by scene id so a scene is never in two sets ----
    ids = sorted({scene_id(p) for p in ds.original_files}, key=int)
    np.random.default_rng(args.seed).shuffle(ids)
    n_test = max(1, int(round(len(ids) * args.test_frac)))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + args.val_scenes])
    train_ids = set(ids[n_test + args.val_scenes:])
    print(f"[SPLIT] {len(train_ids)} train / {len(val_ids)} validation / {len(test_ids)} test scenes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(out.parent / 'split.json', 'w') as f:
        json.dump({'train_ids': sorted(train_ids, key=int), 'val_ids': sorted(val_ids, key=int),
                   'test_ids': sorted(test_ids, key=int), 'seed': args.seed,
                   'test_frac': args.test_frac}, f, indent=2)

    rotate = not args.no_rotations
    files = list(ds.original_files)

    # ---- fixed validation pairs, so every epoch is measured on the same data ----
    ds.original_files = [p for p in files if scene_id(p) in val_ids]
    val_rng = np.random.default_rng(args.seed + 999)
    n = args.val_pairs_per_image
    val_batch = []
    for i in range(len(ds.original_files)):
        val_batch += v3.make_synthetic_pairs(ds.get_training_image(i), 8, n_pos=n // 2, n_neg=n - n // 2,
                                             rng=val_rng, rotate=rotate)
    va, vb, vl = pair_arrays(val_batch)
    ds.original_files = [p for p in files if scene_id(p) in train_ids]

    qnn = ZernikeSwapQNN(n_layers=args.n_layers, img_size=args.img_size, seed=args.seed)
    print(f"[INFO] training {qnn.n_trainable()} rotation angles, {qnn.n_wires} wires, "
          f"device={qnn.dev.name}, diff_method={qnn.diff_method}")

    plain = np.sum(va * vb, axis=1) ** 2
    plain_auc = auc(plain, vl)
    start = validate(qnn, va, vb, vl)
    print(f"[VAL] plain swap test, no layers: AUC {plain_auc:.3f}")
    print(f"[VAL] before training: loss {start['loss']:.4f}  AUC {start['auc']:.3f}  "
          f"copied {start['copied']:.3f}  other {start['other']:.3f}")

    # ---- training ----
    opt = qml.AdamOptimizer(stepsize=args.lr)
    rng = np.random.default_rng(args.seed)
    total = args.epochs * args.images_per_epoch
    step = 0
    best = dict(start)
    best['epoch'] = 0
    qnn.save(out)
    history = [dict(epoch=0, **start)]
    for epoch in range(1, args.epochs + 1):
        losses = []
        for img_idx in rng.integers(0, len(ds.original_files), size=args.images_per_epoch):
            opt.stepsize = args.lr * (args.lr_final_frac + (1 - args.lr_final_frac)
                                      * 0.5 * (1 + math.cos(math.pi * step / total)))
            n_pos = args.pairs_per_image // 2
            batch = v3.make_synthetic_pairs(ds.get_training_image(int(img_idx)), 8, n_pos=n_pos,
                                            n_neg=args.pairs_per_image - n_pos, rng=rng, rotate=rotate)
            a, b, labels = pair_arrays(batch)
            a = pnp.array(a, requires_grad=False)
            b = pnp.array(b, requires_grad=False)
            labels = pnp.array(labels, requires_grad=False)
            offs = pnp.zeros((len(batch), 2), requires_grad=False)
            qnn.weights, loss = opt.step_and_cost(
                lambda w: pnp.mean((qnn.qnode(a, b, offs, w, qnn.offset_weights) - labels) ** 2),
                qnn.weights)
            losses.append(float(loss))
            step += 1

        val = validate(qnn, va, vb, vl)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=float(opt.stepsize), **val))
        print(f"[Epoch {epoch}/{args.epochs}] lr {opt.stepsize:.4f}  train loss {np.mean(losses):.4f}  "
              f"val loss {val['loss']:.4f}  val AUC {val['auc']:.3f}  "
              f"copied {val['copied']:.3f}  other {val['other']:.3f}", flush=True)
        qnn.save(ckpt_dir / f"zernike_epoch{epoch}.npz")
        if val['auc'] > best['auc']:
            best = dict(val, epoch=epoch)
            qnn.save(out)

    with open(ckpt_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=1)

    # ---- did it learn? computed from the validation numbers above ----
    gain = best['auc'] - start['auc']
    learned = gain >= MIN_AUC_GAIN and best['loss'] < start['loss']
    print(f"\n[LEARNING CHECK] best epoch {best['epoch']}: AUC {start['auc']:.3f} -> {best['auc']:.3f}, "
          f"loss {start['loss']:.4f} -> {best['loss']:.4f}")
    print(f"[LEARNING CHECK] gap copied minus other: {start['copied'] - start['other']:.3f} -> "
          f"{best['copied'] - best['other']:.3f}; plain swap test AUC {plain_auc:.3f}")
    print("[LEARNING CHECK] " + ("LEARNED on validation pairs. This does not yet show better detection."
                                  if learned else
                                  "did NOT learn: try another --lr or more --epochs before using these weights."))
    print(f"[DONE] best weights -> {out}")
    print(f"       measure detection: python evaluate_zernike.py --dataset-dir {args.dataset_dir} --checkpoint {out}")


if __name__ == '__main__':
    main()
