"""
CMFD Testing Pipeline - CoMoFoD Dataset
- Real copy-move forgeries with transformations (rotation, scaling, JPEG, noise, blur)
- PennyLane QNN with amplitude encoding
- Sliding window + heatmap generation
- Per-transformation analysis

Usage:
    python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod --num-images 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# PennyLane
import pennylane as qml
from pennylane import numpy as pnp

# Image processing
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import threshold_otsu

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# DATASET LOADING
# ============================================================================

class CoMoFoDDataset:
    """Load CoMoFoD dataset with ground truth masks."""
    
    def __init__(self, dataset_dir: str, img_size: int = 256, grayscale: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.grayscale = grayscale
        
        self.forged_dir = self.dataset_dir / 'Forged'
        self.masks_dir = self.dataset_dir / 'Masks'
        
        if not self.forged_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Could not find CoMoFoD dataset at {self.dataset_dir}\n"
                "Expected: dataset_dir/Forged/ and dataset_dir/Masks/"
            )
        
        # Get all forged images
        self.image_files = sorted(list(self.forged_dir.glob('*.jpg')) + 
                                  list(self.forged_dir.glob('*.png')))
        print(f"[INFO] Found {len(self.image_files)} images in {self.forged_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load image
        img = Image.open(img_path)
        if self.grayscale:
            img = img.convert('L')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Load mask
        mask_path = self.masks_dir / f"{img_name}.png"
        if not mask_path.exists():
            mask_path = self.masks_dir / f"{img_name}.jpg"
        
        if not mask_path.exists():
            return None, None, img_name
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = (np.array(mask, dtype=np.uint8) > 128).astype(np.uint8)
        
        return img_array, mask_array, img_name
    
    @staticmethod
    def get_transformation_type(filename: str) -> str:
        """Extract transformation type from CoMoFoD filename."""
        # Filename format: original_X_Y_Z_T where T is transformation
        parts = filename.split('_')
        if len(parts) >= 4:
            transform = parts[-1]
            transform_map = {
                'B': 'Blur',
                'G': 'Gaussian Noise',
                'JC': 'JPEG Compression',
                'N': 'Noise',
                'R': 'Rotation',
                'S': 'Scaling',
                'O': 'Original'
            }
            return transform_map.get(transform, 'Unknown')
        return 'Unknown'


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_patch_features(patch: np.ndarray, target_dim: int = 24) -> np.ndarray:
    """Extract features from patch for QNN."""
    # DCT features
    dct2 = dct(dct(patch, axis=0), axis=1)
    dct_flat = dct2.flatten()
    
    if len(dct_flat) >= 12:
        idx_sorted = np.argsort(np.abs(dct_flat))[-12:]
        dct_feat = dct_flat[idx_sorted]
    else:
        dct_feat = dct_flat
        dct_feat = np.pad(dct_feat, (0, 12 - len(dct_feat)), mode='constant')
    
    # Histogram features
    hist, _ = np.histogram(patch, bins=12, range=(0, 1))
    hist_feat = hist.astype(np.float32)
    
    features = np.concatenate([dct_feat, hist_feat])
    
    if len(features) < target_dim:
        features = np.pad(features, (0, target_dim - len(features)), mode='constant')
    elif len(features) > target_dim:
        features = features[:target_dim]
    
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features = features / norm
    
    return features


# ============================================================================
# QUANTUM NEURAL NETWORK
# ============================================================================

class QuantumCMFDClassifier:
    """PennyLane QNN for CMFD."""
    
    def __init__(self, n_qubits: int = 5, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.weights = pnp.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3), requires_grad=True)
        self.qnn = qml.QNode(self._circuit, self.dev, diff_method='parameter-shift')
    
    def _circuit(self, features, weights):
        """Quantum circuit."""
        dim = 2 ** self.n_qubits
        if len(features) < dim:
            features = np.pad(features, (0, dim - len(features)), mode='constant')
        else:
            features = features[:dim]
        
        norm = np.linalg.norm(features)
        if norm > 1e-6:
            features = features / norm
        
        qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), 
                               normalize=False, pad_with=0)
        
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            for qubit in range(self.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])
        
        return qml.expval(qml.PauliZ(0))
    
    def predict(self, features):
        """Get forgery probability."""
        dim = 2 ** self.n_qubits
        if len(features) < dim:
            features = np.pad(features, (0, dim - len(features)), mode='constant')
        else:
            features = features[:dim]
        
        expval = self.qnn(features, self.weights)
        prob = float((expval + 1.0) / 2.0)
        return prob


# ============================================================================
# SLIDING WINDOW + HEATMAP
# ============================================================================

def compute_heatmap(image: np.ndarray, qnn: QuantumCMFDClassifier,
                   patch_size: int = 32, stride: int = 4, verbose: bool = True) -> np.ndarray:
    """Compute forgery heatmap via sliding window."""
    H, W = image.shape
    
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.uint32)
    
    total_patches = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
    pbar = tqdm(total=total_patches, desc="Sliding window", disable=not verbose) if verbose else None
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            features = extract_patch_features(patch, target_dim=2**qnn.n_qubits)
            score = qnn.predict(features)
            
            heatmap[y:y + patch_size, x:x + patch_size] += score
            counts[y:y + patch_size, x:x + patch_size] += 1
            
            if pbar:
                pbar.update(1)
    
    if pbar:
        pbar.close()
    
    mask = counts > 0
    heatmap[mask] /= counts[mask]
    
    return heatmap


def threshold_and_clean(heatmap: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """Convert heatmap to binary mask."""
    if method == 'otsu':
        threshold = threshold_otsu(heatmap)
    else:
        threshold = 0.5
    
    mask = heatmap > threshold
    
    kernel = np.ones((3, 3), dtype=bool)
    mask = binary_erosion(mask, structure=kernel, iterations=1)
    mask = binary_dilation(mask, structure=kernel, iterations=1)
    
    mask = binary_dilation(mask, structure=kernel, iterations=1)
    mask = binary_erosion(mask, structure=kernel, iterations=1)
    
    return mask.astype(np.uint8)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_result(image: np.ndarray, heatmap: np.ndarray, 
                    pred_mask: np.ndarray, gt_mask: np.ndarray,
                    metrics: dict, img_name: str = '', save_path: str = None):
    """Visualize results."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Original Image\n{img_name}')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('QNN Heatmap\n(Red=high forgery confidence)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_mask, cmap='gray')
    ax3.set_title('Ground Truth Mask')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(pred_mask, cmap='gray')
    ax4.set_title('Predicted Mask')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(image, cmap='gray')
    ax5.imshow(pred_mask, cmap='Reds', alpha=0.5)
    ax5.set_title('Predicted Mask Overlay')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    metrics_text = f"""
    Metrics:
    
    Precision: {metrics['precision']:.4f}
    Recall:    {metrics['recall']:.4f}
    F1 Score:  {metrics['f1']:.4f}
    IoU:       {metrics['iou']:.4f}
    Accuracy:  {metrics['accuracy']:.4f}
    
    TP={metrics['tp']}  FP={metrics['fp']}
    FN={metrics['fn']}  TN={metrics['tn']}
    """
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[INFO] Saved to {save_path}")
    
    plt.close(fig)
    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CMFD - CoMoFoD Dataset')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to CoMoFoD dataset')
    parser.add_argument('--patch-size', type=int, default=32,
                       help='Patch size (default: 32)')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride (default: 4)')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to test')
    parser.add_argument('--n-qubits', type=int, default=5,
                       help='Number of qubits')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Variational layers')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Start index')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("CMFD Testing Pipeline - CoMoFoD Dataset")
    print("PennyLane QNN + Sliding Window + Real Transformations")
    print("="*80)
    
    # Load dataset
    print(f"\n[1/6] Loading CoMoFoD dataset from {args.dataset_dir}...")
    try:
        dataset = CoMoFoDDataset(args.dataset_dir, img_size=256, grayscale=True)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Initialize QNN
    print(f"\n[2/6] Initializing QNN ({args.n_qubits} qubits, {args.n_layers} layers)...")
    qnn = QuantumCMFDClassifier(n_qubits=args.n_qubits, n_layers=args.n_layers)
    print(f"[INFO] QNN initialized with {np.prod(qnn.weights.shape)} parameters")
    
    # Process images
    num_test = min(args.num_images, len(dataset) - args.start_idx)
    print(f"\n[3/6] Processing {num_test} test images...")
    
    results = []
    transform_stats = defaultdict(lambda: {'f1': [], 'iou': [], 'accuracy': []})
    valid_count = 0
    
    for idx in range(args.start_idx, args.start_idx + num_test):
        if idx >= len(dataset):
            break
        
        print(f"\n--- Image {valid_count+1}/{num_test} ---")
        
        img, gt_mask, img_name = dataset[idx]
        
        if img is None:
            print(f"  [SKIP] Could not load {img_name}")
            continue
        
        valid_count += 1
        
        # Get transformation type
        transform_type = CoMoFoDDataset.get_transformation_type(img_name)
        print(f"  Transformation: {transform_type}")
        
        # Compute heatmap
        print("  Computing heatmap...")
        heatmap = compute_heatmap(img, qnn, 
                                 patch_size=args.patch_size,
                                 stride=args.stride, 
                                 verbose=True)
        
        # Threshold & clean
        print("  Thresholding and cleanup...")
        pred_mask = threshold_and_clean(heatmap, method='otsu')
        
        # Metrics
        metrics = compute_metrics(pred_mask, gt_mask)
        
        print(f"  F1={metrics['f1']:.4f}, IoU={metrics['iou']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Visualize
        viz_path = Path(args.output_dir) / f"{img_name}_result.png"
        visualize_result(img, heatmap, pred_mask, gt_mask, 
                        metrics, img_name=img_name, save_path=str(viz_path))
        
        # Store results
        results.append({
            'image_name': img_name,
            'transformation': transform_type,
            **metrics
        })
        
        # Track by transformation
        transform_stats[transform_type]['f1'].append(metrics['f1'])
        transform_stats[transform_type]['iou'].append(metrics['iou'])
        transform_stats[transform_type]['accuracy'].append(metrics['accuracy'])
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        results_df = pd.DataFrame(results)
        print(f"\nOverall metrics ({len(results)} images):")
        print(results_df[['precision', 'recall', 'f1', 'iou', 'accuracy']].describe())
        
        print(f"\n\nTransformation-Based Analysis:")
        for transform_type in sorted(transform_stats.keys()):
            stats = transform_stats[transform_type]
            if stats['f1']:
                mean_f1 = np.mean(stats['f1'])
                std_f1 = np.std(stats['f1'])
                mean_iou = np.mean(stats['iou'])
                count = len(stats['f1'])
                print(f"\n{transform_type}:")
                print(f"  F1:  {mean_f1:.4f} ± {std_f1:.4f}")
                print(f"  IoU: {mean_iou:.4f}")
                print(f"  Count: {count}")
        
        # Save results
        csv_path = Path(args.output_dir) / 'results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Results saved to {csv_path}")
        print(f"[INFO] Visualizations in {args.output_dir}/")
        
        print("\n✓ Testing complete!")
    else:
        print("\n[ERROR] No valid images processed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
