"""Visualization utilities for segmentation results."""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import matplotlib.patches as mpatches
import json
from matplotlib import cm

def plot_segments_pointseg(data_path: str, out_dir: str) -> None:
    """
    Visualize point-based segmentation results.

    Creates a grid visualization for each image:
    - Row 1: RGB images with point markers
    - Row 2: Ground truth segments
    - Row 3: Predicted segments

    Args:
        data_path: Path to H5 file with results
        out_dir: Directory to save visualizations
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, 'r') as f:
        for img_key in f.keys():
            img_group = f[img_key]

            # Load data
            rgb = img_group['rgb'][:]

            # Handle different naming conventions
            if 'segment' in img_group:
                gt_masks = img_group['segment'][:]
            else:
                gt_masks = img_group['segments_gt'][:]

            n_segments = gt_masks.shape[0]

            # Create figure
            fig, axes = plt.subplots(3, n_segments, figsize=(4 * n_segments, 12))

            # Handle single segment case
            if n_segments == 1:
                axes = axes.reshape(3, 1)

            for i in range(n_segments):
                # Row 1: RGB image with point
                axes[0, i].imshow(rgb)
                axes[0, i].set_title(f'Input Image {i}')
                axes[0, i].axis('off')

                # Get and plot point if available
                seg_key = f'seg{i}'
                if seg_key in img_group and 'pt0' in img_group[seg_key]:
                    x, y = img_group[seg_key]['pt0']['centroid'][:]
                    axes[0, i].scatter([x], [y], c='red', s=100, marker='x', linewidths=2)

                # Row 2: GT mask
                axes[1, i].imshow(gt_masks[i], cmap='gray')
                axes[1, i].set_title(f'GT Segment {i}')
                axes[1, i].axis('off')

                # Row 3: Predicted mask
                if seg_key in img_group and 'pt0' in img_group[seg_key]:
                    pred_mask = img_group[seg_key]['pt0']['segment'][:]
                    axes[2, i].imshow(pred_mask, cmap='gray')

                    # Add IoU score if available
                    iou = compute_iou(pred_mask, gt_masks[i])
                    axes[2, i].set_title(f'Predicted (IoU: {iou:.3f})')
                else:
                    axes[2, i].imshow(np.zeros_like(gt_masks[i]), cmap='gray')
                    axes[2, i].set_title('Predicted (Missing)')
                axes[2, i].axis('off')

            plt.suptitle(f'Point-based Segmentation: {img_key}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            out_file = out_dir / f'{img_key}.png'
            fig.savefig(out_file, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"Saved visualization to {out_file}")


def create_overlay(image: np.ndarray,
                   mask: np.ndarray,
                   alpha: float = 0.5,
                   color: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """
    Create an overlay of mask on image.

    Args:
        image: RGB image (H, W, 3) normalized to [0, 1]
        mask: Binary mask (H, W)
        alpha: Transparency for overlay
        color: RGB color for mask

    Returns:
        Overlayed image
    """
    # Ensure image is float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]

    # Blend
    overlay = image * (1 - alpha * mask[:, :, np.newaxis]) + colored_mask * alpha

    return np.clip(overlay, 0, 1)

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union

def evaluate_AP_AR_known_correspondence(pred_masks: np.ndarray,
                                        gt_masks: np.ndarray,
                                        iou_thresholds: np.ndarray = None) -> Dict:
    """
    Evaluate AP/AR when correspondence between pred and GT is known (point-based).

    Args:
        pred_masks: Predicted masks (N, H, W) - same order as GT
        gt_masks: Ground truth masks (N, H, W)
        iou_thresholds: IoU thresholds for evaluation

    Returns: metrics dictionary
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)

    n_masks = len(gt_masks)

    # Compute IoU for each corresponding pair
    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        ious.append(compute_iou(pred, gt))
    ious = np.array(ious)

    # Compute precision/recall at each threshold
    precisions = []
    recalls = []

    for thresh in iou_thresholds:
        tp = np.sum(ious >= thresh)
        precision = tp / n_masks if n_masks > 0 else 0
        recall = tp / n_masks if n_masks > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        'AP': np.mean(precisions),
        'AR': np.mean(recalls),
        'iou_mean': np.mean(ious),
        'ious': ious,
        'thresholds': iou_thresholds
    }


def get_baseline_metrics_pointseg(data_path: str, output_json_path: str) -> Dict:
    """
    Compute metrics for point-based segmentation results.

    Args:
        data_path: Path to H5 file with results
        output_json_path: Path to save JSON metrics

    Returns: metrics dictionary
    """
    all_results = {}

    with h5py.File(data_path, 'r') as f:
        for image_key in f.keys():
            img_grp = f[image_key]

            # Handle both old and new naming conventions
            if 'segment' in img_grp:
                gt_segments = img_grp['segment'][:]
            else:
                gt_segments = img_grp['segments_gt'][:]

            # Collect predicted segments
            pred_segments = []
            for i in range(gt_segments.shape[0]):
                seg_key = f'seg{i}'
                if seg_key in img_grp:
                    pred_segments.append(img_grp[seg_key]['pt0']['segment'][:])
                else:
                    # Empty prediction if missing
                    pred_segments.append(np.zeros_like(gt_segments[i]))

            pred_segments = np.stack(pred_segments, axis=0)

            # Evaluate with known correspondence
            result = evaluate_AP_AR_known_correspondence(pred_segments, gt_segments)

            # Convert to JSON-serializable format
            serializable_result = {
                'AP': float(result['AP']),
                'AR': float(result['AR']),
                'iou_mean': float(result['iou_mean']),
                'ious': result['ious'].tolist(),
                'thresholds': result['thresholds'].tolist()
            }

            all_results[image_key] = serializable_result

    # Save results
    with open(output_json_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=2)

    print(f"Saved evaluation results to {output_json_path}")

    # Print summary statistics
    avg_ap = np.mean([r['AP'] for r in all_results.values()])
    avg_ar = np.mean([r['AR'] for r in all_results.values()])
    avg_iou = np.mean([r['iou_mean'] for r in all_results.values()])

    print(f"Average AP: {avg_ap:.4f}")
    print(f"Average AR: {avg_ar:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

    return all_results