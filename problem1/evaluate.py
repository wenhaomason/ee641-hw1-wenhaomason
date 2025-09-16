import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import ShapeDetectionDataset, collate_fn, get_transform
from model import create_model
from PIL import Image
from torch.utils.data import DataLoader
from utils import compute_iou, decode_boxes, generate_anchors, get_anchor_configuration


def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        ap: Average precision value
    """
    # Build a flat list of predictions with image indices
    pred_boxes_list = []  # list of tensors [4]
    pred_scores_list = []  # list of floats
    pred_img_idx = []  # list of ints
    for img_idx, pred in enumerate(predictions):
        if len(pred.get('boxes', [])) > 0:
            for b, s in zip(pred['boxes'], pred['scores']):
                pred_boxes_list.append(b)
                pred_scores_list.append(s)
                pred_img_idx.append(img_idx)

    if len(pred_boxes_list) == 0:
        return 0.0

    all_pred_boxes = torch.stack(pred_boxes_list, dim=0)
    all_pred_scores = torch.stack(pred_scores_list, dim=0)

    # Sort by confidence scores (desc)
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    sorted_list = sorted_indices.tolist()
    pred_img_idx = [pred_img_idx[i] for i in sorted_list]

    # Count total ground truth boxes
    total_gt = sum(len(gt.get('boxes', [])) for gt in ground_truths)
    if total_gt == 0:
        return 0.0

    # Track detections
    tp = torch.zeros(len(all_pred_boxes))
    fp = torch.zeros(len(all_pred_boxes))

    # For each image, track which ground truths have been matched
    gt_matched = [torch.zeros(len(gt.get('boxes', [])), dtype=torch.bool) for gt in ground_truths]

    # Process each prediction only against gts from the same image
    for pred_idx, (pred_box, img_idx) in enumerate(zip(all_pred_boxes, pred_img_idx)):
        gt = ground_truths[img_idx]
        if len(gt.get('boxes', [])) == 0:
            fp[pred_idx] = 1
            continue

        # Ensure both tensors are on the same device
        gt_boxes = gt['boxes']
        if pred_box.is_cuda and not gt_boxes.is_cuda:
            gt_boxes = gt_boxes.cuda()
        elif not pred_box.is_cuda and gt_boxes.is_cuda:
            pred_box = pred_box.cuda()
            
        ious = compute_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, max_idx = torch.max(ious, dim=0)

        if max_iou >= iou_threshold:
            if not gt_matched[img_idx][max_idx]:
                tp[pred_idx] = 1
                gt_matched[img_idx][max_idx] = True
            else:
                fp[pred_idx] = 1  # duplicate detection for same GT
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in torch.arange(0, 1.1, 0.1, device=recalls.device):
        mask = recalls >= t
        if mask.sum() == 0:
            p = 0.0
        else:
            p = float(precisions[mask].max().item())
        ap += p / 11.0
    
    return ap

def non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: Tensor of shape [N, 4]
        scores: Tensor of shape [N]
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold to filter low-confidence detections
        
    Returns:
        keep_indices: Indices of boxes to keep
    """
    # Filter by score threshold
    keep_mask = scores > score_threshold
    if not keep_mask.any():
        return torch.tensor([], dtype=torch.long)
    
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    keep_indices_orig = torch.where(keep_mask)[0]
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = compute_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU < threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    keep = torch.tensor(keep, dtype=torch.long)
    return keep_indices_orig[keep]

def decode_predictions_for_eval(predictions, anchors, score_threshold=0.1):
    """
    Decode model predictions into bounding boxes and scores for evaluation.
    
    Args:
        predictions: List of prediction tensors from model
        anchors: List of anchor tensors
        score_threshold: Minimum score threshold
        
    Returns:
        List of dicts with 'boxes', 'scores', 'labels' for each image
    """
    batch_size = predictions[0].shape[0]
    results = []
    
    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for scale_idx, (pred, anchor) in enumerate(zip(predictions, anchors)):
            B, C, H, W = pred.shape
            num_anchors_per_location = 3
            
            # Reshape predictions
            pred_batch = pred[batch_idx]  # [C, H, W]
            pred_batch = pred_batch.permute(1, 2, 0).contiguous()  # [H, W, C]
            pred_batch = pred_batch.view(H, W, num_anchors_per_location, -1)  # [H, W, 3, 8]
            pred_batch = pred_batch.view(-1, pred_batch.shape[-1])  # [H*W*3, 8]
            
            # Split predictions
            bbox_deltas = pred_batch[:, :4]
            obj_scores = torch.sigmoid(pred_batch[:, 4])
            cls_scores = F.softmax(pred_batch[:, 5:], dim=1)
            
            # Decode boxes
            anchor_flat = anchor.view(-1, 4)
            decoded_boxes = decode_boxes(bbox_deltas, anchor_flat)
            # Clamp to image boundaries (anchors were generated for 224x224)
            decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(min=0, max=224)
            decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(min=0, max=224)
            
            # Combine objectness and class scores
            max_cls_scores, predicted_labels = torch.max(cls_scores, dim=1)
            final_scores = obj_scores * max_cls_scores

            # Class-wise NMS and thresholding to avoid cross-class suppression
            for class_id in range(cls_scores.shape[1]):
                class_mask = (predicted_labels == class_id) & (final_scores > score_threshold)
                if class_mask.any():
                    boxes_c = decoded_boxes[class_mask]
                    scores_c = final_scores[class_mask]
                    keep_c = non_max_suppression(boxes_c, scores_c)
                    if keep_c.numel() > 0:
                        all_boxes.append(boxes_c[keep_c])
                        all_scores.append(scores_c[keep_c])
                        all_labels.append(torch.full((keep_c.numel(),), class_id, dtype=torch.long, device=pred.device))

        # Combine all classes and scales for this image
        if len(all_boxes) > 0:
            results.append({
                'boxes': torch.cat(all_boxes, dim=0),
                'scores': torch.cat(all_scores, dim=0),
                'labels': torch.cat(all_labels, dim=0)
            })
        else:
            results.append({
                'boxes': torch.zeros((0, 4), device=predictions[0].device),
                'scores': torch.zeros((0,), device=predictions[0].device),
                'labels': torch.zeros((0,), dtype=torch.long, device=predictions[0].device)
            })
    
    return results

def visualize_detections(image, predictions, ground_truths, save_path):
    """
    Visualize predictions and ground truth boxes.
    
    Args:
        image: PIL Image or numpy array
        predictions: Dict with 'boxes', 'scores', 'labels'
        ground_truths: Dict with 'boxes', 'labels'
        save_path: Path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Color map for classes
    colors = ['red', 'green', 'blue']
    class_names = ['circle', 'square', 'triangle']
    
    # Plot predictions
    ax1.imshow(image)
    ax1.set_title('Predictions')
    ax1.axis('off')
    
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if len(box) == 4:
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors[label % 3], facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Add label and score
            ax1.text(x1, y1-5, f'{class_names[label % 3]}: {score:.2f}',
                    color=colors[label % 3], fontsize=8, weight='bold')
    
    # Plot ground truth
    ax2.imshow(image)
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    for box, label in zip(ground_truths['boxes'], ground_truths['labels']):
        if len(box) == 4:
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors[label % 3], facecolor='none'
            )
            ax2.add_patch(rect)
            
            # Add label
            ax2.text(x1, y1-5, class_names[label % 3],
                    color=colors[label % 3], fontsize=8, weight='bold')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def analyze_scale_performance(model, dataloader, anchors, device):
    """
    Analyze which scales detect which object sizes.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        anchors: List of anchor tensors
        device: Device to run inference on
        
    Returns:
        Dictionary with scale performance analysis
    """
    model.eval()
    
    scale_detections = {
        'scale1': {'small': 0, 'medium': 0, 'large': 0},
        'scale2': {'small': 0, 'medium': 0, 'large': 0},
        'scale3': {'small': 0, 'medium': 0, 'large': 0}
    }
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Process each image
            for batch_idx in range(images.shape[0]):
                target = targets[batch_idx]
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                # Categorize ground truth boxes by size
                for box, label in zip(gt_boxes, gt_labels):
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    
                    if box_area < 1000:  # Small objects (circles)
                        size_category = 'small'
                    elif box_area < 3000:  # Medium objects (squares)
                        size_category = 'medium'
                    else:  # Large objects (triangles)
                        size_category = 'large'
                    
                    # Check which scale would detect this best
                    # (simplified - in practice would need more sophisticated analysis)
                    if size_category == 'small':
                        scale_detections['scale1'][size_category] += 1
                    elif size_category == 'medium':
                        scale_detections['scale2'][size_category] += 1
                    else:
                        scale_detections['scale3'][size_category] += 1
    
    return scale_detections

def visualize_anchor_coverage(sample_image, anchors, save_dir, image_size=224):
    """
    Create anchor coverage visualizations for each scale.

    Args:
        sample_image: tensor [3,H,W] or PIL
        anchors: list of tensors per scale [N,4]
        save_dir: directory to save plots
        image_size: input image size (assumed square)
    """
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(sample_image, torch.Tensor):
        img_np = sample_image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.array(sample_image)

    colors = ['cyan', 'magenta', 'yellow']
    for si, anchor in enumerate(anchors):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np)
        ax.set_title(f'Anchor coverage - Scale {si+1}')
        ax.axis('off')
        # plot a random subset of anchors to avoid clutter
        step = max(1, anchor.shape[0] // 500)
        a = anchor[::step]
        for b in a:
            x1, y1, x2, y2 = b.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5,
                                     edgecolor=colors[si % len(colors)], facecolor='none', alpha=0.5)
            ax.add_patch(rect)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'anchors_scale_{si+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def plot_scale_specialization(scale_counts, save_path):
    """Plot bar charts of which scales cover small/medium/large objects."""
    cats = ['small', 'medium', 'large']
    scales = ['scale1', 'scale2', 'scale3']
    values = [[scale_counts[s][c] for c in cats] for s in scales]
    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7,4))
    for i, s in enumerate(scales):
        ax.bar(x + (i-1)*width, values[i], width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel('Count')
    ax.set_title('Scale specialization by object size')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load validation dataset
    val_image_dir = '../datasets/detection/val'
    val_ann_file = '../datasets/detection/val_annotations.json'
    
    transform = get_transform()
    val_dataset = ShapeDetectionDataset(val_image_dir, val_ann_file, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Load trained model
    model = create_model(num_classes=3)
    checkpoint = torch.load('results/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate anchors
    feature_map_sizes, anchor_scales = get_anchor_configuration()
    anchors = generate_anchors(feature_map_sizes, anchor_scales)
    anchors = [anchor.to(device) for anchor in anchors]
    
    print('Running evaluation...')
    
    # Ensure directories exist
    os.makedirs('results/visualizations', exist_ok=True)

    # Collect predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            decoded_preds = decode_predictions_for_eval(outputs, anchors)
            
            # Store results - move ground truth to same device as predictions
            all_predictions.extend(decoded_preds)
            
            # Move ground truth tensors to device to match predictions
            targets_on_device = []
            for target in targets:
                target_on_device = {
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                }
                targets_on_device.append(target_on_device)
            all_ground_truths.extend(targets_on_device)
            
            # Visualize first 10 images
            if batch_idx < 2:  # First 2 batches = 16 images, we'll take first 10
                for img_idx in range(min(images.shape[0], 10 - batch_idx * 8)):
                    if batch_idx * 8 + img_idx >= 10:
                        break
                        
                    img = images[img_idx]
                    pred = decoded_preds[img_idx]
                    gt = targets_on_device[img_idx]
                    
                    save_path = f'results/visualizations/detection_result_{batch_idx * 8 + img_idx + 1:02d}.png'
                    visualize_detections(img, pred, gt, save_path)

            # Generate anchor coverage plots on first batch only (first image)
            if batch_idx == 0:
                visualize_anchor_coverage(images[0].cpu(), anchors, 'results/visualizations')
            
            if batch_idx % 10 == 0:
                print(f'Processed batch {batch_idx}/{len(val_loader)}')
    
    # Compute Average Precision for each class
    class_names = ['circle', 'square', 'triangle']
    aps = []
    
    for class_id in range(3):
        # Filter predictions and ground truths for this class
        class_preds = []
        class_gts = []
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            # Filter predictions for this class
            class_mask = pred['labels'] == class_id
            class_pred = {
                'boxes': pred['boxes'][class_mask],
                'scores': pred['scores'][class_mask],
                'labels': pred['labels'][class_mask]
            }
            
            # Filter ground truth for this class
            gt_class_mask = gt['labels'] == class_id
            class_gt = {
                'boxes': gt['boxes'][gt_class_mask],
                'labels': gt['labels'][gt_class_mask]
            }
            
            class_preds.append(class_pred)
            class_gts.append(class_gt)
        
        # Compute AP for this class
        ap = compute_ap(class_preds, class_gts)
        aps.append(ap)
        print(f'AP for {class_names[class_id]}: {ap:.4f}')
    
    # Mean Average Precision
    mAP = np.mean(aps)
    print(f'mAP: {mAP:.4f}')
    
    # Analyze scale performance
    scale_analysis = analyze_scale_performance(model, val_loader, anchors, device)
    plot_scale_specialization(scale_analysis, 'results/visualizations/scale_analysis.png')
    
    # Save results
    results = {
        'mAP': mAP,
        'class_APs': {class_names[i]: aps[i] for i in range(3)},
        'scale_analysis': scale_analysis
    }
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\nEvaluation complete!')
    print('Results saved to results/evaluation_results.json')
    print('Visualizations saved to results/visualizations/')

if __name__ == '__main__':
    main()
