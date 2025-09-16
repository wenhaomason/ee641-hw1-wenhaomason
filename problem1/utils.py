import torch


def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    anchors = []
    
    for (feat_h, feat_w), scales in zip(feature_map_sizes, anchor_scales):
        # Calculate stride for this feature map
        stride_h = image_size / feat_h
        stride_w = image_size / feat_w
        
        # Generate grid of anchor centers
        shift_x = torch.arange(0, feat_w) * stride_w + stride_w / 2
        shift_y = torch.arange(0, feat_h) * stride_h + stride_h / 2
        
        # Create meshgrid
        shift_xx, shift_yy = torch.meshgrid(shift_x, shift_y, indexing='xy')
        shift_xx = shift_xx.reshape(-1)
        shift_yy = shift_yy.reshape(-1)
        
        # Generate anchors for each scale at each location
        # We build per-scale anchors first, then interleave per location so that
        # the final ordering is [(loc0, a0..aK), (loc1, a0..aK), ...],
        # matching the prediction flattening order.
        per_scale = []
        for scale in scales:
            half_size = scale / 2
            x1 = shift_xx - half_size
            y1 = shift_yy - half_size
            x2 = shift_xx + half_size
            y2 = shift_yy + half_size
            per_scale.append(torch.stack([x1, y1, x2, y2], dim=1))  # [L,4]

        # Stack as [L, A, 4] then reshape to [L*A, 4] with A varying fastest
        feat_anchors = torch.stack(per_scale, dim=1).reshape(-1, 4)
        
        # Clip anchors to image boundaries
        feat_anchors[:, 0] = torch.clamp(feat_anchors[:, 0], min=0)  # x1
        feat_anchors[:, 1] = torch.clamp(feat_anchors[:, 1], min=0)  # y1
        feat_anchors[:, 2] = torch.clamp(feat_anchors[:, 2], max=image_size)  # x2
        feat_anchors[:, 3] = torch.clamp(feat_anchors[:, 3], max=image_size)  # y2
        
        anchors.append(feat_anchors)
    
    return anchors

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    # Expand dimensions to enable broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # Calculate intersection
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])  # [N, M]
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])  # [N, M]
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])  # [N, M]
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])  # [N, M]
    
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas of both boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N, 1]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, M]
    
    # Calculate union
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou

def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    num_anchors = anchors.shape[0]
    num_targets = target_boxes.shape[0]
    
    if num_targets == 0:
        # No targets, all anchors are negative
        matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
        matched_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
        neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=anchors.device)
        return matched_labels, matched_boxes, pos_mask, neg_mask
    
    # Compute IoU between all anchors and all targets
    iou_matrix = compute_iou(anchors, target_boxes)  # [num_anchors, num_targets]
    
    # Find best matching target for each anchor
    max_iou_per_anchor, best_target_per_anchor = torch.max(iou_matrix, dim=1)
    
    # Initialize all as background (label 0)
    matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
    matched_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)
    
    # Positive anchors: IoU >= pos_threshold
    pos_mask = max_iou_per_anchor >= pos_threshold
    
    # Negative anchors: IoU < neg_threshold
    neg_mask = max_iou_per_anchor < neg_threshold
    
    # Ignore anchors: neg_threshold <= IoU < pos_threshold (not used explicitly)
    
    # Assign labels and boxes for positive anchors
    if pos_mask.any():
        pos_target_indices = best_target_per_anchor[pos_mask]
        matched_labels[pos_mask] = target_labels[pos_target_indices] + 1  # +1 because 0 is background
        matched_boxes[pos_mask] = target_boxes[pos_target_indices]
    
    # Ensure each target has at least one positive anchor (highest IoU)
    for target_idx in range(num_targets):
        _, best_anchor_idx = torch.max(iou_matrix[:, target_idx], dim=0)
        matched_labels[best_anchor_idx] = target_labels[target_idx] + 1
        matched_boxes[best_anchor_idx] = target_boxes[target_idx]
        pos_mask[best_anchor_idx] = True
        neg_mask[best_anchor_idx] = False
    
    return matched_labels, matched_boxes, pos_mask, neg_mask

def encode_boxes(matched_boxes, anchors):
    """
    Encode matched boxes relative to anchors.
    
    Args:
        matched_boxes: Tensor of shape [num_anchors, 4] - ground truth boxes
        anchors: Tensor of shape [num_anchors, 4] - anchor boxes
        
    Returns:
        encoded_boxes: Tensor of shape [num_anchors, 4] - encoded box deltas
    """
    # Convert to center coordinates and dimensions
    # Anchors
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Matched boxes
    matched_cx = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2
    matched_cy = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2
    matched_w = matched_boxes[:, 2] - matched_boxes[:, 0]
    matched_h = matched_boxes[:, 3] - matched_boxes[:, 1]
    
    # Encode as deltas
    tx = (matched_cx - anchor_cx) / (anchor_w + 1e-8)
    ty = (matched_cy - anchor_cy) / (anchor_h + 1e-8)
    tw = torch.log(matched_w / (anchor_w + 1e-8) + 1e-8)
    th = torch.log(matched_h / (anchor_h + 1e-8) + 1e-8)
    
    encoded_boxes = torch.stack([tx, ty, tw, th], dim=1)
    return encoded_boxes

def decode_boxes(box_deltas, anchors):
    """
    Decode box deltas back to absolute coordinates.
    
    Args:
        box_deltas: Tensor of shape [..., 4] - encoded box deltas
        anchors: Tensor of shape [..., 4] - anchor boxes
        
    Returns:
        decoded_boxes: Tensor of shape [..., 4] - decoded boxes in [x1, y1, x2, y2]
    """
    # Convert anchors to center coordinates and dimensions
    anchor_cx = (anchors[..., 0] + anchors[..., 2]) / 2
    anchor_cy = (anchors[..., 1] + anchors[..., 3]) / 2
    anchor_w = anchors[..., 2] - anchors[..., 0]
    anchor_h = anchors[..., 3] - anchors[..., 1]
    
    # Decode deltas
    tx, ty, tw, th = box_deltas[..., 0], box_deltas[..., 1], box_deltas[..., 2], box_deltas[..., 3]
    
    # Apply deltas
    pred_cx = tx * anchor_w + anchor_cx
    pred_cy = ty * anchor_h + anchor_cy
    pred_w = torch.exp(tw) * anchor_w
    pred_h = torch.exp(th) * anchor_h
    
    # Convert back to corner coordinates
    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2
    
    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return decoded_boxes

def get_anchor_configuration():
    """
    Get the anchor configuration for the three scales.
    
    Returns:
        feature_map_sizes: List of (H, W) tuples
        anchor_scales: List of scale lists for each feature map
    """
    # Feature map sizes for each scale
    feature_map_sizes = [
        (56, 56),  # Scale 1
        (28, 28),  # Scale 2  
        (14, 14)   # Scale 3
    ]
    
    # Anchor scales for each feature map
    anchor_scales = [
        [16, 24, 32],     # Scale 1 (56×56): anchor scales [16, 24, 32]
        [48, 64, 96],     # Scale 2 (28×28): anchor scales [48, 64, 96]
        [96, 128, 192]    # Scale 3 (14×14): anchor scales [96, 128, 192]
    ]
    
    return feature_map_sizes, anchor_scales
