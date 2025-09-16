import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import encode_boxes, match_anchors_to_targets


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.
        
        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale
            
        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss  
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        batch_size = len(targets)
        device = predictions[0].device
        
        total_obj_loss = 0
        total_cls_loss = 0  
        total_loc_loss = 0
        total_positive_anchors = 0
        
        # Process each scale
        for scale_idx, (pred, anchor) in enumerate(zip(predictions, anchors)):
            B, C, H, W = pred.shape
            num_anchors_per_location = 3  # 3 scales per location
            
            # Reshape predictions: [B, num_anchors * (5 + num_classes), H, W] 
            # -> [B, H, W, num_anchors, 5 + num_classes]
            pred = pred.permute(0, 2, 3, 1).contiguous()
            pred = pred.view(B, H, W, num_anchors_per_location, 5 + self.num_classes)
            
            # Flatten spatial dimensions: [B, H*W*num_anchors, 5 + num_classes]
            pred = pred.view(B, -1, 5 + self.num_classes)
            
            # Split predictions
            bbox_deltas = pred[:, :, :4]      # [B, num_anchors, 4]
            obj_scores = pred[:, :, 4]        # [B, num_anchors]
            cls_scores = pred[:, :, 5:]       # [B, num_anchors, num_classes]
            
            # Expand anchors to match flattened format
            num_anchors_total = H * W * num_anchors_per_location
            anchor_expanded = anchor.view(-1, 4)  # [H*W*num_anchors, 4]
            
            # Process each image in the batch
            for batch_idx in range(batch_size):
                target = targets[batch_idx]
                target_boxes = target['boxes'].to(device)  # [num_targets, 4]
                target_labels = target['labels'].to(device)  # [num_targets]
                
                # Match anchors to targets
                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anchor_expanded, target_boxes, target_labels
                )
                
                matched_labels = matched_labels.to(device)
                matched_boxes = matched_boxes.to(device)
                pos_mask = pos_mask.to(device)
                neg_mask = neg_mask.to(device)
                
                # Objectness loss (Binary Cross Entropy)
                obj_targets = (matched_labels > 0).float()  # 1 for positive, 0 for negative
                obj_loss_all = F.binary_cross_entropy_with_logits(
                    obj_scores[batch_idx], obj_targets, reduction='none'
                )
                
                # Apply hard negative mining
                selected_neg_mask = self.hard_negative_mining(
                    obj_loss_all, pos_mask, neg_mask, ratio=3
                )
                
                # Combine positive and selected negative masks
                selected_mask = pos_mask | selected_neg_mask
                obj_loss = obj_loss_all[selected_mask].mean()
                total_obj_loss += obj_loss
                
                # Classification loss (only for positive anchors)
                if pos_mask.any():
                    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                    pos_cls_scores = cls_scores[batch_idx][pos_indices]  # [num_pos, num_classes]
                    pos_cls_targets = matched_labels[pos_indices] - 1    # Convert to 0-indexed
                    
                    cls_loss = F.cross_entropy(pos_cls_scores, pos_cls_targets)
                    total_cls_loss += cls_loss
                    
                    # Localization loss (Smooth L1, only for positive anchors)
                    pos_bbox_deltas = bbox_deltas[batch_idx][pos_indices]  # [num_pos, 4]
                    pos_matched_boxes = matched_boxes[pos_indices]         # [num_pos, 4]
                    pos_anchors = anchor_expanded[pos_indices]             # [num_pos, 4]
                    
                    # Encode target boxes relative to anchors
                    target_deltas = encode_boxes(pos_matched_boxes, pos_anchors)
                    target_deltas = target_deltas.to(device)
                    
                    loc_loss = F.smooth_l1_loss(pos_bbox_deltas, target_deltas)
                    total_loc_loss += loc_loss
                    
                    total_positive_anchors += pos_mask.sum().item()
        
        # Average losses
        if total_positive_anchors > 0:
            total_obj_loss /= batch_size
            total_cls_loss /= batch_size  
            total_loc_loss /= batch_size
        else:
            # Handle case with no positive anchors
            total_obj_loss /= batch_size
            total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_loc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Loss weights (as specified in homework)
        loss_weights = {
            'objectness': 1.0,
            'classification': 1.0,
            'localization': 2.0
        }
        
        # Weighted total loss
        total_loss = (
            loss_weights['objectness'] * total_obj_loss +
            loss_weights['classification'] * total_cls_loss + 
            loss_weights['localization'] * total_loc_loss
        )
        
        loss_dict = {
            'loss_obj': total_obj_loss,
            'loss_cls': total_cls_loss,
            'loss_loc': total_loc_loss,
            'loss_total': total_loss
        }
        
        return loss_dict
    
    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.
        
        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio
            
        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = pos_mask.sum().item()
        num_neg_needed = min(ratio * num_pos, neg_mask.sum().item())
        
        if num_neg_needed == 0:
            return torch.zeros_like(neg_mask)
        
        # Get losses for negative anchors only
        neg_losses = loss.clone()
        neg_losses[~neg_mask] = -float('inf')  # Mask out non-negative anchors
        
        # Select top-k hardest negatives
        _, hard_neg_indices = torch.topk(neg_losses, num_neg_needed, largest=True)
        
        # Create mask for selected negatives
        selected_neg_mask = torch.zeros_like(neg_mask)
        selected_neg_mask[hard_neg_indices] = True
        
        # Ensure we only select from actual negative anchors
        selected_neg_mask = selected_neg_mask & neg_mask
        
        return selected_neg_mask
