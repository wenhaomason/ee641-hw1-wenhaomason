import torch.nn as nn


class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors_per_location=3):
        """
        Multi-scale single-shot detector with feature pyramid.
        
        Args:
            num_classes: Number of object classes (3 for our shapes)
            num_anchors_per_location: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_location
        
        # Backbone network - 4 convolutional blocks
        self.backbone = self._build_backbone()
        
        # Detection heads for each scale
        self.detection_heads = self._build_detection_heads()
    
    def _build_backbone(self):
        """Build the backbone network with 4 convolutional blocks."""
        layers = nn.ModuleDict()
        
        # Block 1 (Stem): Conv(3→32, stride=1) → BN → ReLU → Conv(32→64, stride=2) → BN → ReLU [224→112]
        layers['block1'] = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: Conv(64→128, stride=2) → BN → ReLU [112→56] → Output as Scale 1
        layers['block2'] = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: Conv(128→256, stride=2) → BN → ReLU [56→28] → Output as Scale 2
        layers['block3'] = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: Conv(256→512, stride=2) → BN → ReLU [28→14] → Output as Scale 3
        layers['block4'] = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        return layers
    
    def _build_detection_heads(self):
        """Build detection heads for each scale."""
        heads = nn.ModuleDict()
        
        # Scale 1: 128 channels (from block2)
        heads['scale1'] = self._make_detection_head(128)
        
        # Scale 2: 256 channels (from block3)  
        heads['scale2'] = self._make_detection_head(256)
        
        # Scale 3: 512 channels (from block4)
        heads['scale3'] = self._make_detection_head(512)
        
        return heads
    
    def _make_detection_head(self, in_channels):
        """
        Create a detection head for a single scale.
        
        Args:
            in_channels: Number of input channels
            
        Returns:
            Detection head module
        """
        # Each spatial location predicts for each anchor:
        # 4 values: bbox offsets (tx, ty, tw, th)
        # 1 value: objectness score  
        # num_classes values: class scores
        out_channels = self.num_anchors * (5 + self.num_classes)
        
        return nn.Sequential(
            # 3×3 Conv (keep channels same)
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 1×1 Conv → num_anchors * (5 + num_classes) channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        features = []
        
        # Forward through backbone
        x = self.backbone['block1'](x)  # [B, 64, 112, 112]
        
        x = self.backbone['block2'](x)  # [B, 128, 56, 56] - Scale 1
        scale1_features = x
        features.append(scale1_features)
        
        x = self.backbone['block3'](x)  # [B, 256, 28, 28] - Scale 2
        scale2_features = x
        features.append(scale2_features)
        
        x = self.backbone['block4'](x)  # [B, 512, 14, 14] - Scale 3
        scale3_features = x
        features.append(scale3_features)
        
        # Apply detection heads to each scale
        predictions = []
        
        # Scale 1 predictions (56x56)
        pred1 = self.detection_heads['scale1'](features[0])
        predictions.append(pred1)
        
        # Scale 2 predictions (28x28)
        pred2 = self.detection_heads['scale2'](features[1])
        predictions.append(pred2)
        
        # Scale 3 predictions (14x14)
        pred3 = self.detection_heads['scale3'](features[2])
        predictions.append(pred3)
        
        return predictions
    
    def decode_predictions(self, predictions, anchors):
        """
        Decode raw predictions into bounding boxes and scores.
        
        Args:
            predictions: List of prediction tensors from forward()
            anchors: List of anchor tensors for each scale
            
        Returns:
            decoded_boxes: List of decoded box tensors for each scale
            objectness_scores: List of objectness score tensors
            class_scores: List of class score tensors
        """
        decoded_boxes = []
        objectness_scores = []
        class_scores = []
        
        for pred, anchor in zip(predictions, anchors):
            batch_size, channels, height, width = pred.shape
            
            # Reshape predictions: [B, num_anchors * (5 + num_classes), H, W] 
            # -> [B, num_anchors, 5 + num_classes, H, W]
            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, height, width)
            
            # Split into components
            bbox_deltas = pred[:, :, :4, :, :]  # [B, num_anchors, 4, H, W]
            obj_scores = pred[:, :, 4, :, :]    # [B, num_anchors, H, W]
            cls_scores = pred[:, :, 5:, :, :]   # [B, num_anchors, num_classes, H, W]
            
            # Decode bounding boxes from deltas
            # This would apply the anchor box transformations
            # For now, we'll return the raw predictions
            decoded_boxes.append(bbox_deltas)
            objectness_scores.append(obj_scores)
            class_scores.append(cls_scores)
        
        return decoded_boxes, objectness_scores, class_scores

def create_model(num_classes=3):
    """Create the multi-scale detector model."""
    return MultiScaleDetector(num_classes=num_classes)
