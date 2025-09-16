import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load and parse annotations
        with open(annotation_file, 'r') as f:
            self.image_annotations = json.load(f)
        
        # Create mapping from image_id to annotations
        self.images = {img['id']: img for img in self.image_annotations['images']}
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in self.image_annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)
        
        # Store image paths and corresponding annotations
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations.get(image_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        img_w = image_info.get('width', None)
        img_h = image_info.get('height', None)

        for ann in anns:
            bbox = ann['bbox']
            if len(bbox) != 4:
                continue
            x0, y0, a, b = bbox
            # Our generator saves [x1, y1, x2, y2]. Prefer that if values look like absolute corners.
            # Fallback to COCO [x, y, w, h] if needed.
            if (
                img_w is not None and img_h is not None and
                x0 <= a <= img_w and y0 <= b <= img_h and a > x0 and b > y0
            ):
                x1, y1, x2, y2 = x0, y0, a, b
            else:
                x1, y1 = x0, y0
                x2, y2 = x0 + a, y0 + b

            # Clamp to valid image bounds if available
            if img_w is not None and img_h is not None:
                x1 = max(0.0, min(float(x1), float(img_w)))
                y1 = max(0.0, min(float(y1), float(img_h)))
                x2 = max(0.0, min(float(x2), float(img_w)))
                y2 = max(0.0, min(float(y2), float(img_h)))
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.FloatTensor(boxes)
            labels = torch.LongTensor(labels)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: PIL to tensor and normalize
            import numpy as np
            image_array = np.array(image)
            image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        targets = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, targets

def collate_fn(batch):
    """
    Custom collate function to handle variable number of objects per image.
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images, 0)
    
    return images, targets

def default_transform(image):
    """
    Standard transform for the dataset.
    Convert PIL image to tensor and normalize to [0, 1].
    """
    # Convert PIL image to tensor
    import numpy as np
    image_array = np.array(image)
    # Convert from HWC to CHW and normalize to [0, 1]
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    return image_tensor


def get_transform():
    """ 
    Get standard transform for the dataset.
    Returns the module-level transform function that can be pickled.
    """
    return default_transform
