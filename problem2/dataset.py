import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        # Create mapping from image_id to image entries (which include keypoints)
        # In this dataset format, each entry under 'images' contains id, file_name, and keypoints
        self.images = {img['id']: img for img in self.data['images']}
        
        # Get list of image IDs
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        num_keypoints = len(keypoints)
        heatmaps = torch.zeros((num_keypoints, height, width))
        
        # Create coordinate grids
        x_coords, y_coords = torch.meshgrid(
            torch.arange(width), 
            torch.arange(height), 
            indexing='xy'
        )
        
        for i, (kx, ky) in enumerate(keypoints):
            # Scale keypoint coordinates to heatmap size
            # Original image is 128x128, heatmap is typically 64x64
            scale_x = width / 128.0
            scale_y = height / 128.0
            
            hm_x = kx * scale_x
            hm_y = ky * scale_y
            
            # Create 2D gaussian centered at keypoint location
            gaussian = torch.exp(-((x_coords - hm_x) ** 2 + (y_coords - hm_y) ** 2) / (2 * self.sigma ** 2))
            
            # Handle boundary cases - ensure keypoint is within heatmap bounds
            if 0 <= hm_x < width and 0 <= hm_y < height:
                heatmaps[i] = gaussian
        
        return heatmaps
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path)
        
        # Convert to grayscale and resize to 128x128
        if image.mode != 'L':
            image = image.convert('L')
        image = image.resize((128, 128))
        
        # Convert to tensor and normalize to [0, 1]
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension: [1, 128, 128]
        
        # Get keypoints (list of [x, y] coordinates for 5 keypoints)
        keypoints = image_info['keypoints']
        keypoints = np.array(keypoints).reshape(-1, 2)  # Reshape to [num_keypoints, 2]
        
        if self.output_type == 'heatmap':
            # Generate heatmaps
            targets = self.generate_heatmap(keypoints, self.heatmap_size, self.heatmap_size)
        else:  # regression
            # Normalize keypoints to [0, 1] (original image is 128x128)
            normalized_keypoints = keypoints / 128.0
            targets = torch.from_numpy(normalized_keypoints.flatten()).float()
        
        return image_tensor, targets

def collate_fn_heatmap(batch):
    """Collate function for heatmap output."""
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return images, targets

def collate_fn_regression(batch):
    """Collate function for regression output."""
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return images, targets

def get_dataloader(image_dir, annotation_file, output_type='heatmap', 
                  batch_size=16, shuffle=True, **kwargs):
    """
    Get a DataLoader for the keypoint dataset.
    
    Args:
        image_dir: Path to image directory
        annotation_file: Path to annotation file
        output_type: 'heatmap' or 'regression'
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments for KeypointDataset
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    dataset = KeypointDataset(image_dir, annotation_file, output_type=output_type, **kwargs)
    
    if output_type == 'heatmap':
        collate_fn = collate_fn_heatmap
    else:
        collate_fn = collate_fn_regression
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=2
    )
