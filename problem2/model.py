import torch
import torch.nn as nn


class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5, use_skip_connections=True):
        """
        Initialize the heatmap regression network.

        Args:
            num_keypoints: Number of keypoints to detect
            use_skip_connections: Whether to use encoder-decoder skip connections
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_skip_connections = use_skip_connections

        # Encoder (downsampling path)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )

        # Decoder (upsampling path)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # deconv3 input channels depend on whether we concat skip from enc3
        deconv3_in = 256 if self.use_skip_connections else 128
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(deconv3_in, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # deconv2 input channels depend on whether we concat skip from enc2
        deconv2_in = 128 if self.use_skip_connections else 64
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(deconv2_in, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final layer to produce heatmaps at 64x64
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)    # [B, 32, 64, 64]
        enc2 = self.conv2(enc1) # [B, 64, 32, 32]
        enc3 = self.conv3(enc2) # [B, 128, 16, 16]
        enc4 = self.conv4(enc3) # [B, 256, 8, 8]

        # Decoder
        dec4 = self.deconv4(enc4)  # [B, 128, 16, 16]
        dec4_cat = torch.cat([dec4, enc3], dim=1) if self.use_skip_connections else dec4  # [B, 256|128, 16,16]
        dec3 = self.deconv3(dec4_cat)  # [B, 64, 32, 32]

        dec3_cat = torch.cat([dec3, enc2], dim=1) if self.use_skip_connections else dec3  # [B,128|64,32,32]
        dec2 = self.deconv2(dec3_cat)  # [B, 32, 64, 64]

        heatmaps = self.final(dec2)  # [B, num_keypoints, 64, 64]
        return heatmaps

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use same encoder architecture as HeatmapNet
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )
        
        # Global Average Pooling + FC layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, 256, 8, 8] -> [B, 256, 1, 1]
        
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, num_keypoints * 2),
            nn.Sigmoid()  # Output coordinates in [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        # Encoder
        x = self.conv1(x)  # [B, 32, 64, 64]
        x = self.conv2(x)  # [B, 64, 32, 32]
        x = self.conv3(x)  # [B, 128, 16, 16]
        x = self.conv4(x)  # [B, 256, 8, 8]
        
        # Global pooling
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # Fully connected layers
        x = self.fc1(x)  # [B, 128]
        x = self.fc2(x)  # [B, 64]
        coords = self.fc3(x)  # [B, num_keypoints * 2]
        
        return coords

def heatmap_to_coords(heatmaps, threshold=0.1):
    """
    Convert heatmaps to coordinate predictions.
    
    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]
        threshold: Minimum value to consider for finding peak
        
    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2] with (x, y) coordinates
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    coords = torch.zeros(batch_size, num_keypoints, 2, device=heatmaps.device)
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            heatmap = heatmaps[b, k]
            
            # Find the maximum value and its location
            max_val = torch.max(heatmap)
            
            if max_val > threshold:
                # Find the indices of the maximum value
                max_indices = torch.where(heatmap == max_val)
                
                # Take the first occurrence if there are multiple maxima
                y_coord = max_indices[0][0].float()
                x_coord = max_indices[1][0].float()
                
                # Convert to normalized coordinates [0, 1]
                coords[b, k, 0] = x_coord / (width - 1)
                coords[b, k, 1] = y_coord / (height - 1)
    
    return coords

def coords_to_heatmap(coords, heatmap_size=64, sigma=2.0):
    """
    Convert coordinate predictions to heatmaps for visualization.
    
    Args:
        coords: Tensor of shape [batch, num_keypoints, 2] with (x, y) coordinates in [0, 1]
        heatmap_size: Size of output heatmaps
        sigma: Gaussian sigma for heatmap generation
        
    Returns:
        heatmaps: Tensor of shape [batch, num_keypoints, heatmap_size, heatmap_size]
    """
    batch_size, num_keypoints, _ = coords.shape
    heatmaps = torch.zeros(batch_size, num_keypoints, heatmap_size, heatmap_size, 
                          device=coords.device)
    
    # Create coordinate grids
    x_coords, y_coords = torch.meshgrid(
        torch.arange(heatmap_size, device=coords.device),
        torch.arange(heatmap_size, device=coords.device),
        indexing='xy'
    )
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            # Get normalized coordinates and convert to heatmap space
            norm_x, norm_y = coords[b, k]
            hm_x = norm_x * (heatmap_size - 1)
            hm_y = norm_y * (heatmap_size - 1)
            
            # Create 2D gaussian
            gaussian = torch.exp(-((x_coords - hm_x) ** 2 + (y_coords - hm_y) ** 2) / (2 * sigma ** 2))
            heatmaps[b, k] = gaussian
    
    return heatmaps

def create_heatmap_model(num_keypoints=5, use_skip_connections=True):
    """Create the HeatmapNet model.

    Args:
        num_keypoints: number of keypoints to predict
        use_skip_connections: whether to enable encoder-decoder skip connections
    """
    return HeatmapNet(num_keypoints=num_keypoints, use_skip_connections=use_skip_connections)

def create_regression_model(num_keypoints=5):
    """Create the RegressionNet model."""
    return RegressionNet(num_keypoints=num_keypoints)
