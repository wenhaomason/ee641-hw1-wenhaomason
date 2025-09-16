import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import create_heatmap_model, create_regression_model


def _save_heatmap_snapshot(images, pred_heatmaps, targets, save_path, max_kp=5):
    """Save a side-by-side snapshot of predicted vs target heatmaps for first sample."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred = pred_heatmaps[0].detach().cpu().numpy()  # [K,64,64]
    tgt = targets[0].detach().cpu().numpy()  # [K,64,64]
    k = min(pred.shape[0], max_kp)
    cols = 2
    rows = k
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    for i in range(k):
        axes[i, 0].imshow(tgt[i], cmap='hot')
        axes[i, 0].set_title(f'Target k{i}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred[i], cmap='hot')
        axes[i, 1].set_title(f'Pred k{i}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_heatmap_model(model, train_loader, val_loader, device, num_epochs=30, snapshot_epochs=(1,10,20,30)):
    """Train the heatmap-based model using MSE on heatmaps."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Training HeatmapNet...")
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, heatmaps) in enumerate(train_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            optimizer.zero_grad()
            
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, heatmaps)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        if (epoch + 1) in set(snapshot_epochs):
            model.eval()
            with torch.no_grad():
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_targets = sample_targets.to(device)
                preds = model(sample_images)
                snap_path = f'results/visualizations/heatmaps_epoch_{epoch+1:02d}.png'
                _save_heatmap_snapshot(sample_images, preds, sample_targets, snap_path)
                print(f'  Saved heatmap snapshot: {snap_path}')
            model.train()
        
        avg_train_loss = train_loss / train_batches
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                pred_heatmaps = model(images)
                loss = criterion(pred_heatmaps, heatmaps)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss
            }, 'results/heatmap_model.pth')
            print(f'  Saved best model with val loss: {best_val_loss:.6f}')
        
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def train_regression_model(model, train_loader, val_loader, device, num_epochs=30):
    """Train the direct regression model using MSE on coordinates."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Training RegressionNet...")
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, coords) in enumerate(train_loader):
            images = images.to(device)
            coords = coords.to(device)
            
            optimizer.zero_grad()
            
            pred_coords = model(images)
            loss = criterion(pred_coords, coords)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / train_batches
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, coords in val_loader:
                images = images.to(device)
                coords = coords.to(device)
                pred_coords = model(images)
                loss = criterion(pred_coords, coords)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss
            }, 'results/regression_model.pth')
            print(f'  Saved best model with val loss: {best_val_loss:.6f}')
        
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def main():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    torch.manual_seed(641)
    
    train_image_dir = '../datasets/keypoints/train'
    train_ann_file = '../datasets/keypoints/train_annotations.json'
    val_image_dir = '../datasets/keypoints/val'
    val_ann_file = '../datasets/keypoints/val_annotations.json'
    
    if not os.path.exists(train_image_dir):
        print(f"Error: Dataset not found at {train_image_dir}")
        print("Please run the dataset generation script first:")
        print("python generate_datasets.py --seed 641 --num_train 1000 --num_val 200")
        return
    
    batch_size = 32
    num_epochs = 30
    
    print('Training parameters:')
    print(f'  Batch size: {batch_size}')
    print(f'  Number of epochs: {num_epochs}')
    
    print('\nCreating data loaders for HeatmapNet...')
    heatmap_train_loader = get_dataloader(
        train_image_dir, train_ann_file, 
        output_type='heatmap', 
        batch_size=batch_size, 
        shuffle=True
    )
    heatmap_val_loader = get_dataloader(
        val_image_dir, val_ann_file, 
        output_type='heatmap', 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print('\nCreating data loaders for RegressionNet...')
    regression_train_loader = get_dataloader(
        train_image_dir, train_ann_file, 
        output_type='regression', 
        batch_size=batch_size, 
        shuffle=True
    )
    regression_val_loader = get_dataloader(
        val_image_dir, val_ann_file, 
        output_type='regression', 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print('\n' + '='*60)
    print('TRAINING HEATMAP MODEL')
    print('='*60)
    
    heatmap_model = create_heatmap_model(num_keypoints=5, use_skip_connections=True)
    heatmap_results = train_heatmap_model(
        heatmap_model, heatmap_train_loader, heatmap_val_loader, device, num_epochs
    )
    
    print('\n' + '='*60)
    print('TRAINING REGRESSION MODEL')
    print('='*60)
    
    regression_model = create_regression_model(num_keypoints=5)
    regression_results = train_regression_model(
        regression_model, regression_train_loader, regression_val_loader, device, num_epochs
    )
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    training_log = {
        'heatmap_model': {
            'train_losses': heatmap_results['train_losses'],
            'val_losses': heatmap_results['val_losses'],
            'best_val_loss': heatmap_results['best_val_loss']
        },
        'regression_model': {
            'train_losses': regression_results['train_losses'],
            'val_losses': regression_results['val_losses'],
            'best_val_loss': regression_results['best_val_loss']
        },
        'training_params': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'device': str(device)
        }
    }
    
    with open('results/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print('\n' + '='*60)
    print('TRAINING COMPLETED')
    print('='*60)
    print(f'HeatmapNet best validation loss: {heatmap_results["best_val_loss"]:.6f}')
    print(f'RegressionNet best validation loss: {regression_results["best_val_loss"]:.6f}')
    print('\nModels saved:')
    print('  - results/heatmap_model.pth')
    print('  - results/regression_model.pth')
    print('  - results/training_log.json')

if __name__ == '__main__':
    main()
