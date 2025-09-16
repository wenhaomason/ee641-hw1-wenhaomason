import json
import os

import torch
import torch.optim as optim
from dataset import ShapeDetectionDataset, collate_fn, get_transform
from loss import DetectionLoss
from model import create_model
from torch.utils.data import DataLoader
from utils import generate_anchors, get_anchor_configuration


def train_one_epoch(model, dataloader, criterion, optimizer, anchors, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_losses = {
        'loss_obj': 0.0,
        'loss_cls': 0.0,
        'loss_loc': 0.0,
        'loss_total': 0.0
    }
    
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
        predictions = model(images)
        loss_dict = criterion(predictions, targets, anchors)
        optimizer.zero_grad()
        loss_dict['loss_total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for key in total_losses:
            total_losses[key] += loss_dict[key].item()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                  f'Loss: {loss_dict["loss_total"].item():.4f} '
                  f'(Obj: {loss_dict["loss_obj"].item():.4f}, '
                  f'Cls: {loss_dict["loss_cls"].item():.4f}, '
                  f'Loc: {loss_dict["loss_loc"].item():.4f})')
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses

def validate(model, dataloader, criterion, anchors, device):
    """Validate the model."""
    model.eval()
    total_losses = {
        'loss_obj': 0.0,
        'loss_cls': 0.0,
        'loss_loc': 0.0,
        'loss_total': 0.0
    }
    
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            for target in targets:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)
            predictions = model(images)
            loss_dict = criterion(predictions, targets, anchors)
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(42)
    train_image_dir = '../datasets/detection/train'
    train_ann_file = '../datasets/detection/train_annotations.json'
    val_image_dir = '../datasets/detection/val'
    val_ann_file = '../datasets/detection/val_annotations.json'
    transform = get_transform()
    train_dataset = ShapeDetectionDataset(train_image_dir, train_ann_file, transform=transform)
    val_dataset = ShapeDetectionDataset(val_image_dir, val_ann_file, transform=transform)
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    model = create_model(num_classes=3)
    model = model.to(device)
    feature_map_sizes, anchor_scales = get_anchor_configuration()
    anchors = generate_anchors(feature_map_sizes, anchor_scales)
    anchors = [anchor.to(device) for anchor in anchors]
    
    print(f'Generated anchors for {len(anchors)} scales')
    for i, anchor in enumerate(anchors):
        print(f'  Scale {i+1}: {anchor.shape[0]} anchors')
    
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    num_epochs = 50
    best_val_loss = float('inf')
    os.makedirs('results/visualizations', exist_ok=True)

    training_log = {
        'train_losses': [],
        'val_losses': [],
        'epochs': []
    }
    
    print('Starting training...')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, anchors, device, epoch+1
        )
        val_losses = validate(model, val_loader, criterion, anchors, device)
        scheduler.step()
        print(f'\nEpoch {epoch+1} Results:')
        print(f'Train Loss - Total: {train_losses["loss_total"]:.4f}, '
              f'Obj: {train_losses["loss_obj"]:.4f}, '
              f'Cls: {train_losses["loss_cls"]:.4f}, '
              f'Loc: {train_losses["loss_loc"]:.4f}')
        print(f'Val Loss - Total: {val_losses["loss_total"]:.4f}, '
              f'Obj: {val_losses["loss_obj"]:.4f}, '
              f'Cls: {val_losses["loss_cls"]:.4f}, '
              f'Loc: {val_losses["loss_loc"]:.4f}')
        
        # Save training log
        training_log['epochs'].append(epoch + 1)
        training_log['train_losses'].append(train_losses)
        training_log['val_losses'].append(val_losses)
        
        if val_losses['loss_total'] < best_val_loss:
            best_val_loss = val_losses['loss_total']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_losses['loss_total']
            }, 'results/best_model.pth')
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')
        with open('results/training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()
