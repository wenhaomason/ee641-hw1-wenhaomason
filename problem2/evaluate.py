import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloader
from model import create_heatmap_model, create_regression_model, heatmap_to_coords


def compute_pck(predicted_coords, target_coords, threshold=0.1, image_size=128):
    """
    Compute Percentage of Correct Keypoints (PCK).
    
    Args:
        predicted_coords: Tensor of shape [N, num_keypoints, 2] with predicted (x, y) coordinates in [0, 1]
        target_coords: Tensor of shape [N, num_keypoints, 2] with ground truth (x, y) coordinates in [0, 1]
        threshold: Distance threshold as fraction of image size
        image_size: Size of the image for scaling
        
    Returns:
        pck_scores: Dict containing PCK scores per keypoint and overall
    """
    batch_size, num_keypoints, _ = predicted_coords.shape
    
    # Convert normalized coordinates to pixel coordinates
    pred_pixels = predicted_coords * image_size
    target_pixels = target_coords * image_size
    
    # Compute Euclidean distances
    distances = torch.sqrt(torch.sum((pred_pixels - target_pixels) ** 2, dim=2))  # [N, num_keypoints]
    
    # Compute threshold in pixels
    threshold_pixels = threshold * image_size
    
    # Count correct keypoints (distance < threshold)
    correct = distances < threshold_pixels  # [N, num_keypoints]
    
    # Compute PCK for each keypoint
    pck_per_keypoint = torch.mean(correct.float(), dim=0)  # [num_keypoints]
    
    # Compute overall PCK
    overall_pck = torch.mean(correct.float())
    
    keypoint_names = ['head', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
    
    pck_scores = {
        'overall': overall_pck.item(),
        'per_keypoint': {
            keypoint_names[i]: pck_per_keypoint[i].item() 
            for i in range(min(len(keypoint_names), num_keypoints))
        },
        'mean_distance': torch.mean(distances).item()
    }
    
    return pck_scores

def plot_pck_curves(pck_heatmap, pck_regression, thresholds, save_path):
    """Plot PCK curves comparing both methods and save to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    xs = thresholds
    hm = [pck_heatmap[f'PCK@{t}']['overall'] for t in thresholds]
    rg = [pck_regression[f'PCK@{t}']['overall'] for t in thresholds]
    plt.figure(figsize=(6,4))
    plt.plot(xs, hm, marker='o', label='Heatmap')
    plt.plot(xs, rg, marker='s', label='Regression')
    plt.xlabel('Threshold')
    plt.ylabel('PCK')
    plt.title('PCK Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, device, model_type='heatmap'):
    """
    Evaluate a model on the validation set.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        model_type: 'heatmap' or 'regression'
        
    Returns:
        results: Dictionary with evaluation results
    """
    model.eval()
    
    all_predicted_coords = []
    all_target_coords = []
    per_sample_preds = []  # store per-sample coords for failure-case analysis
    total_loss = 0
    num_batches = 0
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            if model_type == 'heatmap':
                # Convert heatmaps to coordinates
                predicted_coords = heatmap_to_coords(outputs)  # [B, num_keypoints, 2]
                
                # Convert target heatmaps to coordinates for comparison
                target_coords = heatmap_to_coords(targets)  # [B, num_keypoints, 2]
                
                # Loss computation (MSE on heatmaps)
                loss = criterion(outputs, targets)
                
            else:  # regression
                # Outputs are already coordinates
                predicted_coords = outputs.view(-1, 5, 2)  # [B, 5, 2]
                target_coords = targets.view(-1, 5, 2)  # [B, 5, 2]
                
                # Loss computation (MSE on coordinates)
                loss = criterion(outputs, targets)
            
            all_predicted_coords.append(predicted_coords.cpu())
            all_target_coords.append(target_coords.cpu())
            per_sample_preds.append(predicted_coords.cpu())
            
            total_loss += loss.item()
            num_batches += 1
    
    # Concatenate all predictions and targets
    all_predicted_coords = torch.cat(all_predicted_coords, dim=0)
    all_target_coords = torch.cat(all_target_coords, dim=0)
    
    # Compute PCK scores at different thresholds
    pck_results = {}
    thresholds = [0.05, 0.1, 0.15, 0.2]
    
    for threshold in thresholds:
        pck_scores = compute_pck(all_predicted_coords, all_target_coords, threshold)
        pck_results[f'PCK@{threshold}'] = pck_scores
    
    avg_loss = total_loss / num_batches
    
    results = {
        'loss': avg_loss,
        'pck_results': pck_results,
        'num_samples': len(all_predicted_coords),
        'pred_coords': all_predicted_coords.numpy().tolist(),
        'gt_coords': all_target_coords.numpy().tolist()
    }
    
    return results

def visualize_predictions(model, dataloader, device, model_type='heatmap', num_samples=5, save_dir='results'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataloader: DataLoader 
        device: Device to run inference on
        model_type: 'heatmap' or 'regression'
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Take first image from batch
            image = images[0]  # [1, 128, 128]
            target = targets[0]
            
            # Forward pass
            output = model(image.unsqueeze(0))  # Add batch dimension
            
            pred_heatmaps = None
            target_heatmaps = None
            if model_type == 'heatmap':
                # Convert heatmaps to coordinates
                pred_coords = heatmap_to_coords(output)[0]  # [5, 2]
                target_coords = heatmap_to_coords(target.unsqueeze(0))[0]  # [5, 2]
                
                # Also get the heatmaps for visualization
                pred_heatmaps = output[0]  # [5, 64, 64]
                target_heatmaps = target  # [5, 64, 64]
            else:
                # Outputs are coordinates
                pred_coords = output[0].view(5, 2)  # [5, 2]
                target_coords = target.view(5, 2)  # [5, 2]
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            img_np = image[0].cpu().numpy()  # Remove channel dimension
            axes[0, 0].imshow(img_np, cmap='gray')
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            # Ground truth keypoints
            axes[0, 1].imshow(img_np, cmap='gray')
            target_coords_pixel = target_coords.cpu().numpy() * 128  # Convert to image coordinates
            axes[0, 1].scatter(target_coords_pixel[:, 0], target_coords_pixel[:, 1], 
                             c='red', s=50, marker='o', label='Ground Truth')
            axes[0, 1].set_title('Ground Truth Keypoints')
            axes[0, 1].legend()
            axes[0, 1].axis('off')
            
            # Predicted keypoints
            axes[0, 2].imshow(img_np, cmap='gray')
            pred_coords_pixel = pred_coords.cpu().numpy() * 128  # Convert to image coordinates
            axes[0, 2].scatter(pred_coords_pixel[:, 0], pred_coords_pixel[:, 1], 
                             c='blue', s=50, marker='x', label='Predicted')
            axes[0, 2].set_title('Predicted Keypoints')
            axes[0, 2].legend()
            axes[0, 2].axis('off')
            
            # Comparison
            axes[1, 0].imshow(img_np, cmap='gray')
            axes[1, 0].scatter(target_coords_pixel[:, 0], target_coords_pixel[:, 1], 
                             c='red', s=50, marker='o', label='Ground Truth')
            axes[1, 0].scatter(pred_coords_pixel[:, 0], pred_coords_pixel[:, 1], 
                             c='blue', s=50, marker='x', label='Predicted')
            axes[1, 0].set_title('Comparison')
            axes[1, 0].legend()
            axes[1, 0].axis('off')
            
            if model_type == 'heatmap' and pred_heatmaps is not None and target_heatmaps is not None:
                # Show one example heatmap (head keypoint)
                axes[1, 1].imshow(target_heatmaps[0].cpu().numpy(), cmap='hot')
                axes[1, 1].set_title('Target Heatmap (Head)')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(pred_heatmaps[0].cpu().numpy(), cmap='hot')
                axes[1, 2].set_title('Predicted Heatmap (Head)')
                axes[1, 2].axis('off')
            else:
                # For regression, show error bars
                errors = np.linalg.norm(pred_coords_pixel - target_coords_pixel, axis=1)
                keypoint_names = ['head', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
                
                axes[1, 1].bar(keypoint_names, errors)
                axes[1, 1].set_title('Per-Keypoint Error (pixels)')
                axes[1, 1].set_ylabel('Euclidean Distance')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{model_type}_sample_{batch_idx + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved visualization: {save_path}')

def find_failure_cases(hm_results, rg_results, threshold=0.1, max_examples=5):
    """Identify indices where heatmap succeeds but regression fails, vice versa, and both fail."""
    hm_preds = torch.tensor(hm_results['pred_coords'])  # [N,5,2]
    rg_preds = torch.tensor(rg_results['pred_coords'])
    gts = torch.tensor(hm_results['gt_coords'])
    N = hm_preds.shape[0]
    img_size = 128
    thr_pix = threshold * img_size

    def correct_mask(pred):
        d = torch.sqrt(torch.sum(((pred - gts) * img_size) ** 2, dim=2))  # [N,5]
        return (d < thr_pix)  # [N,5]

    hm_ok = correct_mask(hm_preds)
    rg_ok = correct_mask(rg_preds)

    # success defined if all keypoints within threshold
    hm_success = hm_ok.all(dim=1)
    rg_success = rg_ok.all(dim=1)

    cases = {
        'heatmap_success_regression_fail': [],
        'regression_success_heatmap_fail': [],
        'both_fail': []
    }

    for i in range(N):
        if len(cases['heatmap_success_regression_fail']) < max_examples and hm_success[i] and not rg_success[i]:
            cases['heatmap_success_regression_fail'].append(i)
        if len(cases['regression_success_heatmap_fail']) < max_examples and rg_success[i] and not hm_success[i]:
            cases['regression_success_heatmap_fail'].append(i)
        if len(cases['both_fail']) < max_examples and not hm_success[i] and not rg_success[i]:
            cases['both_fail'].append(i)
        if all(len(v) >= max_examples for v in cases.values()):
            break
    return cases

def visualize_failure_cases(dataloader, hm_model, rg_model, device, cases, save_dir='results/visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    # gather a flat list of indices we need
    needed = set(cases['heatmap_success_regression_fail'] + cases['regression_success_heatmap_fail'] + cases['both_fail'])
    images = []
    gts = []
    # iterate once to collect those indices
    idx_map = {}
    cursor = 0
    for imgs, targets in dataloader:
        b = imgs.shape[0]
        for j in range(b):
            if cursor in needed:
                images.append(imgs[j:j+1])
                gts.append(targets[j:j+1])
                idx_map[len(images)-1] = cursor
            cursor += 1
            if len(images) == len(needed):
                break
        if len(images) == len(needed):
            break
    if not images:
        return
    images = torch.cat(images, dim=0).to(device)
    gts = torch.cat(gts, dim=0).to(device)
    with torch.no_grad():
        hm_out = hm_model(images)
        hm_coords = heatmap_to_coords(hm_out)
        rg_coords = rg_model(images).view(-1, 5, 2)
    img_np = images.cpu().numpy()[:,0]
    gt_coords = gts.view(-1,5,2).cpu().numpy()
    hm_coords_np = hm_coords.cpu().numpy()
    rg_coords_np = rg_coords.cpu().numpy()
    # plot each category up to 3 examples
    def plot_case(indices, title_prefix):
        for k, idx in enumerate(indices):
            fig, ax = plt.subplots(1,1,figsize=(4,4))
            ax.imshow(img_np[k], cmap='gray')
            ax.scatter(gt_coords[k,:,0]*128, gt_coords[k,:,1]*128, c='g', label='GT')
            ax.scatter(hm_coords_np[k,:,0]*128, hm_coords_np[k,:,1]*128, c='b', marker='x', label='Heatmap')
            ax.scatter(rg_coords_np[k,:,0]*128, rg_coords_np[k,:,1]*128, c='r', marker='+', label='Regression')
            ax.set_title(f'{title_prefix} (idx={idx_map.get(k,idx)})')
            ax.legend()
            ax.axis('off')
            out = os.path.join(save_dir, f'{title_prefix.lower().replace(" ", "_")}_{k+1}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
    plot_case(cases['heatmap_success_regression_fail'], 'Heatmap success, Regression fail')
    plot_case(cases['regression_success_heatmap_fail'], 'Regression success, Heatmap fail')
    plot_case(cases['both_fail'], 'Both fail')

def main():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset paths
    val_image_dir = '../datasets/keypoints/val'
    val_ann_file = '../datasets/keypoints/val_annotations.json'
    
    if not os.path.exists(val_image_dir):
        print(f"Error: Validation dataset not found at {val_image_dir}")
        return
    
    print('Loading validation datasets...')
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Create data loaders
    heatmap_val_loader = get_dataloader(
        val_image_dir, val_ann_file, 
        output_type='heatmap', 
        batch_size=16, 
        shuffle=False
    )
    
    regression_val_loader = get_dataloader(
        val_image_dir, val_ann_file, 
        output_type='regression', 
        batch_size=16, 
        shuffle=False
    )
    
    # Load trained models
    print('\nLoading trained models...')
    
    # HeatmapNet
    heatmap_model = create_heatmap_model(num_keypoints=5)
    if os.path.exists('results/heatmap_model.pth'):
        checkpoint = torch.load('results/heatmap_model.pth', map_location=device)
        heatmap_model.load_state_dict(checkpoint['model_state_dict'])
        heatmap_model = heatmap_model.to(device)
        print('Loaded HeatmapNet model')
    else:
        print('Warning: HeatmapNet model not found. Please train the model first.')
        return
    
    # RegressionNet
    regression_model = create_regression_model(num_keypoints=5)
    if os.path.exists('results/regression_model.pth'):
        checkpoint = torch.load('results/regression_model.pth', map_location=device)
        regression_model.load_state_dict(checkpoint['model_state_dict'])
        regression_model = regression_model.to(device)
        print('Loaded RegressionNet model')
    else:
        print('Warning: RegressionNet model not found. Please train the model first.')
        return
    
    # Evaluate HeatmapNet
    print('\n' + '='*50)
    print('EVALUATING HEATMAPNET')
    print('='*50)
    
    heatmap_results = evaluate_model(heatmap_model, heatmap_val_loader, device, 'heatmap')
    
    print('HeatmapNet Results:')
    print(f'  Validation Loss: {heatmap_results["loss"]:.6f}')
    
    for threshold_key, pck_data in heatmap_results['pck_results'].items():
        print(f'  {threshold_key}:')
        print(f'    Overall: {pck_data["overall"]:.4f}')
        for kp_name, pck_score in pck_data['per_keypoint'].items():
            print(f'    {kp_name}: {pck_score:.4f}')
        print(f'    Mean Distance: {pck_data["mean_distance"]:.2f} pixels')
    
    # Evaluate RegressionNet
    print('\n' + '='*50)
    print('EVALUATING REGRESSIONNET')
    print('='*50)
    
    regression_results = evaluate_model(regression_model, regression_val_loader, device, 'regression')
    
    print('RegressionNet Results:')
    print(f'  Validation Loss: {regression_results["loss"]:.6f}')
    
    for threshold_key, pck_data in regression_results['pck_results'].items():
        print(f'  {threshold_key}:')
        print(f'    Overall: {pck_data["overall"]:.4f}')
        for kp_name, pck_score in pck_data['per_keypoint'].items():
            print(f'    {kp_name}: {pck_score:.4f}')
        print(f'    Mean Distance: {pck_data["mean_distance"]:.2f} pixels')
    
    # Generate visualizations
    print('\n' + '='*50)
    print('GENERATING VISUALIZATIONS')
    print('='*50)
    
    print('Creating visualizations for HeatmapNet...')
    visualize_predictions(heatmap_model, heatmap_val_loader, device, 'heatmap', 5, 'results/visualizations')
    
    print('Creating visualizations for RegressionNet...')
    visualize_predictions(regression_model, regression_val_loader, device, 'regression', 5, 'results/visualizations')

    # Plot PCK curves
    thresholds = [0.05, 0.1, 0.15, 0.2]
    hm_pcks = heatmap_results['pck_results']
    rg_pcks = regression_results['pck_results']
    plot_pck_curves(hm_pcks, rg_pcks, thresholds, 'results/visualizations/pck_curves.png')

    # Failure case analysis: find samples where one succeeds and the other fails at 0.1
    print('Analyzing failure cases (threshold=0.1)...')
    cases = find_failure_cases(heatmap_results, regression_results, threshold=0.1, max_examples=5)
    # visualize those cases
    visualize_failure_cases(regression_val_loader, heatmap_model, regression_model, device, cases, save_dir='results/visualizations')
    
    # Save evaluation results
    evaluation_results = {
        'heatmap_model': heatmap_results,
        'regression_model': regression_results,
        'comparison': {
            'best_overall_pck_0.1': {
                'heatmap': heatmap_results['pck_results']['PCK@0.1']['overall'],
                'regression': regression_results['pck_results']['PCK@0.1']['overall']
            },
            'mean_distance': {
                'heatmap': heatmap_results['pck_results']['PCK@0.1']['mean_distance'],
                'regression': regression_results['pck_results']['PCK@0.1']['mean_distance']
            }
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print('\n' + '='*50)
    print('EVALUATION COMPLETED')
    print('='*50)
    
    # Print comparison
    hm_pck = heatmap_results['pck_results']['PCK@0.1']['overall']
    reg_pck = regression_results['pck_results']['PCK@0.1']['overall']
    
    print('Performance Comparison (PCK@0.1):')
    print(f'  HeatmapNet:    {hm_pck:.4f}')
    print(f'  RegressionNet: {reg_pck:.4f}')
    
    if hm_pck > reg_pck:
        print(f'  Winner: HeatmapNet (+{hm_pck - reg_pck:.4f})')
    else:
        print(f'  Winner: RegressionNet (+{reg_pck - hm_pck:.4f})')
    
    print('\nResults saved to:')
    print('  - results/evaluation_results.json')
    print('  - results/heatmap_sample_*.png')
    print('  - results/regression_sample_*.png')

if __name__ == '__main__':
    main()
