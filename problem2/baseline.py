import json
import os

import torch
from dataset import get_dataloader
from evaluate import evaluate_model, plot_pck_curves
from model import create_heatmap_model, create_regression_model
from train import train_heatmap_model, train_regression_model


def run_experiment(heatmap_size=64, sigma=2.0, use_skip=True, num_epochs=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    train_image_dir = '../datasets/keypoints/train'
    train_ann_file = '../datasets/keypoints/train_annotations.json'
    val_image_dir = '../datasets/keypoints/val'
    val_ann_file = '../datasets/keypoints/val_annotations.json'
    os.makedirs('results/ablation', exist_ok=True)
    # loaders
    hm_train = get_dataloader(train_image_dir, train_ann_file, output_type='heatmap', batch_size=32, shuffle=True, heatmap_size=heatmap_size, sigma=sigma)
    hm_val = get_dataloader(val_image_dir, val_ann_file, output_type='heatmap', batch_size=32, shuffle=False, heatmap_size=heatmap_size, sigma=sigma)
    rg_train = get_dataloader(train_image_dir, train_ann_file, output_type='regression', batch_size=32, shuffle=True)
    rg_val = get_dataloader(val_image_dir, val_ann_file, output_type='regression', batch_size=32, shuffle=False)

    # models
    hm_model = create_heatmap_model(num_keypoints=5, use_skip_connections=use_skip)
    rg_model = create_regression_model(num_keypoints=5)

    # train (short)
    train_heatmap_model(hm_model, hm_train, hm_val, device, num_epochs=num_epochs, snapshot_epochs=())
    train_regression_model(rg_model, rg_train, rg_val, device, num_epochs=num_epochs)

    # eval
    hm_eval = evaluate_model(hm_model, hm_val, device, 'heatmap')
    rg_eval = evaluate_model(rg_model, rg_val, device, 'regression')

    tag = f'h{heatmap_size}_s{sigma}_skip{int(use_skip)}'
    out_json = f'results/ablation/exp_{tag}.json'
    with open(out_json, 'w') as f:
        json.dump({'heatmap': hm_eval, 'regression': rg_eval, 'params': {'heatmap_size': heatmap_size, 'sigma': sigma, 'use_skip': use_skip}}, f, indent=2)
    # plot pck
    thresholds = [0.05, 0.1, 0.15, 0.2]
    plot_pck_curves(hm_eval['pck_results'], rg_eval['pck_results'], thresholds, f'results/ablation/pck_{tag}.png')
    return hm_eval, rg_eval


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Experiments: vary sigma and skip connections
    # Note: All experiments use 64x64 heatmaps to match model architecture
    configs = [
        {'heatmap_size': 64, 'sigma': 1.0, 'use_skip': True},
        {'heatmap_size': 64, 'sigma': 2.0, 'use_skip': True},
        {'heatmap_size': 64, 'sigma': 3.0, 'use_skip': True},
        {'heatmap_size': 64, 'sigma': 4.0, 'use_skip': True},
        {'heatmap_size': 64, 'sigma': 2.0, 'use_skip': False},
        # {'heatmap_size': 32, 'sigma': 2.0, 'use_skip': True},  # Requires model architecture changes
    ]
    results = []
    for cfg in configs:
        print(f"Running ablation: {cfg}")
        hm_eval, rg_eval = run_experiment(device=device, num_epochs=5, **cfg)
        results.append({'config': cfg, 'heatmap': hm_eval, 'regression': rg_eval})
    with open('results/ablation/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Ablation complete. Artifacts in results/ablation')


if __name__ == '__main__':
    main()
