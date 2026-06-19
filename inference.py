"""Inference script for the demographic-conditioned Mamba2 ECG age estimator."""

import os
import json
import argparse
from warnings import warn

import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import MultimodalDeepMambaECG, HuberLoss
from dataloader import LazyECGDataset
from metrics import calculate_metrics

AGE_SCALE = 100.0


def inference(model, dataload, amp_dtype, device, loss_fn):
    """Run inference on a dataset and collect predictions.

    Args:
        model: Trained model.
        dataload: DataLoader for the evaluation dataset.
        amp_dtype: AMP data type (bfloat16 or float16).
        device: Compute device.
        loss_fn: Loss function for computing evaluation loss.

    Returns:
        metrics: Dictionary of evaluation metrics.
        all_preds: Array of predicted ages (in years).
        all_targets: Array of ground-truth ages (in years).
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    n_entries = 0

    eval_desc = "Inference - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(0), position=0)

    with torch.no_grad():
        for i, batch in enumerate(dataload):
            traces, ages, _ = batch[0], batch[1], batch[2]
            sex = batch[3] if len(batch) > 3 else None

            traces = traces.transpose(1, 2).to(device, dtype=torch.float32)
            ages = ages.to(device, dtype=torch.float32)
            if sex is not None:
                sex = sex.to(device, dtype=torch.long)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred_ages, time_weights, lead_weights = model(traces, sex)
                loss = loss_fn(pred_ages, ages)

            bs = len(traces)
            total_loss += loss.item() * bs
            n_entries += bs

            all_preds.append(pred_ages.flatten().detach().float().cpu().numpy() * AGE_SCALE)
            all_targets.append(ages.detach().float().cpu().numpy() * AGE_SCALE)

            eval_bar.desc = eval_desc.format(total_loss / n_entries)
            eval_bar.update(1)

    eval_bar.close()

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / n_entries

    return metrics, all_preds, all_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained ECG age predictor')
    parser.add_argument('path_to_traces', help='Path to HDF5 file containing ECG traces')
    parser.add_argument('path_to_csv', help='Path to CSV file containing patient metadata')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--folder', default='model/', help='Folder containing model checkpoint')
    parser.add_argument('--traces_dset', default='tracings', help='HDF5 dataset name for traces')
    parser.add_argument('--ids_dset', default='', help='HDF5 dataset name for exam IDs')
    parser.add_argument('--age_col', default='age', help='CSV column for age')
    parser.add_argument('--ids_col', default='exam_id', help='CSV column for exam IDs')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID (e.g., "0")')
    args, unk = parser.parse_known_args()
    if unk:
        warn("Unknown arguments: " + str(unk))

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.folder, exist_ok=True)
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    # Data loading
    print("Building data loader...")
    df = pd.read_csv(args.path_to_csv, index_col=args.ids_col)

    with h5py.File(args.path_to_traces, 'r') as f:
        if args.ids_dset:
            print("Reindexing CSV to match HDF5 order...")
            h5ids = f[args.ids_dset][:]
            df = df.reindex(h5ids, fill_value=False, copy=True)
        total_samples = f[args.traces_dset].shape[0]
        assert len(df) == total_samples, \
            f"CSV and HDF5 lengths do not match! CSV: {len(df)}, HDF5: {total_samples}"

    ages_npy = df[args.age_col].values.astype(np.float32)
    weights_npy = np.zeros_like(ages_npy, dtype=np.float32)

    if 'sex' in df.columns:
        sex_npy = df['sex'].map({'F': 0, 'M': 1, 'female': 0, 'male': 1}).fillna(0).astype(int).values
    elif 'is_male' in df.columns:
        sex_npy = df['is_male'].values.astype(int)
    else:
        sex_npy = None

    # Select one exam per patient if patient_id column exists
    valid_mask = np.ones(len(df), dtype=bool)
    if 'patient_id' in df.columns:
        print("Selecting first exam per patient...")
        _, idx = np.unique(df['patient_id'], return_index=True)
        valid_mask = np.zeros(len(df), dtype=bool)
        valid_mask[idx] = True

    valid_ds = LazyECGDataset(
        h5_path=args.path_to_traces,
        dset_name=args.traces_dset,
        ages=ages_npy,
        weights=weights_npy,
        sex_labels=sex_npy,
        mask=valid_mask,
        age_scale=AGE_SCALE,
    )

    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Inference batches: {len(valid_loader)}")

    # Model
    print("Loading model...")
    model = MultimodalDeepMambaECG(
        num_layers=4, hidden_dim=128, num_leads=12,
        num_sex_classes=2, downsample_factor=8,
    )
    loss_fn = HuberLoss(delta=0.05, loss_scale=200.0).to(device)

    checkpoint = torch.load(os.path.join(args.folder, 'model.pth'), map_location='cpu')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Mixed precision
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Using BFloat16")
    else:
        amp_dtype = torch.float16
        print("Using Float16")

    # Run inference
    print("Running inference...")
    valid_metrics, all_preds, all_targets = inference(model, valid_loader, amp_dtype, device, loss_fn)

    # Save results
    results = pd.DataFrame([{
        "loss": valid_metrics['loss'],
        "mae": valid_metrics['mae'],
        "mae_std": valid_metrics['mae_std'],
        "rmse": valid_metrics['rmse'],
        "r2": valid_metrics['r2'],
        "r": valid_metrics['r'],
        "pred_mean": valid_metrics['pred_mean'],
        "pred_std": valid_metrics['pred_std'],
        "real_mean": valid_metrics['real_mean'],
        "real_std": valid_metrics['real_std'],
    }])
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.folder, 'inference_history.csv'), index=False)

    # Save per-sample predictions
    if len(all_preds) < len(df):
        full_preds = np.full(len(df), np.nan)
        full_targets = np.full(len(df), np.nan)
        full_preds[valid_mask] = all_preds
        full_targets[valid_mask] = all_targets
        all_preds = full_preds
        all_targets = full_targets

    df['predicted_age'] = all_preds
    df['target_age'] = all_targets
    df.to_csv(os.path.join(args.folder, 'inference_predictions.csv'))

    print("Done!")
