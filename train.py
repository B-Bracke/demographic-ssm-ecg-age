"""Training script for the demographic-conditioned Mamba2 ECG age estimator."""

import os
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from model import MultimodalDeepMambaECG, HuberLoss
from dataloader import LazyECGDataset
from metrics import calculate_metrics

AGE_SCALE = 100.0


def get_binned_weights(train_ages, num_bins=20, max_age=100):
    """Compute inverse-frequency sample weights based on age bins.

    Args:
        train_ages: Array of training ages.
        num_bins: Number of histogram bins.
        max_age: Upper age bound for binning.

    Returns:
        bin_weights: Normalized weight per bin.
        bin_edges: Bin edge values.
    """
    hist, bin_edges = np.histogram(train_ages, bins=num_bins, range=(0, max_age))
    bin_weights = 1.0 / (hist + 1e-6)
    bin_weights[hist == 0] = 0.0
    valid_weights = bin_weights[bin_weights > 0]
    bin_weights = bin_weights / valid_weights.mean()
    return bin_weights, bin_edges


def get_grouped_params(model, weight_decay):
    """Separate model parameters into weight-decay and no-decay groups.

    Biases, normalization layers, and embedding weights are excluded from
    weight decay to prevent regularization interference.

    Args:
        model: The neural network model.
        weight_decay: Weight decay value for non-exempt parameters.

    Returns:
        List of parameter group dictionaries for the optimizer.
    """
    decay_params = []
    no_decay_params = []

    blacklist_weight_modules = (
        nn.LayerNorm, nn.BatchNorm1d, nn.InstanceNorm1d,
        nn.GroupNorm, nn.Embedding,
    )
    no_decay_ids = set()

    for module_name, module in model.named_modules():
        if isinstance(module, blacklist_weight_modules):
            for param in module.parameters():
                no_decay_ids.add(id(param))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name:
            no_decay_params.append(param)
        elif id(param) in no_decay_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def train_epoch(ep, dataload, model, optimizer, scaler, amp_dtype,
                device, loss_fn, scheduler, accum_iter):
    """Run one training epoch.

    Args:
        ep: Current epoch number.
        dataload: Training DataLoader.
        model: Model.
        optimizer: Optimizer instance.
        scaler: Gradient scaler for mixed precision.
        amp_dtype: AMP data type (bfloat16 or float16).
        device: Compute device.
        loss_fn: Loss function.
        scheduler: Learning rate scheduler.
        accum_iter: Gradient accumulation steps.

    Returns:
        Dictionary of training metrics.
    """
    model.train()

    all_preds, all_targets = [], []
    total_loss = 0
    n_entries = 0

    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0), position=0)

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(dataload):
        traces, ages, weights = batch[0], batch[1], batch[2]
        sex = batch[3] if len(batch) > 3 else None

        traces = traces.transpose(1, 2).to(device, dtype=torch.float32, non_blocking=True)
        ages = ages.to(device, dtype=torch.float32, non_blocking=True)
        weights = weights.to(device, dtype=torch.float32, non_blocking=True)
        if sex is not None:
            sex = sex.to(device, dtype=torch.long, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out_aug, _, _ = model(
                traces, sex, targets=None,
                do_mixup=False,
                do_time_masking=True,
                do_channel_masking=True,
                do_noise=True,
                do_gamma=True,
                do_random_filter=True,
                aug_intensity=0.5,
            )
            loss = loss_fn(out_aug, ages, weights) / accum_iter

        scaler.scale(loss).backward()

        if (i + 1) % accum_iter == 0 or (i + 1) == len(dataload):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        bs = len(traces)
        total_loss += (loss.item() * accum_iter) * bs
        n_entries += bs

        pred_ages = out_aug.flatten().detach().float()
        target_ages = ages.flatten().detach().float()
        all_preds.append(pred_ages.cpu().numpy() * AGE_SCALE)
        all_targets.append(target_ages.cpu().numpy() * AGE_SCALE)

        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)

    train_bar.close()

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / n_entries
    return metrics


def evaluate(ep, dataload, model, amp_dtype, device, loss_fn):
    """Run one evaluation epoch.

    Args:
        ep: Current epoch number.
        dataload: Validation DataLoader.
        model: Model to evaluate.
        amp_dtype: AMP data type.
        device: Compute device.
        loss_fn: Loss function.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    n_entries = 0

    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0), position=0)

    with torch.no_grad():
        for i, batch in enumerate(dataload):
            traces, ages, weights = batch[0], batch[1], batch[2]
            sex = batch[3] if len(batch) > 3 else None

            traces = traces.transpose(1, 2).to(device, dtype=torch.float32, non_blocking=True)
            ages = ages.to(device, dtype=torch.float32, non_blocking=True)
            if sex is not None:
                sex = sex.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                out, _, _ = model(traces, sex, targets=None, aug_intensity=0)
                loss = loss_fn(out, ages)

            bs = len(traces)
            total_loss += loss.item() * bs
            n_entries += bs

            all_preds.append(out.flatten().float().cpu().numpy() * AGE_SCALE)
            all_targets.append(ages.flatten().float().cpu().numpy() * AGE_SCALE)

            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)

    eval_bar.close()

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / n_entries
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multimodal ECG Age Predictor')
    parser.add_argument('path_to_traces', help='Path to HDF5 file containing ECG traces')
    parser.add_argument('path_to_csv', help='Path to CSV file containing patient metadata')
    parser.add_argument('--traces_dset', default='tracings', help='HDF5 dataset name for traces')
    parser.add_argument('--ids_dset', default='', help='HDF5 dataset name for exam IDs')
    parser.add_argument('--ids_col', default='id_exam', help='CSV column for exam IDs')
    parser.add_argument('--age_col', default='age', help='CSV column for age')
    parser.add_argument('--folder', default='model_output/', help='Output folder for checkpoints and logs')
    parser.add_argument('--n_valid', type=int, default=500, help='Number of validation samples')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--accum_iter', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4, help='Number of BiMamba2 blocks')
    parser.add_argument('--downsample', type=int, default=8, help='Temporal downsampling factor')
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--time_mask_prob', type=float, default=0.5)
    parser.add_argument('--channel_mask_prob', type=float, default=0.5)
    parser.add_argument('--gpu', type=str, default=None, help='GPU device ID (e.g., "0")')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.folder, exist_ok=True)
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    # Data loading
    print("Loading metadata...")
    df = pd.read_csv(args.path_to_csv, index_col=args.ids_col)

    if args.ids_dset:
        with h5py.File(args.path_to_traces, 'r') as f:
            h5ids = f[args.ids_dset][:]
            df = df.reindex(h5ids, fill_value=False, copy=True)

    ages_npy = df[args.age_col].values.astype(np.float32)
    bin_weights, bin_edges = get_binned_weights(ages_npy, num_bins=20)
    patient_bin_indices = np.clip(np.digitize(ages_npy, bin_edges) - 1, 0, len(bin_weights) - 1)
    weights_npy = bin_weights[patient_bin_indices].astype(np.float32)

    if 'sex' in df.columns:
        sex_map = {'F': 0, 'M': 1, 'female': 0, 'male': 1}
        sex_npy = df['sex'].map(sex_map).fillna(0).astype(int).values
    else:
        sex_npy = np.zeros_like(ages_npy, dtype=int)

    n_valid = args.n_valid
    valid_mask = np.zeros(len(df), dtype=bool)
    valid_mask[:n_valid] = True
    train_mask = ~valid_mask
    print(f"Split: {len(df) - n_valid} train, {n_valid} valid")

    train_ds = LazyECGDataset(args.path_to_traces, args.traces_dset, ages_npy, weights_npy, sex_npy, train_mask, AGE_SCALE)
    valid_ds = LazyECGDataset(args.path_to_traces, args.traces_dset, ages_npy, weights_npy, sex_npy, valid_mask, AGE_SCALE)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    print("Initializing model...")
    model = MultimodalDeepMambaECG(
        num_layers=args.layers,
        hidden_dim=args.hidden_dim,
        num_leads=12,
        downsample_factor=args.downsample,
        num_sex_classes=2,
        mixup_alpha=args.mixup_alpha,
        time_mask_prob=args.time_mask_prob,
        channel_mask_prob=args.channel_mask_prob,
        use_mixstyle=False,
    ).to(device)

    loss_fn = HuberLoss(delta=0.05, loss_scale=200.0, weight_power=1.0).to(device)
    grouped_params = get_grouped_params(model, weight_decay=0.05)
    optimizer = AdamW(grouped_params, lr=args.lr, betas=(0.9, 0.95))

    # Learning rate schedule
    optimization_steps_per_epoch = len(train_loader) // args.accum_iter
    total_steps = args.epochs * optimization_steps_per_epoch
    warmup_steps = int(0.01 * total_steps)
    decay_steps = total_steps - warmup_steps

    print(f"Physical batch: {args.batch_size} | Accumulation: {args.accum_iter} | "
          f"Effective batch: {args.batch_size * args.accum_iter}")
    print(f"Total optimization steps: {total_steps} (warmup: {warmup_steps})")

    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    # Mixed precision setup
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        print("Using BFloat16")
    else:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        print("Using Float16")

    # Resume from checkpoint
    start_epoch = 0
    best_loss = np.inf
    history = pd.DataFrame()

    if args.resume:
        ckpt_path = os.path.join(args.folder, 'model.pth')
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])

            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('valid_loss', np.inf)

            hist_path = os.path.join(args.folder, 'history.csv')
            if os.path.exists(hist_path):
                history = pd.read_csv(hist_path)
                history = history[history['epoch'] < start_epoch]

        # Re-create and fast-forward scheduler
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        scheduler_decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

        resumed_step = start_epoch * optimization_steps_per_epoch
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(resumed_step):
                scheduler.step()
        print(f"Scheduler fast-forwarded to step {resumed_step}. Current LR: {scheduler.get_last_lr()[0]:.2e}")

    # Training loop
    print("Starting training...")
    for ep in range(start_epoch, args.epochs):
        train_m = train_epoch(ep, train_loader, model, optimizer, scaler,
                              amp_dtype, device, loss_fn, scheduler, args.accum_iter)
        valid_m = evaluate(ep, valid_loader, model, amp_dtype, device, loss_fn)

        # Save checkpoint
        ckpt_data = {
            'epoch': ep,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'valid_loss': valid_m['loss'],
        }

        if valid_m['loss'] < best_loss:
            best_loss = valid_m['loss']
            torch.save(ckpt_data, os.path.join(args.folder, 'best.pth'))
            print(f"New best model saved with loss {best_loss:.6f} at epoch {ep}")
        else:
            torch.save(ckpt_data, os.path.join(args.folder, 'model.pth'))

        # Log metrics
        row = {'epoch': ep, 'lr': optimizer.param_groups[0]['lr']}
        row.update({f'train_{k}': v for k, v in train_m.items()})
        row.update({f'valid_{k}': v for k, v in valid_m.items()})

        epoch_df = pd.DataFrame([row])
        history = pd.concat([history, epoch_df], ignore_index=True)
        history.to_csv(os.path.join(args.folder, 'history.csv'), index=False)

        print(f"Ep {ep} | T_Loss: {train_m['loss']:.4f} | V_Loss: {valid_m['loss']:.4f} | "
              f"V_MAE: {valid_m['mae']:.2f}")