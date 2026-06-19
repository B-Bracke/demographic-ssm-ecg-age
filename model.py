"""Model architecture for demographic-conditioned bidirectional Mamba2 ECG age estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2

from augmentations import apply_time_masking, get_channel_mask


class MixStyle(nn.Module):
    """Feature-level style mixing for domain generalization.

    Randomly interpolates instance-level feature statistics (mean and variance)
    between samples within a batch, controlled by a Beta distribution.

    Args:
        p: Probability of applying style mixing per forward pass.
        alpha: Parameter for the Beta distribution controlling mixing intensity.
    """

    def __init__(self, p=1.0, alpha=0.1):
        super().__init__()
        self.p = p
        self.base_alpha = alpha
        self.current_alpha = 0.1

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
        if self.current_alpha < 1e-6:
            return x

        B = x.size(0)
        beta_dist = torch.distributions.Beta(self.current_alpha, self.current_alpha)

        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        sig = (var + 1e-6).sqrt()

        perm = torch.randperm(B)
        mu2 = mu[perm]
        sig2 = sig[perm]

        lam = beta_dist.sample((B, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2

        return ((x - mu) / sig) * sig_mix + mu_mix


class MaskedAttentionPooling(nn.Module):
    """Learnable attention-based pooling with optional context conditioning.

    Computes weighted sums over a sequence dimension using learned attention scores.
    Supports padding masks and an optional context vector for conditional scoring.

    Args:
        input_dim: Dimensionality of the input features.
        initial_temperature: Initial temperature for attention score scaling.
        use_sigmoid: If True, use sigmoid + normalization instead of softmax.
    """

    def __init__(self, input_dim, initial_temperature=2.5, use_sigmoid=True):
        super().__init__()
        self.attention_projection = nn.Linear(input_dim, 1)
        self.context_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.log_temp = nn.Parameter(torch.tensor(initial_temperature).log())
        self.use_sigmoid = use_sigmoid
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.xavier_uniform_(self.attention_projection.weight, gain=0.01)
        nn.init.constant_(self.attention_projection.bias, 0.0)

    def forward(self, x, padding_mask=None, context=None):
        if context is not None:
            ctx_proj = self.context_projection(context)
            ctx_expanded = ctx_proj.unsqueeze(1)
            features_for_scoring = self.tanh(x + ctx_expanded)
        else:
            features_for_scoring = x

        scores = self.attention_projection(features_for_scoring)
        clamped_log_temp = torch.clamp(self.log_temp, min=-5, max=5.0)
        scores = scores / torch.exp(clamped_log_temp)

        if self.use_sigmoid:
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1)
                scores = scores.masked_fill(mask_expanded == 0, -1e9)
            weights = torch.sigmoid(scores)
            normalization = torch.sum(weights, dim=1, keepdim=True) + 1e-6
            weights = weights / normalization
        else:
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1)
                scores = scores.masked_fill(mask_expanded == 0, -1e9)
            weights = F.softmax(scores, dim=1)

        weighted_embeddings = torch.sum(x * weights, dim=1)
        return weighted_embeddings, weights


def create_padding_mask(x):
    """Creates a binary mask indicating non-padded time steps (energy > threshold)."""
    energy = torch.abs(x).sum(dim=1)
    return (energy > 1e-6).float()


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth per-sample drop path."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class StochasticDepth(nn.Module):
    """Stochastic depth regularization module."""

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class AdaLNModulator(nn.Module):
    """Adaptive Layer Normalization modulator for conditioning.

    Generates shift and scale parameters from a conditioning embedding to
    modulate layer-normalized features (used in DiT-style conditioning).

    Args:
        d_model: Dimensionality of the conditioning embedding and output modulations.
    """

    def __init__(self, d_model):
        super().__init__()
        self.si_lu = nn.SiLU()
        self.linear = nn.Linear(d_model, 4 * d_model, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, condition_embedding):
        out = self.linear(self.si_lu(condition_embedding))
        return out.chunk(4, dim=1)


class BiMamba2Block(nn.Module):
    """Bidirectional Mamba2 block with adaptive layer normalization.

    Processes input sequences in both forward and reverse directions using
    separate Mamba2 modules, then combines the outputs. Conditioning is
    applied via AdaLN modulation on both the SSM and FFN sub-layers.

    Args:
        d_model: Model dimensionality.
        d_state: SSM state dimensionality.
        headdim: Head dimension for Mamba2.
        expand: Expansion factor for the internal Mamba2 projection.
        drop_path: Drop path rate for stochastic depth.
    """

    def __init__(self, d_model, d_state=64, headdim=64, expand=2, drop_path=0.0):
        super().__init__()
        self.forward_mamba = Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand)
        self.backward_mamba = Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ada_ln_modulator = AdaLNModulator(d_model)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, condition_embedding):
        shift_m, scale_m, shift_f, scale_f = self.ada_ln_modulator(condition_embedding)
        shift_m, scale_m = shift_m.unsqueeze(1), scale_m.unsqueeze(1)
        shift_f, scale_f = shift_f.unsqueeze(1), scale_f.unsqueeze(1)

        residual = x
        x_norm = self.norm(x) * (1 + scale_m) + shift_m
        out_fwd = self.forward_mamba(x_norm)
        out_rev = torch.flip(self.backward_mamba(torch.flip(x_norm, dims=[1])), dims=[1])
        x = residual + self.drop_path(out_fwd + out_rev)

        residual = x
        x_norm = self.norm_ffn(x) * (1 + scale_f) + shift_f
        x = residual + self.drop_path(self.ffn(x_norm))
        return x


class MultimodalDeepMambaECG(nn.Module):
    """Demographic-conditioned bidirectional Mamba2 model for ECG-based age estimation.

    Architecture overview:
        1. Per-lead convolutional stem with optional MixStyle
        2. Learnable lead-specific positional embeddings
        3. Stacked BiMamba2 blocks with AdaLN demographic conditioning
        4. Hierarchical attention pooling (time → lead)
        5. Regression head predicting normalized age

    Args:
        num_layers: Number of BiMamba2 blocks.
        hidden_dim: Hidden dimensionality throughout the model.
        num_leads: Number of ECG leads (default: 12).
        dropout: Dropout rate in the prediction head.
        drop_path_rate: Maximum stochastic depth rate (linearly increased).
        downsample_factor: Temporal downsampling factor in the convolutional stem.
        num_sex_classes: Number of sex categories for the demographic embedding.
        mixup_alpha: Alpha parameter for mixup augmentation.
        time_mask_prob: Probability of applying time masking augmentation.
        channel_mask_prob: Probability of applying channel masking augmentation.
        use_mixstyle: Whether to apply MixStyle after the stem.
    """

    def __init__(self, num_layers=4, hidden_dim=128, num_leads=12, dropout=0.2, drop_path_rate=0.2,
                 downsample_factor=8, num_sex_classes=2, mixup_alpha=0.1,
                 time_mask_prob=1.0, channel_mask_prob=1.0, use_mixstyle=True):
        super().__init__()
        self.num_leads = num_leads
        self.downsample_factor = downsample_factor
        self.mixup_alpha = mixup_alpha
        self.time_mask_prob = time_mask_prob
        self.channel_mask_prob = channel_mask_prob
        self.use_mixstyle = use_mixstyle

        self.sex_embedding = nn.Embedding(num_sex_classes, hidden_dim)
        self.lead_embedding = nn.Parameter(torch.randn(1, num_leads, 1, hidden_dim) * 0.02)

        # Convolutional stem
        layers = []
        stride1 = 2 if downsample_factor >= 2 else 1
        layers.append(nn.Conv1d(1, hidden_dim // 2, kernel_size=7, stride=stride1, padding=3))
        layers.append(nn.InstanceNorm1d(hidden_dim // 2, affine=True))
        layers.append(nn.GELU())
        stride2 = downsample_factor // stride1
        layers.append(nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=stride2, padding=2))
        layers.append(nn.InstanceNorm1d(hidden_dim, affine=True))
        layers.append(nn.GELU())
        self.stem = nn.Sequential(*layers)

        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        self.backbone_norm = nn.LayerNorm(hidden_dim)

        # Backbone
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            BiMamba2Block(d_model=hidden_dim, drop_path=dpr[i]) for i in range(num_layers)
        ])

        # Hierarchical attention pooling
        self.time_pooler = MaskedAttentionPooling(hidden_dim, initial_temperature=2.5, use_sigmoid=True)
        self.lead_pooler = MaskedAttentionPooling(hidden_dim, initial_temperature=2.5, use_sigmoid=True)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.head_mu = nn.Linear(hidden_dim // 2, 1)

        self.apply(self._init_weights)
        self.reset_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def reset_parameters(self):
        """Reset parameters with task-specific initialization."""
        self.apply(self._init_weights)
        nn.init.normal_(self.sex_embedding.weight, std=0.02)

        # AdaLN zero-initialization for identity start
        for layer in self.layers:
            nn.init.constant_(layer.ada_ln_modulator.linear.weight, 0)
            nn.init.constant_(layer.ada_ln_modulator.linear.bias, 0)

        self.time_pooler.reset_parameters()
        self.lead_pooler.reset_parameters()

        # Bias initialization to dataset mean age (normalized)
        target_mean = 0.54
        self.head_mu.bias.data.fill_(target_mean)

    def _generate_mixup_context(self, batch_size, device):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(device)
        return {'lam': lam, 'index': index}

    def _mix_tensor(self, tensor, ctx):
        if tensor is None:
            return None
        return ctx['lam'] * tensor + (1 - ctx['lam']) * tensor[ctx['index']]

    def forward(self, x, sex, targets=None, time_mask=None, channel_mask=None,
                do_mixup=True, do_time_masking=True, do_channel_masking=True,
                do_gamma=True, do_noise=True, do_random_filter=True, aug_intensity=0.5):
        """Forward pass with optional training-time augmentations.

        Args:
            x: ECG tensor of shape [B, leads, seq_len].
            sex: Sex labels of shape [B] (0=female, 1=male).
            targets: Age targets of shape [B] (only used for mixup during training).
            time_mask: Optional precomputed padding mask.
            channel_mask: Optional precomputed channel mask.
            do_mixup: Enable mixup augmentation.
            do_time_masking: Enable time masking augmentation.
            do_channel_masking: Enable channel dropout augmentation.
            do_gamma: Enable gamma augmentation.
            do_noise: Enable additive noise augmentation.
            do_random_filter: Enable random convolution filter augmentation.
            aug_intensity: Scaling factor for augmentation probabilities.

        Returns:
            prediction: Predicted normalized age [B, 1].
            time_weights: Temporal attention weights.
            lead_weights: Lead attention weights.
            targets: (Only during training with mixup) Mixed targets.
        """
        batch_size, num_leads, seq_len = x.shape
        device = x.device

        if time_mask is None:
            time_mask = create_padding_mask(x)

        # Instance normalization
        with torch.no_grad():
            mu = x.mean(dim=2, keepdim=True)
            sigma = x.std(dim=2, keepdim=True) + 1e-6
            x = (x - mu) / sigma
            mask_broad = time_mask.unsqueeze(1)
            x = x * mask_broad

        current_time_prob = 0.5
        current_channel_prob = 0.5
        current_noise_scale = 0.05

        # Training-time augmentations
        if self.training:
            if do_random_filter and torch.rand(1).item() < aug_intensity:
                k_size = 7
                rand_kernel = torch.randn(num_leads, 1, k_size, device=device)
                rand_kernel = rand_kernel / (rand_kernel.abs().sum(dim=2, keepdim=True) + 1e-6)
                x = F.conv1d(x, rand_kernel, padding=k_size // 2, groups=num_leads)
                x = x * mask_broad
            if do_gamma and torch.rand(1).item() < aug_intensity:
                gamma = (torch.rand(x.shape[0], 1, 1, device=device) * 0.2) + 0.9
                x = x.sign() * (x.abs() ** gamma)
            if do_noise:
                noise = torch.randn_like(x) * current_noise_scale
                x = x + noise
                x = x * mask_broad
            if do_time_masking:
                x = apply_time_masking(x, prob=current_time_prob, max_mask_ratio=0.2, num_masks=2)

        if do_channel_masking and self.training:
            channel_mask = get_channel_mask(x, drop_prob=current_channel_prob, max_drop=11)
        else:
            channel_mask = torch.ones((batch_size, num_leads), device=device)

        # Convolutional stem
        x_reshaped = x.reshape(batch_size * num_leads, 1, seq_len)
        x_features = self.stem(x_reshaped)
        x_features = x_features.permute(0, 2, 1)
        if self.use_mixstyle and self.training:
            x_features = self.mixstyle(x_features)
        x_features = self.backbone_norm(x_features)

        # Add lead embeddings
        x_features = x_features.reshape(batch_size, num_leads, -1, x_features.shape[-1])
        x_features = x_features + self.lead_embedding

        # Mixup
        if do_mixup and self.training:
            ctx = self._generate_mixup_context(batch_size, device)
            x_features = self._mix_tensor(x_features, ctx)
            targets = self._mix_tensor(targets, ctx)
            emb_a = self.sex_embedding(sex)
            emb_b = self.sex_embedding(sex[ctx['index']])
            sex_vector = ctx['lam'] * emb_a + (1 - ctx['lam']) * emb_b
        else:
            sex_vector = self.sex_embedding(sex)

        # Backbone
        x_features = x_features.reshape(batch_size * num_leads, -1, x_features.shape[-1])
        sex_cond_expanded = sex_vector.repeat_interleave(num_leads, dim=0)

        for layer in self.layers:
            x_features = layer(x_features, sex_cond_expanded)

        # Downsample padding mask
        if time_mask is not None:
            mask_reshaped = time_mask.unsqueeze(1)
            scale = 1.0 / self.downsample_factor
            mask_ds = F.interpolate(mask_reshaped, scale_factor=scale, mode='nearest').squeeze(1)
            mask_final = mask_ds.repeat_interleave(num_leads, dim=0)
        else:
            mask_final = None

        # Hierarchical pooling: time → lead
        lead_vectors, time_weights = self.time_pooler(x_features, padding_mask=mask_final, context=sex_cond_expanded)
        lead_vectors = lead_vectors.reshape(batch_size, num_leads, -1)
        time_weights = time_weights.reshape(batch_size, num_leads, -1)
        patient_vector, lead_weights = self.lead_pooler(lead_vectors, padding_mask=channel_mask, context=sex_vector)

        # Prediction
        final_vector = torch.cat([patient_vector, sex_vector], dim=1)
        x_head = self.head(final_vector)
        prediction = torch.sigmoid(self.head_mu(x_head)) # sigmoid to constrain output to [0,1] 0-100 years range after scaling

        if do_mixup and self.training:
            return prediction, time_weights, lead_weights, targets
        else:
            return prediction, time_weights, lead_weights


class HuberLoss(nn.Module):
    """Scaled Huber loss with optional per-sample weighting.

    Applies a scaling factor to counteract small gradient magnitudes when
    operating on normalized (0-1) targets, and supports label weights.

    Args:
        delta: Huber loss delta threshold.
        loss_scale: Multiplicative scaling factor for the loss.
        weight_power: Exponent applied to per-sample weights.
    """

    def __init__(self, delta=0.05, loss_scale=200.0, weight_power=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        self.loss_scale = loss_scale
        self.weight_power = weight_power

    def forward(self, pred, target, weights=None):
        if pred.dim() > 1:
            pred = pred.squeeze()
        if target.dim() > 1:
            target = target.squeeze()

        loss = self.huber(pred, target)
        loss = loss * self.loss_scale

        if weights is not None:
            if weights.dim() > 1:
                weights = weights.squeeze()
            if self.weight_power != 1.0:
                weights = weights.pow(self.weight_power)
            weights = weights / (weights.mean() + 1e-6)
            loss = loss * weights

        return loss.mean()
