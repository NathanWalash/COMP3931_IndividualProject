import numpy as np
import torch


class RunConditionedCorrectiveSurrogate(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, depth=4, correction_scale=0.4, use_learned_gate=False, gate_init_bias=-2.0):
        super().__init__()
        # correction_scale controls additive correction magnitude.
        # j_base + scale * tanh(raw) * |j_base| gives a bounded fractional delta.
        self.correction_scale = float(correction_scale)
        self.use_learned_gate = bool(use_learned_gate)

        # Flux head predicts a smooth correction over a time basis.
        flux_basis_count = 48
        flux_layers = []
        flux_in_dim = int(feature_dim)
        flux_index = 0
        while flux_index < int(depth):
            flux_layers.append(torch.nn.Linear(flux_in_dim, int(hidden_dim)))
            flux_layers.append(torch.nn.Tanh())
            flux_in_dim = int(hidden_dim)
            flux_index += 1
        self.j_hidden = torch.nn.Sequential(*flux_layers)
        self.j_coeff_head = torch.nn.Linear(flux_in_dim, flux_basis_count)
        torch.nn.init.zeros_(self.j_coeff_head.weight)
        torch.nn.init.zeros_(self.j_coeff_head.bias)
        self.j_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("flux_basis_centers", torch.linspace(0.0, 1.0, flux_basis_count))
        self.flux_basis_sigma = 1.5 / float(max(1, flux_basis_count - 1))

        # Learned gate: predicts per-sample lambda(x) in [0,1] that controls
        # how much to trust the corrective branch vs the backbone.
        # lambda=0 means pure backbone, lambda=1 means full corrective branch.
        if self.use_learned_gate:
            gate_layers = []
            gate_in_dim = int(feature_dim)
            gate_depth = max(2, int(depth) - 1)
            gate_hidden = int(hidden_dim) // 2
            gate_idx = 0
            while gate_idx < gate_depth:
                gate_layers.append(torch.nn.Linear(gate_in_dim, gate_hidden))
                gate_layers.append(torch.nn.Tanh())
                gate_in_dim = gate_hidden
                gate_idx += 1
            self.gate_hidden = torch.nn.Sequential(*gate_layers)
            self.gate_head = torch.nn.Linear(gate_in_dim, 1)
            # Initialize gate bias to control initial trust level.
            # -2.0 gives sigmoid~0.12 (trust backbone), 0.0 gives sigmoid=0.5 (balanced).
            torch.nn.init.zeros_(self.gate_head.weight)
            torch.nn.init.constant_(self.gate_head.bias, float(gate_init_bias))

    def forward_gate(self, x_feat):
        # Predict per-sample gate lambda in [0, 1].
        # lambda=0 means pure backbone, lambda=1 means full corrective branch.
        if not self.use_learned_gate:
            return None
        gate_hidden = self.gate_hidden(x_feat)
        gate_logit = self.gate_head(gate_hidden)
        return torch.sigmoid(gate_logit)

    def forward_j(self, x_feat, t_star, j_base=None, correction_scale_override=None):
        hidden = self.j_hidden(x_feat)
        coeff = self.j_coeff_head(hidden)
        t_clamped = torch.clamp(t_star, min=0.0, max=1.0)
        centers = self.flux_basis_centers.to(dtype=t_clamped.dtype, device=t_clamped.device).reshape(1, -1)
        z = (t_clamped - centers) / float(self.flux_basis_sigma)
        basis = torch.exp(-0.5 * (z * z))
        raw = torch.sum(coeff * basis, dim=1, keepdim=True) + self.j_bias
        if j_base is None:
            # Standalone flux branch is only used for debugging or ablations.
            return torch.nn.functional.softplus(raw) * 1e-9
        # Use override scale if provided (for progressive annealing), else class default.
        active_scale = float(correction_scale_override) if correction_scale_override is not None else self.correction_scale
        # Additive-only correction path: delta scales with local backbone
        # magnitude so the correction is scale-appropriate at each time point.
        # Detach j_base from the scaling so gradients only flow through raw.
        scale_ref = torch.abs(j_base).detach() + 1e-15
        delta = active_scale * torch.tanh(raw) * scale_ref
        j_corrected = torch.clamp(j_base + delta, min=0.0)
        # If learned gate is active, blend backbone and corrected prediction.
        if self.use_learned_gate:
            # Gate is computed per-run, but we may have time-expanded points.
            gate_lambda = self.forward_gate(x_feat)
            j_corrected = (1.0 - gate_lambda) * j_base + gate_lambda * j_corrected
            j_corrected = torch.clamp(j_corrected, min=0.0)
        return j_corrected

    def forward(self, *args, **kwargs):
        raise RuntimeError("RunConditionedCorrectiveSurrogate uses forward_j for flux correction.")


def curriculum_scale(epoch_index, warmup_epochs, ramp_epochs):
    # Increase physics-loss influence gradually after warmup.
    if int(epoch_index) <= int(warmup_epochs):
        return 0.0
    ramp = max(1, int(ramp_epochs))
    progress = float(int(epoch_index) - int(warmup_epochs)) / float(ramp)
    return float(min(1.0, max(0.0, progress)))


def sample_time_ids(signal_per_time, sample_count):
    # Bias sampling toward informative times where concentration/flux is larger.
    weights = torch.clamp(signal_per_time, min=0.0) + 1e-8
    weights = weights / torch.sum(weights)
    return torch.multinomial(weights, num_samples=int(sample_count), replacement=True)


def predict_flux_curves(model, split_torch, batch_runs=16):
    # Predict full J(t) curves in batches to keep GPU memory bounded.
    model.eval()
    n_rows = int(split_torch["x_feat"].shape[0])
    t_count = int(split_torch["t_norm"].shape[1])
    out_rows = []

    start = 0
    while start < n_rows:
        end = min(start + int(batch_runs), n_rows)
        row_ids = torch.arange(start, end, device=split_torch["x_feat"].device, dtype=torch.long)
        x_batch = split_torch["x_feat"][row_ids]
        j_base_batch = split_torch["j_base"][row_ids]
        t_batch = split_torch["t_norm"][row_ids]
        local_count = int(end - start)

        run_ids = torch.arange(local_count, device=split_torch["x_feat"].device, dtype=torch.long)
        run_ids = run_ids.repeat_interleave(t_count)
        x_points = x_batch[run_ids]
        j_base_points = j_base_batch.reshape(-1, 1)
        t_points = t_batch.reshape(-1, 1)
        j_points = model.forward_j(x_points, t_points, j_base=j_base_points)
        out_rows.append(j_points.reshape(local_count, t_count).detach().cpu().numpy().astype(np.float32))

        start = end

    return np.concatenate(out_rows, axis=0)
