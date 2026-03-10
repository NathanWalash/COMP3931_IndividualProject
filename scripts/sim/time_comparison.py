import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ml.blackbox_data import load_npz
from scripts.ml.common import load_json, resolve_split_key
from scripts.ml.pinn_model import RunConditionedPINN, predict_flux_curves
from skin_diffusion.config import load_config
from skin_diffusion.ml_curve_backbone import fit_curve_backbone, predict_curve_backbone
from skin_diffusion.run_utils import run_simulation
from skin_diffusion.utils import ensure_dir


def load_ml_splits(ml_dir):
    # Load train/test split arrays using split names from ml/meta.json.
    ml_path = Path(ml_dir)
    meta = load_json(ml_path / "meta.json")
    split_map = meta["splits"]
    train_key = resolve_split_key(split_map, "train")
    test_key = resolve_split_key(split_map, "test")
    train = load_npz(ml_path / (train_key + ".npz"))
    test = load_npz(ml_path / (test_key + ".npz"))
    return train, test


def time_single_simulation(config_path, n_repeats=5):
    # Time one full PDE simulation pipeline per repeat, averaged over n_repeats.
    cfg = load_config(config_path)

    timings = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        # Use the same entrypoint used by dataset generation/truth evaluation.
        run_simulation(cfg)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return {
        "mean_seconds": float(np.mean(timings)),
        "std_seconds": float(np.std(timings)),
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings)),
        "n_repeats": n_repeats,
        "grid": f"{cfg.grid.H}x{cfg.grid.W}",
        "T": cfg.grid.T,
        "dt": cfg.grid.dt,
        "n_steps": int(np.ceil(cfg.grid.T / cfg.grid.dt)) + 1,
    }


def time_blackbox_inference(ml_dir):
    # Time blackbox surrogate training + test-set inference.
    train_data, test_data = load_ml_splits(ml_dir)

    X_train = train_data["X"]
    J_train = train_data["J"]
    X_test = test_data["X"]
    J_test = test_data["J"]
    n_test = X_test.shape[0]

    # Time training.
    t0 = time.perf_counter()
    backbone = fit_curve_backbone(
        X_train, J_train, X_test, J_test,
        curve_components=20, use_xgboost=False,
    )
    t_train = time.perf_counter() - t0

    # Time inference (multiple repeats for stability).
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        predict_curve_backbone(backbone, X_test)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return {
        "train_seconds": float(t_train),
        "inference_mean_seconds": float(np.mean(timings)),
        "inference_std_seconds": float(np.std(timings)),
        "inference_per_sample_ms": float(np.mean(timings) / n_test * 1000),
        "n_test": n_test,
        "n_train": int(X_train.shape[0]),
    }


def time_pinn_inference(ml_dir, model_path, device="cpu"):
    # Time end-to-end stageFinal PINN inference on the test set.
    torch_device = torch.device(device)

    model_dir = Path(model_path).parent

    # Read model config from summary.json next to the checkpoint.
    summary_path = model_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    pcfg = summary["pinn_config"]
    flux_head_mode = pcfg.get("flux_head_mode", "corrective")

    # Load test features + build backbone for j_base.
    train_data, test_data = load_ml_splits(ml_dir)
    n_test = test_data["X"].shape[0]
    feat_dim = test_data["X"].shape[1]
    X_test = test_data["X"]

    # Rebuild backbone to get j_base predictions.
    backbone = fit_curve_backbone(
        train_data["X"], train_data["J"],
        X_test, test_data["J"],
        curve_components=20, use_xgboost=False,
    )

    model = RunConditionedPINN(
        feature_dim=feat_dim,
        hidden_dim=pcfg.get("hidden_dim", 128),
        depth=pcfg.get("depth", 4),
        fourier_frequencies=pcfg.get("fourier_frequencies", 6),
        correction_scale=pcfg.get("correction_scale", 0.4),
        additive_correction=pcfg.get("additive_correction", True),
        use_learned_gate=pcfg.get("use_learned_gate", True),
        gate_init_bias=pcfg.get("gate_init_bias", -2.0),
    )
    checkpoint = torch.load(model_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    # Precompute normalized time grid once; this is part of regular inference prep.
    t = test_data["t"]
    t_max = t.max(axis=1, keepdims=True)
    t_max = np.where(t_max > 0, t_max, 1.0)
    t_norm = (t / t_max).astype(np.float32)
    t_norm_t = torch.as_tensor(t_norm, dtype=torch.float32, device=torch_device)

    def run_stagefinal_once():
        # End-to-end stageFinal inference:
        # 1) backbone predict (j_base + x_scaled) 2) PINN correction.
        j_base, x_scaled = predict_curve_backbone(backbone, X_test)
        test_torch = {
            "x_feat": torch.as_tensor(x_scaled, dtype=torch.float32, device=torch_device),
            "j_base": torch.as_tensor(j_base, dtype=torch.float32, device=torch_device),
            "t_norm": t_norm_t,
        }
        return predict_flux_curves(
            model=model,
            split_torch=test_torch,
            batch_runs=n_test,
        )

    # Warm-up (excluded from timed repeats).
    with torch.no_grad():
        for _ in range(3):
            run_stagefinal_once()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()

    # Time inference (multiple repeats, end-to-end stageFinal path).
    timings = []
    with torch.no_grad():
        for _ in range(20):
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_stagefinal_once()
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append(t1 - t0)

    return {
        "inference_mean_seconds": float(np.mean(timings)),
        "inference_std_seconds": float(np.std(timings)),
        "inference_per_sample_ms": float(np.mean(timings) / n_test * 1000),
        "n_test": n_test,
        "device": str(torch_device),
        "includes_backbone_stage": True,
        "pinn_flux_head_mode": flux_head_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Time comparison: simulator vs surrogates")
    parser.add_argument("--config", default="configs/sim/v3_layers_literature_clearance.yaml")
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--pinn_model", default=None, help="Path to pinn_model.pt checkpoint")
    parser.add_argument("--out_dir", default="results/timing")
    parser.add_argument("--sim_repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="Device for PINN inference (cpu or cuda)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    results = {}

    # 1. Simulator timing.
    print("Timing: single PDE simulation")
    sim_timing = time_single_simulation(args.config, n_repeats=args.sim_repeats)
    results["simulator"] = sim_timing
    print(f"Mean: {sim_timing['mean_seconds']:.3f}s (+/- {sim_timing['std_seconds']:.3f}s) over {sim_timing['n_repeats']} repeats")
    print(f"Grid: {sim_timing['grid']}, steps: {sim_timing['n_steps']}")

    # 2. Blackbox timing.
    print("\nTiming: blackbox (Ridge PCA)")
    try:
        bb_timing = time_blackbox_inference(args.ml_dir)
        results["blackbox_ridge"] = bb_timing
        print(f"Train: {bb_timing['train_seconds']:.3f}s")
        print(f"Inference ({bb_timing['n_test']} samples): {bb_timing['inference_mean_seconds']*1000:.2f}ms (+/- {bb_timing['inference_std_seconds']*1000:.2f}ms)")
        print(f"Per sample: {bb_timing['inference_per_sample_ms']:.3f}ms")
    except Exception as e:
        print(f"Skipped: {e}")
    # 3. PINN timing.
    if args.pinn_model and Path(args.pinn_model).exists():
        print("\nTiming: PINN inference")
        try:
            pinn_timing = time_pinn_inference(args.ml_dir, args.pinn_model, args.device)
            results["pinn"] = pinn_timing
            print(f"Inference ({pinn_timing['n_test']} samples): {pinn_timing['inference_mean_seconds']*1000:.2f}ms (+/- {pinn_timing['inference_std_seconds']*1000:.2f}ms)")
            print(f"Per sample: {pinn_timing['inference_per_sample_ms']:.3f}ms")
        except Exception as e:
            print(f"Skipped: {e}")
    else:
        print("\nPINN timing skipped (no --pinn_model provided)")

    # 4. Summary.
    if "simulator" in results and "blackbox_ridge" in results:
        sim_seconds_per_run = float(results["simulator"]["mean_seconds"])
        bb_seconds_per_sample = float(results["blackbox_ridge"]["inference_per_sample_ms"]) / 1000.0
        if bb_seconds_per_sample > 0.0:
            results["speedup_blackbox_vs_sim_per_sample"] = float(sim_seconds_per_run / bb_seconds_per_sample)
            print(f"\nSpeedup per sample: blackbox is {results['speedup_blackbox_vs_sim_per_sample']:.0f}x faster than simulator")
    if "simulator" in results and "pinn" in results:
        sim_seconds_per_run = float(results["simulator"]["mean_seconds"])
        pinn_seconds_per_sample = float(results["pinn"]["inference_per_sample_ms"]) / 1000.0
        if pinn_seconds_per_sample > 0.0:
            results["speedup_pinn_vs_sim_per_sample"] = float(sim_seconds_per_run / pinn_seconds_per_sample)
            print(f"Speedup per sample: PINN is {results['speedup_pinn_vs_sim_per_sample']:.0f}x faster than simulator")

    # Include context fields so reported speedups are interpretable.
    results["notes"] = {
        "simulator_unit": "seconds per full PDE run",
        "surrogate_unit": "seconds per test split and per sample (derived)",
        "speedup_definition": "simulator_seconds_per_run / surrogate_seconds_per_sample",
    }

    out_path = out_dir / "time_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
