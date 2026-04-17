import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ml.blackbox_data import load_npz
from scripts.ml.common import load_json, resolve_split_key
from scripts.ml.corrective_model import RunConditionedCorrectiveSurrogate, predict_flux_curves
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


def fit_shared_backbone(train_data, test_data):
    # Fit one shared blackbox backbone for apples-to-apples inference timing.
    x_train = np.asarray(train_data["X"], dtype=np.float32)
    j_train = np.asarray(train_data["J"], dtype=np.float32)
    x_test = np.asarray(test_data["X"], dtype=np.float32)
    j_test = np.asarray(test_data["J"], dtype=np.float32)

    t0 = time.perf_counter()
    backbone = fit_curve_backbone(
        x_train,
        j_train,
        x_test,
        j_test,
        curve_components=20,
        use_xgboost=False,
    )
    fit_seconds = time.perf_counter() - t0
    return backbone, float(fit_seconds)


def summarize_inference_timings(timings, n_test, warmup_repeats, inference_repeats, device):
    # Convert raw timing samples into a shared result schema.
    return {
        "inference_mean_seconds": float(np.mean(timings)),
        "inference_std_seconds": float(np.std(timings)),
        "inference_per_sample_ms": float(np.mean(timings) / float(n_test) * 1000.0),
        "n_test": int(n_test),
        "warmup_repeats": int(warmup_repeats),
        "inference_repeats": int(inference_repeats),
        "device": str(device),
        "comparison_scope": "inference_only_apples_to_apples",
    }


def time_blackbox_inference(backbone, x_test, warmup_repeats=3, inference_repeats=20):
    # Time blackbox stageFinal inference only.
    n_test = int(x_test.shape[0])

    for _ in range(int(warmup_repeats)):
        predict_curve_backbone(backbone, x_test)

    timings = []
    for _ in range(int(inference_repeats)):
        t0 = time.perf_counter()
        predict_curve_backbone(backbone, x_test)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    result = summarize_inference_timings(
        timings=timings,
        n_test=n_test,
        warmup_repeats=warmup_repeats,
        inference_repeats=inference_repeats,
        device="cpu",
    )
    result["includes_backbone_stage"] = True
    return result


def load_corrective_model(model_path, feature_dim, torch_device):
    # Load current-format corrective checkpoint and enforce additive-only mode.
    model_dir = Path(model_path).parent
    summary_path = model_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    ccfg = summary.get("corrective_config")
    if not isinstance(ccfg, dict):
        raise ValueError("summary.json is missing corrective_config")
    if ccfg.get("additive_correction", True) is not True:
        raise ValueError("Only additive correction runs are supported by this codebase.")

    checkpoint = torch.load(model_path, map_location=torch_device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")

    model = RunConditionedCorrectiveSurrogate(
        feature_dim=int(feature_dim),
        hidden_dim=int(ccfg.get("hidden_dim", 128)),
        depth=int(ccfg.get("depth", 4)),
        correction_scale=float(ccfg.get("correction_scale", 0.4)),
        use_learned_gate=bool(ccfg.get("use_learned_gate", True)),
        gate_init_bias=float(ccfg.get("gate_init_bias", -2.0)),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        raise ValueError(
            "Checkpoint is incompatible with the current corrective architecture. "
            "Re-train using the current scripts.ml.train_corrective code path."
        ) from None
    model.to(torch_device)
    model.eval()

    return model, str(ccfg.get("flux_head_mode", "corrective"))


def time_corrective_inference(
    backbone,
    x_test,
    t_test,
    model,
    torch_device,
    warmup_repeats=3,
    inference_repeats=20,
    corrective_flux_head_mode="corrective",
):
    # Time end-to-end stageFinal corrective-surrogate inference only.
    n_test = int(x_test.shape[0])

    t = np.asarray(t_test, dtype=np.float32)
    t_max = t.max(axis=1, keepdims=True)
    t_max = np.where(t_max > 0.0, t_max, 1.0)
    t_norm = (t / t_max).astype(np.float32)
    t_norm_t = torch.as_tensor(t_norm, dtype=torch.float32, device=torch_device)

    def run_stagefinal_once():
        # End-to-end stageFinal inference:
        # 1) backbone predict (j_base + x_scaled) 2) corrective branch.
        j_base, x_scaled = predict_curve_backbone(backbone, x_test)
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

    with torch.no_grad():
        for _ in range(int(warmup_repeats)):
            run_stagefinal_once()
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(int(inference_repeats)):
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_stagefinal_once()
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append(t1 - t0)

    result = summarize_inference_timings(
        timings=timings,
        n_test=n_test,
        warmup_repeats=warmup_repeats,
        inference_repeats=inference_repeats,
        device=torch_device,
    )
    result["includes_backbone_stage"] = True
    result["corrective_flux_head_mode"] = corrective_flux_head_mode
    return result


def main():
    parser = argparse.ArgumentParser(description="Time comparison: simulator vs surrogates")
    parser.add_argument("--config", default="configs/sim/v3_layers_literature_clearance.yaml")
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--corrective_model", default=None, help="Path to corrective_model.pt checkpoint")
    parser.add_argument("--out_dir", default="outputs/timing")
    parser.add_argument("--sim_repeats", type=int, default=5)
    parser.add_argument("--warmup_repeats", type=int, default=3)
    parser.add_argument("--inference_repeats", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="Apples-to-apples surrogate device; must be cpu")
    parser.add_argument("--best_hardware_device", default=None, help="Optional extra corrective timing device for deployment view (for example: cuda)")
    args = parser.parse_args()

    if int(args.sim_repeats) < 1:
        raise ValueError("--sim_repeats must be >= 1")
    if int(args.warmup_repeats) < 0:
        raise ValueError("--warmup_repeats must be >= 0")
    if int(args.inference_repeats) < 1:
        raise ValueError("--inference_repeats must be >= 1")
    apples_device_text = str(args.device).strip().lower()
    if args.corrective_model and apples_device_text != "cpu":
        raise ValueError("Apples-to-apples surrogate comparison requires --device cpu because blackbox inference is CPU-only.")
    best_device_text = None
    if args.best_hardware_device is not None:
        best_device_text = str(args.best_hardware_device).strip().lower()
        if best_device_text == "":
            best_device_text = None
    best_device_requested_text = best_device_text
    best_device_skip_reason = None
    if best_device_text and not args.corrective_model:
        raise ValueError("--best_hardware_device requires --corrective_model")
    if best_device_text and best_device_text.startswith("cuda") and not torch.cuda.is_available():
        best_device_skip_reason = "--best_hardware_device requested CUDA but torch.cuda.is_available() is False; skipping best-hardware timing"
        best_device_text = None

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    results = {}

    # 1. Simulator timing.
    print("Timing: single PDE simulation")
    sim_timing = time_single_simulation(args.config, n_repeats=int(args.sim_repeats))
    results["simulator"] = sim_timing
    print(f"Mean: {sim_timing['mean_seconds']:.3f}s (+/- {sim_timing['std_seconds']:.3f}s) over {sim_timing['n_repeats']} repeats")
    print(f"Grid: {sim_timing['grid']}, steps: {sim_timing['n_steps']}")

    # 2. Shared surrogate setup.
    train_data, test_data = load_ml_splits(args.ml_dir)
    x_test = np.asarray(test_data["X"], dtype=np.float32)
    t_test = np.asarray(test_data["t"], dtype=np.float32)
    backbone, backbone_fit_seconds = fit_shared_backbone(train_data, test_data)
    results["surrogate_setup"] = {
        "comparison_scope": "inference_only_apples_to_apples",
        "backbone_fit_seconds": float(backbone_fit_seconds),
        "n_train": int(train_data["X"].shape[0]),
        "n_test": int(x_test.shape[0]),
        "warmup_repeats": int(args.warmup_repeats),
        "inference_repeats": int(args.inference_repeats),
        "best_hardware_device_requested": best_device_requested_text,
        "best_hardware_skip_reason": best_device_skip_reason,
    }

    # 3. Blackbox inference timing.
    print("\nTiming: blackbox (Ridge PCA, inference only)")
    bb_timing = time_blackbox_inference(
        backbone=backbone,
        x_test=x_test,
        warmup_repeats=int(args.warmup_repeats),
        inference_repeats=int(args.inference_repeats),
    )
    results["blackbox_ridge"] = bb_timing
    print(f"Inference ({bb_timing['n_test']} samples): {bb_timing['inference_mean_seconds']*1000:.2f}ms (+/- {bb_timing['inference_std_seconds']*1000:.2f}ms)")
    print(f"Per sample: {bb_timing['inference_per_sample_ms']:.3f}ms")

    # 4. Corrective inference timing (strict current checkpoint contract).
    if args.corrective_model and Path(args.corrective_model).exists():
        print("\nTiming: physics-corrected surrogate inference (inference only, apples-to-apples CPU)")
        torch_device_cpu = torch.device("cpu")
        model_cpu, flux_head_mode = load_corrective_model(
            model_path=args.corrective_model,
            feature_dim=x_test.shape[1],
            torch_device=torch_device_cpu,
        )
        corrective_timing = time_corrective_inference(
            backbone=backbone,
            x_test=x_test,
            t_test=t_test,
            model=model_cpu,
            torch_device=torch_device_cpu,
            warmup_repeats=int(args.warmup_repeats),
            inference_repeats=int(args.inference_repeats),
            corrective_flux_head_mode=flux_head_mode,
        )
        results["corrective_surrogate"] = corrective_timing
        print(f"Inference ({corrective_timing['n_test']} samples): {corrective_timing['inference_mean_seconds']*1000:.2f}ms (+/- {corrective_timing['inference_std_seconds']*1000:.2f}ms)")
        print(f"Per sample: {corrective_timing['inference_per_sample_ms']:.3f}ms")

        if best_device_text and best_device_text != "cpu":
            print(f"\nTiming: physics-corrected surrogate inference (best hardware: {best_device_text})")
            torch_device_best = torch.device(best_device_text)
            model_best, flux_head_mode_best = load_corrective_model(
                model_path=args.corrective_model,
                feature_dim=x_test.shape[1],
                torch_device=torch_device_best,
            )
            corrective_timing_best = time_corrective_inference(
                backbone=backbone,
                x_test=x_test,
                t_test=t_test,
                model=model_best,
                torch_device=torch_device_best,
                warmup_repeats=int(args.warmup_repeats),
                inference_repeats=int(args.inference_repeats),
                corrective_flux_head_mode=flux_head_mode_best,
            )
            results["corrective_surrogate_best_hardware"] = corrective_timing_best
            print(f"Inference ({corrective_timing_best['n_test']} samples): {corrective_timing_best['inference_mean_seconds']*1000:.2f}ms (+/- {corrective_timing_best['inference_std_seconds']*1000:.2f}ms)")
            print(f"Per sample: {corrective_timing_best['inference_per_sample_ms']:.3f}ms")
        elif best_device_skip_reason:
            print(f"\nBest-hardware timing skipped ({best_device_skip_reason})")
        elif best_device_text == "cpu":
            print("\nBest-hardware timing skipped (--best_hardware_device=cpu matches apples-to-apples CPU run)")
    else:
        print("\nPhysics-corrected surrogate timing skipped (no --corrective_model provided)")

    # 5. Derived speedups and surrogate-to-surrogate ratio.
    if "simulator" in results and "blackbox_ridge" in results:
        sim_seconds_per_run = float(results["simulator"]["mean_seconds"])
        bb_seconds_per_sample = float(results["blackbox_ridge"]["inference_per_sample_ms"]) / 1000.0
        if bb_seconds_per_sample > 0.0:
            results["speedup_blackbox_vs_sim_per_sample"] = float(sim_seconds_per_run / bb_seconds_per_sample)
            print(f"\nSpeedup per sample: blackbox is {results['speedup_blackbox_vs_sim_per_sample']:.0f}x faster than simulator")

    if "simulator" in results and "corrective_surrogate" in results:
        sim_seconds_per_run = float(results["simulator"]["mean_seconds"])
        corrective_seconds_per_sample = float(results["corrective_surrogate"]["inference_per_sample_ms"]) / 1000.0
        if corrective_seconds_per_sample > 0.0:
            results["speedup_corrective_vs_sim_per_sample"] = float(sim_seconds_per_run / corrective_seconds_per_sample)
            print(f"Speedup per sample: physics-corrected surrogate is {results['speedup_corrective_vs_sim_per_sample']:.0f}x faster than simulator")

    if "simulator" in results and "corrective_surrogate_best_hardware" in results:
        sim_seconds_per_run = float(results["simulator"]["mean_seconds"])
        corrective_best_seconds_per_sample = float(results["corrective_surrogate_best_hardware"]["inference_per_sample_ms"]) / 1000.0
        if corrective_best_seconds_per_sample > 0.0:
            results["speedup_corrective_best_hardware_vs_sim_per_sample"] = float(sim_seconds_per_run / corrective_best_seconds_per_sample)
            print(f"Speedup per sample (best hardware): physics-corrected surrogate is {results['speedup_corrective_best_hardware_vs_sim_per_sample']:.0f}x faster than simulator")

    if "blackbox_ridge" in results and "corrective_surrogate" in results:
        bb_mean = float(results["blackbox_ridge"]["inference_mean_seconds"])
        corrective_mean = float(results["corrective_surrogate"]["inference_mean_seconds"])
        if bb_mean > 0.0:
            ratio = float(corrective_mean / bb_mean)
            results["corrective_vs_blackbox_inference_ratio"] = ratio
            results["corrective_overhead_vs_blackbox_percent"] = float((ratio - 1.0) * 100.0)

    if "corrective_surrogate" in results and "corrective_surrogate_best_hardware" in results:
        corrective_apples_mean = float(results["corrective_surrogate"]["inference_mean_seconds"])
        corrective_best_mean = float(results["corrective_surrogate_best_hardware"]["inference_mean_seconds"])
        if corrective_apples_mean > 0.0:
            ratio = float(corrective_best_mean / corrective_apples_mean)
            results["corrective_best_hardware_vs_apples_ratio"] = ratio
            results["corrective_best_hardware_improvement_percent"] = float((1.0 - ratio) * 100.0)

    if "blackbox_ridge" in results and "corrective_surrogate_best_hardware" in results:
        bb_mean = float(results["blackbox_ridge"]["inference_mean_seconds"])
        corrective_best_mean = float(results["corrective_surrogate_best_hardware"]["inference_mean_seconds"])
        if bb_mean > 0.0:
            ratio = float(corrective_best_mean / bb_mean)
            results["corrective_best_hardware_vs_blackbox_inference_ratio"] = ratio
            results["corrective_best_hardware_overhead_vs_blackbox_percent"] = float((ratio - 1.0) * 100.0)

    # Include context fields so reported speedups are interpretable.
    results["notes"] = {
        "comparison_scope": "inference_only_apples_to_apples_with_optional_best_hardware",
        "simulator_unit": "seconds per full PDE run",
        "surrogate_unit": "seconds per test split and per sample (derived)",
        "speedup_definition": "simulator_seconds_per_run / surrogate_seconds_per_sample",
        "apples_to_apples_device": "cpu",
        "best_hardware_device_requested": best_device_requested_text,
        "best_hardware_skip_reason": best_device_skip_reason,
    }

    out_path = out_dir / "time_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
