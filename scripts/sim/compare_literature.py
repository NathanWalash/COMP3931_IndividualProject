import argparse

from skin_diffusion.config import load_config
from skin_diffusion.run_utils import run_simulation


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # run sim
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

    # get outputs
    P_sim = metrics.get("P")
    Tlag_sim = metrics.get("Tlag")
    J_curve = metrics.get("J")
    t_curve = metrics.get("t")

    # convert Tlag to hours (assumes time is seconds)
    if Tlag_sim is None:
        Tlag_hours = None
    else:
        Tlag_hours = Tlag_sim / 3600.0

    # compare to targets if they exist
    lit = cfg.extras.get("literature", {})
    drug = lit.get("drug", "unknown")
    P_target = lit.get("P_target_cm_s", None)
    Tlag_target = lit.get("Tlag_target_hours", None)

    print("literature compare")
    print("drug:", drug)
    print("targets: P =", P_target, "cm/s | Tlag =", Tlag_target, "hours")
    print("P_sim (cm/s):", P_sim)
    print("Tlag_sim (hours):", Tlag_hours)
    print("P_target (cm/s):", P_target)
    print("Tlag_target (hours):", Tlag_target)
    if J_curve is not None and t_curve is not None:
        print("flux stats: J_min =", min(J_curve), "J_max =", max(J_curve))
        print("flux stats: J_final =", J_curve[-1])
        print("time span (hours):", float(t_curve[-1]) / 3600.0)

    # simple ratio checks if targets are present
    if P_target is not None and P_sim is not None and P_target != 0:
        print("P ratio (sim/target):", P_sim / P_target)
    if Tlag_target is not None and Tlag_hours is not None and Tlag_target != 0:
        print("Tlag ratio (sim/target):", Tlag_hours / Tlag_target)
    if Tlag_hours is None:
        print("note: Tlag was not computable (flux tail not linear or slope <= 0)")


if __name__ == "__main__":
    main()
