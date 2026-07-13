from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.env_sumo import PuneSUMOEnv
from src.baseline import FixedTimeController


def run(scenario: str, episodes: int, max_steps: int, n: int, seed: int):
    env = PuneSUMOEnv({
        "n_intersections": n,
        "scenario": scenario,
        "render": False,
        "seed": seed,
        "max_steps": max_steps,
        "use_global_reward": True,
        "beta": 0.0,
    })

    controller = FixedTimeController(n_agents=n)

    # Statistics
    per_int_step_co2 = []      # Per-intersection, per-step CO2 (mg)
    ep_total_co2 = []          # Network total CO2 per episode (mg)
    ep_co2_per_veh = []        # CO2 per completed trip (mg)
    ep_throughput = []

    for ep in range(episodes):
        obs = env.reset()
        controller.reset()

        done = False
        while not done:
            actions = controller.act(obs)
            obs, _, done, info = env.step(actions)

            for tl_id in env.tl_ids:
                per_int_step_co2.append(env._get_intersection_co2(tl_id))

        ep_total_co2.append(info.get("episode_co2", 0.0))
        ep_co2_per_veh.append(info.get("co2_per_veh", 0.0))
        ep_throughput.append(info.get("throughput", 0.0))

    env.close()

    # ==========================================================
    # Compute statistics
    # ==========================================================

    a = np.array(per_int_step_co2, dtype=np.float64)
    a_nonzero = a[a > 0]

    summary = {
        "scenario": scenario,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "intersections": n,

        "per_step_mean": float(a.mean()),
        "per_step_median": float(np.median(a)),
        "per_step_p90": float(np.percentile(a, 90)),
        "per_step_p99": float(np.percentile(a, 99)),
        "per_step_max": float(a.max()),

        "nonzero_mean": float(a_nonzero.mean()) if a_nonzero.size else 0.0,
        "share_nonzero": float(a_nonzero.size / max(a.size, 1)),

        "episode_total_co2_mean": float(np.mean(ep_total_co2)),
        "episode_total_co2_std": float(np.std(ep_total_co2)),

        "co2_per_vehicle_mean": float(np.mean(ep_co2_per_veh)),
        "co2_per_vehicle_std": float(np.std(ep_co2_per_veh)),

        "throughput_mean": float(np.mean(ep_throughput)),

        "suggested_co2_norm": float(np.percentile(a, 90)),
    }

    # ==========================================================
    # Save outputs
    # ==========================================================

    output_dir = Path("outputs") / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"probe_{seed}_{scenario}"

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    # ==========================================================
    # Existing terminal output (unchanged)
    # ==========================================================

    print(f"\n=== CO2 probe: scenario={scenario}, {episodes} episodes ===")

    print(
        f"per-intersection per-step CO2 (mg): "
        f"mean={a.mean():.1f}  "
        f"median={np.median(a):.1f}  "
        f"p90={np.percentile(a,90):.1f}  "
        f"p99={np.percentile(a,99):.1f}  "
        f"max={a.max():.1f}"
    )

    print(
        f"  (nonzero only) "
        f"mean={a_nonzero.mean() if a_nonzero.size else 0:.1f}  "
        f"share_nonzero={a_nonzero.size/max(a.size,1):.2%}"
    )

    print(
        f"network total CO2 per episode (mg): "
        f"mean={np.mean(ep_total_co2):.0f}  "
        f"std={np.std(ep_total_co2):.0f}"
    )

    print(
        f"network CO2 per completed trip (mg): "
        f"mean={np.mean(ep_co2_per_veh):.1f}  "
        f"std={np.std(ep_co2_per_veh):.1f}"
    )

    print(f"throughput (veh/episode): mean={np.mean(ep_throughput):.0f}")

    print(
        f"\nSuggested co2_norm ~= p90 = "
        f"{np.percentile(a,90):.0f} "
        f"(round to a clean number)."
    )

    print("\nSaved probe results:")
    print(f"  JSON : {json_path}")
    print(f"  CSV  : {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", type=str, default="uniform")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--N", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    run(
        scenario=args.scenario,
        episodes=args.episodes,
        max_steps=args.max_steps,
        n=args.N,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()