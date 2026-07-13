
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- known scenarios (longest first so multi-word names match before shorter) -
SCENARIOS = [
    "bus_heavy_saturated",
    "high_saturation",
    "morning_peak",
    "evening_peak",
    "bus_heavy",
    "uniform",
]

# metric key -> (display label, lower_is_better)
METRICS = {
    "avg_queue": ("Avg queue (PCU)", True),
    "co2_per_veh": ("CO2 per vehicle (mg)", True),
    "episode_co2": ("Episode CO2 (mg)", True),
    "nox_per_veh": ("NOx per vehicle (mg)", True),
    "pmx_per_veh": ("PMx per vehicle (mg)", True),
    "throughput": ("Throughput (veh/ep)", False),
    "avg_travel_time": ("Avg travel time (s)", True),
    "avg_reward": ("Avg reward/step", False),
    "loss": ("Loss", True),
}

BASELINE_METRIC_MAP = {  # dashboard key -> key inside baseline_results.json
    "avg_queue": "avg_queue_pcu",
    "throughput": "throughput",
    "avg_travel_time": "avg_travel_time",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_filename(stem: str):
    """Return (model, seed, scenario, beta) from a metrics-file stem.

    beta is a float in [0,1] or None when the filename has no _b### segment.
    """
    name = stem[:-len("_metrics")] if stem.endswith("_metrics") else stem

    beta = None
    m = re.search(r"_b(\d{3})$", name)
    if m:
        beta = int(m.group(1)) / 100.0
        name = name[: m.start()]

    scenario = None
    for sc in SCENARIOS:
        if name.endswith("_" + sc):
            scenario = sc
            name = name[: -(len(sc) + 1)]
            break
    if scenario is None:
        return None  # unrecognized layout

    m = re.search(r"_(\d+)$", name)
    if not m:
        return None
    seed = int(m.group(1))
    model = name[: m.start()]
    return model, seed, scenario, beta


@st.cache_data(show_spinner=False)
def load_runs(outputs_dir: str) -> pd.DataFrame:
    """Concatenate all *_metrics.csv into one long dataframe with run tags."""
    frames = []
    for csv in sorted(Path(outputs_dir).glob("*_metrics.csv")):
        parsed = parse_filename(csv.stem)
        if parsed is None:
            continue
        model, seed, scenario, beta = parsed
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        df["model"] = model
        df["seed"] = seed
        df["scenario"] = scenario
        df["beta"] = beta
        df["source_file"] = csv.name
        beta_tag = "b--" if beta is None else f"b{int(round(beta * 100)):03d}"
        df["run_id"] = f"{model} | {scenario} | seed {seed} | {beta_tag}"
        df["settings"] = f"{model} | {scenario} | {beta_tag}"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_baselines(outputs_dir: str) -> dict:
    path = Path(outputs_dir) / "baseline_results.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def last_n_mean(df: pd.DataFrame, col: str, n: int) -> float:
    if col not in df.columns:
        return float("nan")
    s = df.sort_values("episode")[col].dropna()
    if s.empty:
        return float("nan")
    return float(s.tail(n).mean())


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def get_outputs_dir() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outputs", default=None)
    args, _ = parser.parse_known_args()
    if args.outputs:
        return args.outputs
    here = Path(__file__).parent
    for cand in (here / "outputs", Path.cwd() / "outputs"):
        if cand.exists():
            return str(cand)
    return str(here / "outputs")


def main():
    st.set_page_config(page_title="traffic-MARL run analysis", layout="wide")
    st.title("traffic-MARL run analysis")

    default_dir = get_outputs_dir()
    outputs_dir = st.sidebar.text_input("Outputs directory", value=default_dir)

    data = load_runs(outputs_dir)
    baselines = load_baselines(outputs_dir)

    if data.empty:
        st.warning(
            f"No *_metrics.csv found under '{outputs_dir}'. "
            "Point the sidebar to your outputs folder."
        )
        st.stop()

    smoothing = st.sidebar.slider("EMA smoothing (episodes)", 1, 40, 15)
    last_n = st.sidebar.slider("Last-N episodes for summary", 5, 100, 30)

    st.sidebar.markdown("---")
    st.sidebar.caption("Filter runs")
    models = sorted(data["model"].unique())
    scenarios = sorted(data["scenario"].unique())
    sel_models = st.sidebar.multiselect("Model", models, default=models)
    sel_scenarios = st.sidebar.multiselect("Scenario", scenarios, default=scenarios)

    view = data[
        data["model"].isin(sel_models) & data["scenario"].isin(sel_scenarios)
    ].copy()
    if view.empty:
        st.warning("No runs match the current filter.")
        st.stop()

    available_metrics = [m for m in METRICS if m in view.columns]

    tab_curves, tab_seeds, tab_beta, tab_table = st.tabs(
        ["Training curves", "Seed comparison", "Beta / emissions", "Summary table"]
    )

    # ---------------- Training curves --------------------------------------- #
    with tab_curves:
        st.subheader("Per-episode training curves")
        metric = st.selectbox(
            "Metric", available_metrics,
            format_func=lambda m: METRICS[m][0], key="curve_metric",
        )
        run_ids = sorted(view["run_id"].unique())
        chosen = st.multiselect("Runs", run_ids, default=run_ids, key="curve_runs")
        if chosen:
            plot_df = pd.DataFrame()
            for rid in chosen:
                r = view[view["run_id"] == rid].sort_values("episode")
                if metric not in r.columns:
                    continue
                plot_df[rid] = ema(
                    r.set_index("episode")[metric], smoothing
                )
            if not plot_df.empty:
                st.line_chart(plot_df)
                st.caption(f"{METRICS[metric][0]} - EMA span {smoothing}.")

            # classical baseline reference (if this metric maps to one)
            if metric in BASELINE_METRIC_MAP and baselines:
                rows = []
                for sc in sel_scenarios:
                    for ctrl, mset in baselines.get(sc, {}).items():
                        mm = mset.get(BASELINE_METRIC_MAP[metric])
                        if mm:
                            rows.append(
                                {"scenario": sc, "controller": ctrl,
                                 "mean": mm["mean"], "std": mm.get("std", 0.0)}
                            )
                if rows:
                    st.caption("Classical baselines for this metric")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

    # ---------------- Seed comparison --------------------------------------- #
    with tab_seeds:
        st.subheader("Same settings, different seeds")
        st.caption(
            "Groups runs that share model + scenario + beta and overlays their "
            "seeds. Shows cross-seed spread (the reproducibility signal)."
        )
        settings_opts = sorted(view["settings"].unique())
        chosen_settings = st.selectbox("Settings group", settings_opts)
        metric = st.selectbox(
            "Metric", available_metrics,
            format_func=lambda m: METRICS[m][0], key="seed_metric",
        )
        grp = view[view["settings"] == chosen_settings]
        seeds = sorted(grp["seed"].unique())

        if metric in grp.columns:
            curve_df = pd.DataFrame()
            for sd in seeds:
                r = grp[grp["seed"] == sd].sort_values("episode")
                curve_df[f"seed {sd}"] = ema(
                    r.set_index("episode")[metric], smoothing
                )
            if not curve_df.empty:
                st.line_chart(curve_df)

            summ = []
            for sd in seeds:
                r = grp[grp["seed"] == sd]
                summ.append({"seed": sd,
                             f"last-{last_n} mean": last_n_mean(r, metric, last_n)})
            summ_df = pd.DataFrame(summ)
            vals = summ_df[f"last-{last_n} mean"].dropna()
            c1, c2, c3 = st.columns(3)
            c1.metric("Seeds", len(seeds))
            c2.metric("Cross-seed mean",
                      f"{vals.mean():.3f}" if not vals.empty else "n/a")
            c3.metric("Cross-seed std",
                      f"{vals.std():.3f}" if len(vals) > 1 else "n/a")
            st.dataframe(summ_df, use_container_width=True, hide_index=True)

    # ---------------- Beta / emissions -------------------------------------- #
    with tab_beta:
        st.subheader("Congestion vs emissions across beta")
        st.caption(
            "Plots last-N-episode means against beta, averaged over seeds. "
            "Needs runs with beta in the filename (…_b###). The two curves are "
            "the frontier: as beta rises, emissions should fall and queue rise "
            "if the objectives conflict."
        )
        beta_view = view[view["beta"].notna()].copy()
        if beta_view.empty:
            st.info(
                "No beta-tagged runs found. Re-run training after the filename "
                "fix so files carry a _b### segment."
            )
        else:
            b_scen = st.selectbox(
                "Scenario", sorted(beta_view["scenario"].unique()), key="beta_scen"
            )
            b_model = st.selectbox(
                "Model", sorted(beta_view["model"].unique()), key="beta_model"
            )
            sub = beta_view[
                (beta_view["scenario"] == b_scen) & (beta_view["model"] == b_model)
            ]
            emis_metric = "co2_per_veh" if "co2_per_veh" in sub.columns else None
            rows = []
            for (beta, seed), g in sub.groupby(["beta", "seed"]):
                rows.append({
                    "beta": beta,
                    "avg_queue": last_n_mean(g, "avg_queue", last_n),
                    "emissions": last_n_mean(g, emis_metric, last_n)
                    if emis_metric else float("nan"),
                })
            per_beta = (
                pd.DataFrame(rows).groupby("beta").mean(numeric_only=True)
                .sort_index()
            )
            if not per_beta.empty:
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.caption("Avg queue (PCU) vs beta")
                    st.line_chart(per_beta[["avg_queue"]])
                with cc2:
                    if emis_metric:
                        st.caption(f"{METRICS[emis_metric][0]} vs beta")
                        st.line_chart(per_beta[["emissions"]])
                st.caption("Per-beta means (averaged over seeds)")
                st.dataframe(per_beta.reset_index(), use_container_width=True,
                             hide_index=True)

                # endpoint divergence readout
                if {0.0, 1.0}.issubset(set(per_beta.index)):
                    q0 = per_beta.loc[0.0, "avg_queue"]
                    q1 = per_beta.loc[1.0, "avg_queue"]
                    e0 = per_beta.loc[0.0, "emissions"]
                    e1 = per_beta.loc[1.0, "emissions"]
                    st.markdown("**Endpoint divergence (beta 0 vs 1)**")
                    d1, d2 = st.columns(2)
                    if np.isfinite(q0) and np.isfinite(q1) and q0:
                        d1.metric("Queue change, b0->b1",
                                  f"{q1:.2f}", f"{(q1 - q0) / q0 * 100:+.1f}%")
                    if np.isfinite(e0) and np.isfinite(e1) and e0:
                        d2.metric("Emissions change, b0->b1",
                                  f"{e1:.0f}", f"{(e1 - e0) / e0 * 100:+.1f}%",
                                  delta_color="inverse")

    # ---------------- Summary table ----------------------------------------- #
    with tab_table:
        st.subheader(f"Last-{last_n}-episode summary, one row per run")
        rows = []
        for rid, g in view.groupby("run_id"):
            row = {"run": rid}
            for mk in available_metrics:
                row[METRICS[mk][0]] = last_n_mean(g, mk, last_n)
            row["episodes"] = int(g["episode"].max())
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("run")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download summary CSV",
            table.to_csv(index=False).encode(),
            file_name="run_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
