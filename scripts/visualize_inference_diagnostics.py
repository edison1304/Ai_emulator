#!/usr/bin/env python3
"""Generate quick-look plots for inference diagnostics NetCDF outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _parse_var_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"all", "*"}:
        return []
    items: List[str] = []
    for chunk in raw.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.extend(part for part in chunk.split() if part)
    return items or None


def _resolve_sample_labels(ds: xr.Dataset, override: Sequence[str] | None) -> List[str]:
    count = ds.dims.get("sample", 0)
    if count == 0:
        return []
    if override is not None:
        labels = list(override)
        if len(labels) != count:
            raise ValueError(
                f"Expected {count} sample labels, received {len(labels)}: {labels}"
            )
        return labels
    return [f"sample_{idx}" for idx in range(count)]


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_annual(ds: xr.Dataset, variables: Sequence[str], labels: Sequence[str], output: Path) -> None:
    years = ds.coords.get("year")
    if years is None:
        return
    x = years.values
    colors = plt.cm.get_cmap("tab10", len(labels))
    for var in variables:
        if var not in ds:
            print(f"[annual] skip missing variable: {var}")
            continue
        data = ds[var]
        if data.ndim != 2 or data.dims[0] != "sample":
            print(f"[annual] unexpected shape for {var}: {data.dims}")
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for idx, label in enumerate(labels):
            ax.plot(x, data.isel(sample=idx).values, label=label, color=colors(idx))
        ax.set_title(f"Annual diagnostics · {var}")
        ax.set_xlabel("Year")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        target = output / f"annual_{var}.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(f"saved {target}")


def _plot_enso(ds: xr.Dataset, labels: Sequence[str], output: Path, rolling: int) -> None:
    if "surface_temperature" not in ds:
        return
    data = ds["surface_temperature"]
    if data.ndim != 2:
        return
    time = data.coords.get("time")
    if time is None:
        return
    x = time.values
    colors = plt.cm.get_cmap("tab10", len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, label in enumerate(labels):
        series = data.isel(sample=idx)
        ax.plot(x, series.values, color=colors(idx), alpha=0.35, label=f"{label} raw")
        if rolling > 1:
            filt = series.rolling(time=rolling, center=True).mean()
            ax.plot(filt.time.values, filt.values, color=colors(idx), label=f"{label} ({rolling}-step MA)")
    ax.set_title("ENSO index diagnostics")
    ax.set_xlabel("Time")
    ax.set_ylabel("surface_temperature")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    target = output / "enso_surface_temperature.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"saved {target}")


def _plot_forecast_step(
    ds: xr.Dataset,
    variables: Sequence[str],
    output: Path,
    rolling: int,
    label: str,
) -> None:
    steps = ds.coords.get("forecast_step")
    if steps is None:
        return
    x = steps.values
    for var in variables:
        if var not in ds:
            print(f"[forecast] skip missing variable: {var}")
            continue
        data = ds[var]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, data.values, label=label)
        if rolling > 1:
            kernel = np.ones(rolling) / rolling
            smoothed = np.convolve(data.values, kernel, mode="same")
            ax.plot(x, smoothed, label=f"{label} ({rolling}-pt MA)", color="red")
        ax.set_title(f"Forecast-step diagnostics · {var}")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        target = output / f"forecast_{var}.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(f"saved {target}")


def _plot_maps(ds: xr.Dataset, variables: Sequence[str], output: Path, cmap: str) -> None:
    lat = ds.coords.get("lat")
    lon = ds.coords.get("lon")
    if lat is None or lon is None:
        return
    X, Y = np.meshgrid(lon.values, lat.values)
    for var in variables:
        if var not in ds:
            print(f"[map] skip missing variable: {var}")
            continue
        data = ds[var]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        mesh = ax.pcolormesh(X, Y, data.values, cmap=cmap, shading="auto")
        ax.set_title(f"Time-mean diagnostics · {var}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
        cbar.set_label(data.name)
        ax.set_aspect("auto")
        fig.tight_layout()
        target = output / f"map_{var}.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(f"saved {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("inference_results/epoch_0000"),
        help="Directory containing *_diagnostics.nc files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_results/epoch_0000/figures"),
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--sample-labels",
        type=str,
        default=None,
        help="Comma separated labels for the sample dimension (applies to annual/ENSO).",
    )
    parser.add_argument(
        "--annual-vars",
        type=str,
        default="surface_temperature,total_water_path,total_energy_ace2_path",
        help="Comma separated list of annual diagnostics variables to plot (use 'all' to include every variable).",
    )
    parser.add_argument(
        "--enso-rolling",
        type=int,
        default=3,
        help="Window length for rolling mean applied to ENSO plots (set 1 to disable).",
    )
    parser.add_argument(
        "--forecast-vars",
        type=str,
        default="weighted_mean_gen-surface_temperature,weighted_mean_gen-total_water_path,weighted_std_gen-surface_temperature",
        help="Comma separated list of forecast-step diagnostics variables to plot (use 'all' to include every variable).",
    )
    parser.add_argument(
        "--forecast-rolling",
        type=int,
        default=73,
        help="Moving-average window for forecast-step series (set 1 to disable).",
    )
    parser.add_argument(
        "--map-vars",
        type=str,
        default="gen_map-surface_temperature,gen_map-total_water_path",
        help="Comma separated list of time-mean map variables to plot (use 'all' to include every variable).",
    )
    parser.add_argument(
        "--map-cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name for map plots.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for every variable in each diagnostics file.",
    )
    args = parser.parse_args()

    output_dir = _ensure_output_dir(args.output)

    annual_path = args.root / "annual_diagnostics.nc"
    if annual_path.exists():
        with xr.open_dataset(annual_path) as ds:
            annual_spec = "all" if args.all else args.annual_vars
            variables = _parse_var_list(annual_spec) or list(ds.data_vars)
            labels = _resolve_sample_labels(ds, _parse_var_list(args.sample_labels))
            _plot_annual(ds, variables, labels, output_dir)

    enso_path = args.root / "enso_index_diagnostics.nc"
    if enso_path.exists():
        with xr.open_dataset(enso_path) as ds:
            labels = _resolve_sample_labels(ds, _parse_var_list(args.sample_labels))
            if labels:
                _plot_enso(ds, labels, output_dir, max(args.enso_rolling, 1))

    mean_path = args.root / "mean_diagnostics.nc"
    if mean_path.exists():
        with xr.open_dataset(mean_path) as ds:
            forecast_spec = "all" if args.all else args.forecast_vars
            variables = _parse_var_list(forecast_spec) or list(ds.data_vars)
            _plot_forecast_step(ds, variables, output_dir, max(args.forecast_rolling, 1), label="gen")

    map_path = args.root / "time_mean_diagnostics.nc"
    if map_path.exists():
        with xr.open_dataset(map_path) as ds:
            map_spec = "all" if args.all else args.map_vars
            variables = _parse_var_list(map_spec) or list(ds.data_vars)
            _plot_maps(ds, variables, output_dir, args.map_cmap)


if __name__ == "__main__":
    main()
