#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def _try_imports():
    try:
        import xarray as xr  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        xr = None  # type: ignore
    try:
        import netCDF4  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        pass
    return xr


def _find_coord_name(keys: List[str], candidates: List[str]) -> Optional[str]:
    lower_to_key = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand in lower_to_key:
            return lower_to_key[cand]
    return None


def _infer_time_resolution_hours(times) -> Optional[float]:
    try:
        if times is None:
            return None
        if len(times) < 2:
            return None
        delta = times[1] - times[0]
        # handle numpy timedelta64 or datetime.timedelta
        if hasattr(delta, "astype"):
            # numpy timedelta64 -> hours
            return float(delta.astype("timedelta64[h]") / (1))
        elif hasattr(delta, "total_seconds"):
            return delta.total_seconds() / 3600.0
        return None
    except Exception:
        return None


def _infer_grid_resolution_deg(lat_values, lon_values) -> Tuple[Optional[float], Optional[float]]:
    try:
        dlat = None
        dlon = None
        if lat_values is not None and len(lat_values) > 1:
            dlat = float(abs(lat_values[1] - lat_values[0]))
        if lon_values is not None and len(lon_values) > 1:
            dlon = float(abs(lon_values[1] - lon_values[0]))
        return dlat, dlon
    except Exception:
        return None, None


def _era5_heuristics(ds) -> Dict[str, object]:
    hints = []
    score = 0

    # file/global attrs
    for key in ("title", "institution", "source", "history"):
        val = str(ds.attrs.get(key, "")).lower()
        if "era5" in val:
            hints.append(f"global_attr:{key}:contains_era5")
            score += 2
        if "ecmwf" in val:
            hints.append(f"global_attr:{key}:contains_ecmwf")
            score += 1

    # variable name hints typical for ERA5 or ACE-ERA5 mapping
    era5_like_names = {
        # common ERA5 short names
        "t2m", "u10", "v10", "sp", "msl", "tp", "t", "u", "v", "q", "z",
        # ACE-ERA5 used names
        "PRESsfc", "surface_temperature", "DSWRFtoa", "DLWRFsfc", "ULWRFsfc",
        "ULWRFtoa", "DSWRFsfc", "USWRFsfc", "USWRFtoa", "LHTFLsfc", "SHTFLsfc",
        "air_temperature_0", "eastward_wind_0", "northward_wind_0", "specific_total_water_0",
        "land_fraction", "ocean_fraction", "sea_ice_fraction",
        # ERA5 NCAR mirror names noted in pipeline
        "swvl1", "swvl2", "swvl3", "swvl4", "ci", "lsm", "z",
    }
    present = set(map(str, ds.data_vars.keys()))
    matches = present.intersection(era5_like_names)
    if matches:
        hints.append(f"vars_match:{sorted(matches)[:6]}{'...' if len(matches) > 6 else ''}")
        score += min(len(matches), 6)  # cap influence

    likely = score >= 2
    return {"likely_era5": likely, "score": score, "hints": hints}


def inspect_with_xarray(path: str) -> Dict[str, object]:
    xr = _try_imports()
    if xr is None:
        raise RuntimeError("xarray not available")

    ds = xr.open_dataset(path)
    try:
        dims = {k: int(v) for k, v in ds.dims.items()}

        coord_keys = list(ds.coords.keys())
        lat_key = _find_coord_name(coord_keys, ["lat", "latitude", "y"])
        lon_key = _find_coord_name(coord_keys, ["lon", "longitude", "x"])
        time_key = _find_coord_name(coord_keys, ["time", "valid_time", "forecast_initial_time"])  # noqa: E501

        lat_values = ds[lat_key].values if lat_key in ds.coords else None
        lon_values = ds[lon_key].values if lon_key in ds.coords else None
        time_values = ds[time_key].values if time_key in ds.coords else None

        time_res_h = _infer_time_resolution_hours(time_values)
        dlat, dlon = _infer_grid_resolution_deg(lat_values, lon_values)

        variables = []
        for name, da in ds.data_vars.items():
            variables.append({
                "name": str(name),
                "dims": tuple(map(str, da.dims)),
                "shape": tuple(map(int, da.shape)),
                "units": str(da.attrs.get("units", "")),
                "long_name": str(da.attrs.get("long_name", "")),
            })

        era5 = _era5_heuristics(ds)

        return {
            "path": path,
            "dims": dims,
            "coords": {"lat": lat_key, "lon": lon_key, "time": time_key},
            "grid_resolution_deg": {"dlat": dlat, "dlon": dlon},
            "time_resolution_hours": time_res_h,
            "n_variables": len(variables),
            "variables_preview": variables[:20],
            "era5_detection": era5,
        }
    finally:
        ds.close()


def inspect_with_ncdump(path: str) -> Dict[str, object]:
    import subprocess

    try:
        out = subprocess.check_output(["ncdump", "-h", path], stderr=subprocess.STDOUT, text=True)
        header = out.splitlines()[:200]
        return {"path": path, "ncdump_header": header}
    except Exception as e:
        return {"path": path, "error": f"ncdump failed: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Inspect a NetCDF file and detect if it looks like ERA5.")
    parser.add_argument("path", help="Path to NetCDF file (.nc)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(2)

    try:
        info = inspect_with_xarray(args.path)
    except Exception as e:
        # Fallback to ncdump if xarray/netCDF4 not available or file unreadable by those
        info = {"fallback": True, "reason": str(e)}
        dump = inspect_with_ncdump(args.path)
        info.update(dump)

    if args.as_json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(f"Path: {info.get('path')}")
        if "dims" in info:
            print(f"Dims: {info['dims']}")
        if "coords" in info:
            print(f"Coords: {info['coords']}")
        if "grid_resolution_deg" in info:
            print(f"Grid resolution (deg): {info['grid_resolution_deg']}")
        if "time_resolution_hours" in info:
            print(f"Time resolution (hours): {info['time_resolution_hours']}")
        if "era5_detection" in info:
            ed = info["era5_detection"]
            print(f"Likely ERA5: {ed.get('likely_era5')} (score={ed.get('score')}, hints={ed.get('hints')})")
        if info.get("fallback"):
            print(f"Used fallback: {info.get('reason')}")
        if "variables_preview" in info:
            print("Variables (preview):")
            for v in info["variables_preview"]:
                print(f"  - {v['name']} dims={v['dims']} shape={v['shape']} units='{v['units']}'")
        if "ncdump_header" in info:
            print("ncdump -h (truncated):")
            for line in info["ncdump_header"]:
                print(line)


if __name__ == "__main__":
    main()


