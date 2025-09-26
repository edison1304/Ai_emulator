#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List


def _try_imports():
    try:
        import xarray as xr  # type: ignore
    except Exception:
        xr = None  # type: ignore
    try:
        import netCDF4  # type: ignore  # noqa: F401
    except Exception:
        netCDF4 = None  # type: ignore
    return xr, netCDF4


def list_variables_with_xarray(path: str, details: bool) -> List[str]:
    import xarray as xr  # type: ignore

    ds = None
    last_err = None  # type: ignore
    for engine in (None, "netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
            break
        except Exception as e:
            last_err = e
            ds = None
            continue
    if ds is None:
        raise last_err if last_err else RuntimeError("Failed to open dataset with xarray")
    try:
        lines: List[str] = []
        if details:
            for name, da in ds.data_vars.items():
                dims = tuple(map(str, da.dims))
                shape = tuple(map(int, da.shape))
                units = str(da.attrs.get("units", ""))
                long_name = str(da.attrs.get("long_name", ""))
                lines.append(f"{name} dims={dims} shape={shape} units='{units}' long_name='{long_name}'")
        else:
            lines.extend([str(name) for name in ds.data_vars.keys()])
        return lines
    finally:
        ds.close()


def list_variables_with_netCDF4(path: str, details: bool) -> List[str]:
    import netCDF4  # type: ignore

    ds = netCDF4.Dataset(path, mode="r")
    try:
        lines: List[str] = []
        if details:
            for name, var in ds.variables.items():
                dims = tuple(var.dimensions)
                shape = tuple(int(x) for x in var.shape) if hasattr(var, "shape") else ()
                units = getattr(var, "units", "")
                long_name = getattr(var, "long_name", "")
                lines.append(f"{name} dims={dims} shape={shape} units='{units}' long_name='{long_name}'")
        else:
            lines.extend([str(name) for name in ds.variables.keys()])
        return lines
    finally:
        ds.close()


def list_variables_with_ncdump(path: str) -> List[str]:
    import subprocess
    # Parse variable names from ncdump -h output (best-effort)
    try:
        out = subprocess.check_output(["ncdump", "-h", path], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return [f"Failed to run ncdump: {e}"]

    lines: List[str] = []
    in_variables = False
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("variables:"):
            in_variables = True
            continue
        if in_variables:
            if line.endswith(";") and "=" not in line:
                # Expected format: <type> <name>(dims...);
                tokens = line.split()
                if len(tokens) >= 2:
                    name_part = tokens[1]
                    name = name_part.split("(")[0]
                    lines.append(name)
            if line.endswith("}"):
                break
    return lines


def list_variables_with_h5py(path: str, details: bool) -> List[str]:
    # Best-effort inspection of NetCDF4/HDF5 structure without netCDF4/xarray
    import h5py  # type: ignore

    lines: List[str] = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            # Only datasets are candidate variables
            if not isinstance(obj, h5py.Dataset):
                return
            # Skip pure dimension scales
            cls = obj.attrs.get("CLASS", b"")
            if isinstance(cls, bytes):
                is_dim_scale = cls == b"DIMENSION_SCALE"
            else:
                is_dim_scale = False
            if is_dim_scale:
                return
            var_name = name.split("/")[-1]
            if details:
                shape = tuple(int(x) for x in obj.shape)
                units = obj.attrs.get("units", b"")
                long_name = obj.attrs.get("long_name", b"")
                if isinstance(units, bytes):
                    units = units.decode("utf-8", "ignore")
                if isinstance(long_name, bytes):
                    long_name = long_name.decode("utf-8", "ignore")
                lines.append(f"{var_name} shape={shape} units='{units}' long_name='{long_name}'")
            else:
                lines.append(var_name)

        f.visititems(visit)

    return lines


def main():
    parser = argparse.ArgumentParser(description="List variables in a NetCDF file")
    parser.add_argument("path", help="Path to NetCDF file (.nc)")
    parser.add_argument("--details", action="store_true", help="Print detailed info for each variable")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(2)

    xr, nc4 = _try_imports()
    try:
        if xr is not None:
            try:
                rows = list_variables_with_xarray(args.path, args.details)
            except Exception:
                try:
                    if nc4 is not None:
                        rows = list_variables_with_netCDF4(args.path, args.details)
                    else:
                        raise RuntimeError("netCDF4 module not available")
                except Exception:
                    try:
                        rows = list_variables_with_h5py(args.path, args.details)
                    except Exception:
                        rows = list_variables_with_ncdump(args.path)
        elif nc4 is not None:
            rows = list_variables_with_netCDF4(args.path, args.details)
        else:
            try:
                rows = list_variables_with_h5py(args.path, args.details)
            except Exception:
                rows = list_variables_with_ncdump(args.path)
    except Exception as e:
        print(f"Error reading variables: {e}", file=sys.stderr)
        sys.exit(1)

    for row in rows:
        print(row)
    print(f"Total variables: {len(rows)}")


if __name__ == "__main__":
    main()


