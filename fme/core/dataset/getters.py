import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.config import (
    ConcatDatasetConfig,
    MergeDatasetConfig,
    MergeNoConcatDatasetConfig,
    RepeatedInterval,
    TimeSlice,
    XarrayDataConfig,
)
from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.subset import XarraySubset
import re

from fme.core.dataset.xarray import XarrayDataset, get_raw_paths
from fme.core.typing_ import Slice


def _as_index_selection(
    subset: Slice | TimeSlice | RepeatedInterval, dataset: XarrayDataset
) -> slice | np.ndarray:
    """Converts a subset defined either as a Slice or TimeSlice into an index slice
    based on time coordinate in provided dataset.
    """
    if isinstance(subset, Slice):
        index_selection = subset.slice
    elif isinstance(subset, TimeSlice):
        index_selection = subset.slice(dataset.sample_start_times)
    elif isinstance(subset, RepeatedInterval):
        try:
            index_selection = subset.get_boolean_mask(len(dataset), dataset.timestep)
        except ValueError as e:
            raise ValueError(f"Error when applying RepeatedInterval to dataset: {e}")
    else:
        raise TypeError(f"subset must be Slice or TimeSlice, got {type(subset)}")
    return index_selection


def get_xarray_dataset(
    config: XarrayDataConfig, names: list[str], n_timesteps: int
) -> tuple["XarraySubset", DatasetProperties]:
    dataset = XarrayDataset(config, names, n_timesteps)
    properties = dataset.properties
    index_slice = _as_index_selection(config.subset, dataset)
    dataset = XarraySubset(dataset, index_slice)
    return dataset, properties


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    names: list[str],
    n_timesteps: int,
    strict: bool = True,
) -> tuple[list[XarraySubset], DatasetProperties]:
    datasets = []
    properties: DatasetProperties | None = None
    for config in dataset_configs:
        dataset, new_properties = get_xarray_dataset(config, names, n_timesteps)
        datasets.append(dataset)
        if properties is None:
            properties = new_properties
        elif not strict:
            try:
                properties.update(new_properties)
            except ValueError as e:
                warnings.warn(
                    f"Metadata for each ensemble member are not the same: {e}"
                )
        else:
            properties.update(new_properties)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")

    return datasets, properties


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    names: list[str],
    n_timesteps: int,
    strict: bool = True,
) -> tuple[XarrayConcat, DatasetProperties]:
    datasets, properties = get_datasets(
        dataset_configs, names, n_timesteps, strict=strict
    )
    ensemble = XarrayConcat(datasets)
    return ensemble, properties


def _infer_available_variables(config: XarrayDataConfig):
    """
    Infer the available variables from a XarrayDataset.
    """
    paths = get_raw_paths(config.data_path, config.file_pattern)
    if len(paths) == 0:
        raise ValueError(
            f"No files found for data_path='{config.data_path}' with file_pattern='{config.file_pattern}'."
        )
    dataset = xr.open_dataset(
        paths[0],
        decode_times=False,
        decode_timedelta=False,
        engine=config.engine,
        chunks=None,
    )
    present = set(dataset.data_vars)
    available = set(present)
    dims = set(dataset.dims)
    # Add derived/renamed names consistent with loader preprocessing
    # Simple renames (single-level)
    rename_map = {
        "skt": "surface_temperature",
        "sp": "PRESsfc",
        "t2m": "TMP2m",
        "u10": "UGRD10m",
        "v10": "VGRD10m",
        "mslhf": "LHTFLsfc",
        "msshf": "SHTFLsfc",
        "mtpr": "PRATEsfc",
        "z_over_g0": "HGTsfc",
    }
    for raw, ace in rename_map.items():
        if raw in present:
            available.add(ace)
    # HGTsfc can be computed from z/g0
    if "z" in present:
        available.add("HGTsfc")
    if "tisr" in present:
        available.add("DSWRFtoa")
    if "mtnlwrf" in present:
        available.add("ULWRFtoa")
    # Derived vars: mark as available if at least one sibling input is present,
    # because preprocessing can cross-load the missing sibling from its folder.
    if ("tisr" in present) or ("mtnswrf" in present):
        available.add("USWRFtoa")
    if ("avg_sdswrf" in present) or ("msnswrf" in present) or ("msdwswrf" in present):
        available.add("USWRFsfc")
    if ("avg_sdlwrf" in present) or ("msnlwrf" in present) or ("msdwlwrf" in present):
        available.add("ULWRFsfc")
    if ("d2m" in present) or ("sp" in present):
        available.add("Q2m")
    if "z_over_g0" in present:
        available.add("HGTsfc")
    # Sea ice fraction can be derived from ERA5 'siconc'
    if "siconc" in present:
        available.add("sea_ice_fraction")
    # Multi-level expansions
    def _add_levels(base, tmpl):
        if base in present and ("lev" in dims or "level" in dims or "plev" in dims or "p" in dims):
            lev_dim = None
            for cand in ("lev", "level", "plev", "p"):
                if cand in dataset.dims:
                    lev_dim = cand
                    break
            nlev = int(dataset.sizes.get(lev_dim, 0)) if lev_dim else 0
            for i in range(min(13, nlev)):
                available.add(tmpl.format(i))
    _add_levels("t", "air_temperature_{}")
    _add_levels("q", "specific_total_water_{}")
    _add_levels("u", "eastward_wind_{}")
    _add_levels("v", "northward_wind_{}")
    _add_levels("z", "geopotential_{}")

    # Enable sibling-backed expansions for u/v/q when only 't' is present.
    # If a level dimension exists, we can request eastward/northward wind and humidity
    # per-level fields, and the loader's preprocessing will populate them from siblings.
    if ("t" in present) and ("lev" in dims or "level" in dims or "plev" in dims or "p" in dims):
        lev_dim = None
        for cand in ("lev", "level", "plev", "p"):
            if cand in dataset.dims:
                lev_dim = cand
                break
        nlev = int(dataset.sizes.get(lev_dim, 0)) if lev_dim else 0
        max_levels = min(13, nlev) if nlev > 0 else 13
        for i in range(max_levels):
            available.add(f"eastward_wind_{i}")
            available.add(f"northward_wind_{i}")
            available.add(f"specific_total_water_{i}")
    # Always allow preprocess-added fields to be assigned to at least one dataset
    available.update({"global_mean_co2", "land_fraction", "ocean_fraction", "sea_ice_fraction"})
    return list(available)


def get_era5_daily_dataset(
    daily_root: str,
    names: list[str],
    n_timesteps: int,
    engine: str = "netcdf4",
    strict: bool = True,
):
    """Build a merged dataset from ERA5 daily directory layout (compute_stats style).

    - daily_root expects subfolders single_lev/<token>/*.nc and multi_lev/<token>/*.nc
    - names are ACE variable names expected by the model (expanded level names, etc.)
    """
    # Map ACE variable names to source folder tokens
    single_map = {
        "surface_temperature": "skt",
        "PRESsfc": "sp",
        "TMP2m": "t2m",
        "UGRD10m": "u10",
        "VGRD10m": "v10",
        "PRATEsfc": "mtpr",
        "LHTFLsfc": "mslhf",
        "SHTFLsfc": "msshf",
        "DSWRFtoa": "tisr",
        "HGTsfc": "z",  # loader will compute HGTsfc = z / g0
        "Q2m": "d2m",  # derive using sibling sp
        "USWRFsfc": "avg_sdswrf",  # sibling msnswrf
        "ULWRFsfc": "avg_sdlwrf",  # sibling msnlwrf
        "USWRFtoa": "tisr",  # sibling mtnswrf
        "ULWRFtoa": "mtnlwrf",
    }
    multi_tokens = {"t": r"^air_temperature_\d+$", "q": r"^specific_total_water_\d+$", "u": r"^eastward_wind_\d+$", "v": r"^northward_wind_\d+$", "z": r"^geopotential_\d+$"}

    tokens_single: set[str] = set()
    tokens_multi: set[str] = set()
    for name in names:
        # direct single-level map
        if name in single_map:
            tokens_single.add(single_map[name])
            # special: HGTsfc fallback to multi z if no single token exists at runtime
            continue
        # regex multi-level
        matched = False
        for token, pat in multi_tokens.items():
            if re.match(pat, name):
                tokens_multi.add(token)
                matched = True
                break
        if matched:
            continue
        # handle DSWRFtoa raw name if user included tisr directly
        if name == "tisr":
            tokens_single.add("tisr")

    # Always include sp for Q2m derivation if requested
    if any(n == "Q2m" for n in names):
        tokens_single.add("sp")
        tokens_single.add("d2m")
    # Fallbacks for HGTsfc: prefer single_lev 'z' (geopotential at surface)
    if "HGTsfc" in names:
        tokens_single.add("z")
    # Ensure paired inputs are present for derived variables
    if "USWRFsfc" in names:
        tokens_single.add("avg_sdswrf")
        tokens_single.add("msnswrf")
    if "ULWRFsfc" in names:
        tokens_single.add("avg_sdlwrf")
        tokens_single.add("msnlwrf")
    if "USWRFtoa" in names:
        tokens_single.add("tisr")
        tokens_single.add("mtnswrf")
    if "ULWRFtoa" in names:
        tokens_single.add("mtnlwrf")

    # Token-specific root overrides (from local repo outputs)
    OVERRIDE_SINGLE_ROOTS = {
        # Prefer local post-processed outputs; include 350x180 if that's where they live
        "avg_sdswrf": "/home/yjlee/ace-main/Ace_data_construction/out_350x180/daily/single_lev",
        "avg_sdlwrf": "/home/yjlee/ace-main/Ace_data_construction/out_350x180/daily/single_lev",
        "avg_sduvrf": "/home/yjlee/ace-main/Ace_data_construction/out_350x180/daily/single_lev",
    }

    # Build dataset configs
    configs: list[XarrayDataConfig] = []
    # Avoid adding avg_* override tokens as separate datasets to prevent time misalignment;
    # these will be loaded on-demand in preprocessing with padding.
    skip_as_dataset = {"avg_sdswrf"}
    for tok in sorted(tokens_single - skip_as_dataset):
        root_path = OVERRIDE_SINGLE_ROOTS.get(tok, f"{daily_root}/single_lev")
        configs.append(
            XarrayDataConfig(
                data_path=f"{root_path}/{tok}",
                file_pattern="*.nc",
                engine=engine,
            )
        )
    for tok in sorted(tokens_multi):
        configs.append(
            XarrayDataConfig(
                data_path=f"{daily_root}/multi_lev/{tok}",
                file_pattern="*.nc",
                engine=engine,
            )
        )

    # Ensure at least one stable source carries preprocess-added fields
    if len(configs) == 0:
        raise ValueError("No source tokens inferred for requested names.")
    # Prefer the first multi-level dataset as anchor; otherwise first single-level
    anchor_idx = None
    for i, cfg in enumerate(configs):
        if "/multi_lev/" in cfg.data_path:
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = 0
    # Append a duplicate anchor config that requests only preprocess-added names if needed
    pre_names = {"global_mean_co2", "land_fraction", "ocean_fraction"}
    if any(n in names for n in pre_names):
        anchor = configs[anchor_idx]
        configs.append(
            XarrayDataConfig(
                data_path=anchor.data_path,
                file_pattern=anchor.file_pattern,
                engine=anchor.engine,
            )
        )

    if not configs:
        raise ValueError("No source tokens inferred for requested names.")

    merged_config = MergeDatasetConfig(merge=configs)
    merged_dataset, props = get_merged_datasets(merged_config, names, n_timesteps)
    return merged_dataset, props


def get_per_dataset_names(
    merged_config: MergeDatasetConfig | MergeNoConcatDatasetConfig,
    names: list[str],
) -> list[list[str]]:
    merged_required_names = names.copy()
    per_dataset_names = []
    # Prefer assigning preprocess-added fields to the first multi-level source (anchor)
    pre_names = {"global_mean_co2", "land_fraction", "ocean_fraction"}
    anchor_idx = None
    for i, cfg in enumerate(merged_config.merge):
        if isinstance(cfg, XarrayDataConfig) and "/multi_lev/" in cfg.data_path:
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = 0
    for i, config in enumerate(merged_config.merge):
        if isinstance(config, XarrayDataConfig):
            current_source_variables = _infer_available_variables(config)
        elif isinstance(config, ConcatDatasetConfig):
            current_source_variables = _infer_available_variables(config.concat[0])
        # Allow only anchor to claim preprocess-added names
        allowed = set(current_source_variables)
        if i != anchor_idx:
            allowed = allowed - pre_names
        current_source_names = [name for name in merged_required_names if name in allowed]
        per_dataset_names.append(current_source_names)
        for name in current_source_names:
            merged_required_names.remove(name)
    return per_dataset_names


def get_merged_datasets(
    merged_config: MergeDatasetConfig,
    names: list[str],
    n_timesteps: int,
) -> tuple[MergedXarrayDataset, DatasetProperties]:
    merged_xarray_datasets = []
    merged_properties: DatasetProperties | None = None
    per_dataset_names = get_per_dataset_names(merged_config, names)
    config_counter = 0
    for config in merged_config.merge:
        if isinstance(config, XarrayDataConfig):
            current_source_xarray_dataset, current_source_properties = (
                get_xarray_dataset(
                    config,
                    per_dataset_names[config_counter],
                    n_timesteps,
                )
            )
            merged_xarray_datasets.append(current_source_xarray_dataset)
        elif isinstance(config, ConcatDatasetConfig):
            current_source_datasets, current_source_properties = get_datasets(
                config.concat,
                per_dataset_names[config_counter],
                n_timesteps,
                strict=config.strict,
            )
            current_source_ensemble = XarrayConcat(current_source_datasets)
            merged_xarray_datasets.append(current_source_ensemble)

        if merged_properties is None:
            merged_properties = current_source_properties
        else:
            merged_properties.update_merged_dataset(current_source_properties)
        config_counter += 1

    if merged_properties is None:
        raise ValueError("At least one dataset must be provided.")
    merged_datasets = MergedXarrayDataset(datasets=merged_xarray_datasets)
    return merged_datasets, merged_properties
