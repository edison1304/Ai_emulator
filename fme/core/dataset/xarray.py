import datetime
import time as _time
import json
import logging
import os
import re
import warnings
from collections import namedtuple
from functools import lru_cache
from urllib.parse import urlparse

import fsspec
import numpy as np
import torch
import xarray as xr
import warnings as _warnings
from xarray.coding.times import CFDatetimeCoder
import pandas as pd
from fme.core.constants import GRAVITY as G0

from fme.core.coordinates import (
    DepthCoordinate,
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    NullVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset.properties import DatasetProperties
from fme.core.mask_provider import MaskProvider
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict

from .config import XarrayDataConfig
from .data_typing import VariableMetadata
from .utils import (
    as_broadcasted_tensor,
    get_horizontal_coordinates,
    get_nonspacetime_dimensions,
    load_series_data,
    load_series_data_zarr_async,
)

SLICE_NONE = slice(None)
# Suppress noisy xarray time serialization warnings globally for this loader
try:
    from xarray.coding.times import SerializationWarning as _XRSerializationWarning
    _warnings.filterwarnings("ignore", category=_XRSerializationWarning)
except Exception:
    pass
logger = logging.getLogger(__name__)

VariableNames = namedtuple(
    "VariableNames",
    (
        "time_dependent_names",
        "time_invariant_names",
        "static_derived_names",
    ),
)


def _get_vertical_coordinate(
    ds: xr.Dataset, dtype: torch.dtype | None
) -> VerticalCoordinate:
    """
    Get vertical coordinate from a dataset.

    If the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number, then a hybrid sigma-pressure coordinate
    will be returned. If the dataset contains variables named
    `idepth_N` then a depth coordinate will be returned. If neither thing
    is true, a hybrid sigma-pressure coordinate of lenght 0 is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.
    """
    ak_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("ak_")
    }
    bk_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("bk_")
    }
    ak_list = [ak_mapping[k] for k in sorted(ak_mapping.keys())]
    bk_list = [bk_mapping[k] for k in sorted(bk_mapping.keys())]

    idepth_mapping = {
        int(v[7:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("idepth_")
    }
    idepth_list = [idepth_mapping[k] for k in sorted(idepth_mapping.keys())]

    if len(ak_list) > 0 and len(bk_list) > 0 and len(idepth_list) > 0:
        raise ValueError(
            "Dataset contains both hybrid sigma-pressure and depth coordinates. "
            "Can only provide one, or else the vertical coordinate is ambiguous."
        )

    coordinate: VerticalCoordinate
    surface_mask = None
    if len(idepth_list) > 0:
        if "mask_0" in ds.data_vars:
            mask_layers = {
                name: torch.as_tensor(ds[name].values, dtype=dtype)
                for name in ds.data_vars
                if re.match(r"mask_(\d+)$", name)
            }
            for name in mask_layers:
                if "time" in ds[name].dims:
                    raise ValueError("The ocean mask must by time-independent.")
            stacker = Stacker({"mask": ["mask_"]})
            mask = stacker("mask", mask_layers)
            if "surface_mask" in ds.data_vars:
                if "time" in ds["surface_mask"].dims:
                    raise ValueError("The surface mask must be time-independent.")
                surface_mask = torch.as_tensor(ds["surface_mask"].values, dtype=dtype)
        else:
            logger.warning(
                "Dataset does not contain a mask. Providing a DepthCoordinate with "
                "mask set to 1 at all layers."
            )
            mask = torch.ones(len(idepth_list) - 1, dtype=dtype)
        coordinate = DepthCoordinate(
            torch.as_tensor(idepth_list, dtype=dtype), mask, surface_mask
        )
    elif len(ak_list) > 0 and len(bk_list) > 0:
        coordinate = HybridSigmaPressureCoordinate(
            ak=torch.as_tensor(ak_list, dtype=dtype),
            bk=torch.as_tensor(bk_list, dtype=dtype),
        )
    else:
        # Attempt to infer a fixed-pressure vertical coordinate from a level coordinate
        level_coord_name = None
        for cand in ("plev", "level", "lev", "p"):
            if cand in ds.coords and ds[cand].ndim == 1 and ds[cand].size > 1:
                level_coord_name = cand
                break
        if level_coord_name is not None:
            try:
                p_mid = np.asarray(ds[level_coord_name].values, dtype=float)
                units = str(getattr(ds[level_coord_name], "attrs", {}).get("units", "")).lower()
                if units in ("pa", "pascal", "pascals"):
                    scale = 1.0
                elif units in ("hpa", "millibar", "mbar"):
                    scale = 100.0
                else:
                    scale = 100.0 if np.nanmax(p_mid) < 5000 else 1.0
                p_mid_pa = p_mid * scale
                # sort by descending pressure (top to bottom)
                order = np.argsort(-p_mid_pa)
                p_sorted = p_mid_pa[order]
                logp = np.log(p_sorted)
                logp_interfaces = np.empty(len(logp) + 1, dtype=float)
                logp_interfaces[1:-1] = 0.5 * (logp[:-1] + logp[1:])
                logp_interfaces[0] = logp[0] + (logp[0] - logp[1]) / 2.0
                logp_interfaces[-1] = logp[-1] + (logp[-1] - logp[-2]) / 2.0
                p_interfaces = np.exp(logp_interfaces)
                # If dataset has expanded per-level vars, align number of layers
                desired_n: int | None = None
                try:
                    import re as _re
                    max_idx = -1
                    for name in ds.data_vars:
                        m = _re.search(r"(air_temperature|specific_total_water|eastward_wind|northward_wind)_(\d+)$", name)
                        if m:
                            ii = int(m.group(2))
                            if ii > max_idx:
                                max_idx = ii
                    if max_idx >= 0:
                        desired_n = max_idx + 1
                except Exception:
                    desired_n = None
                if desired_n is not None and desired_n + 1 <= len(p_interfaces):
                    p_interfaces = p_interfaces[: desired_n + 1]
                ak = torch.as_tensor(p_interfaces, dtype=dtype)
                bk = torch.zeros_like(ak)
                coordinate = HybridSigmaPressureCoordinate(ak=ak, bk=bk)
                logger.info(
                    f"Inferred fixed-pressure vertical coordinate from '{level_coord_name}' with {len(ak)} interfaces."
                )
            except Exception:
                logger.warning("Dataset does not contain a vertical coordinate.")
                coordinate = NullVerticalCoordinate()
        else:
            logger.warning("Dataset does not contain a vertical coordinate.")
            coordinate = NullVerticalCoordinate()

    return coordinate


def _get_raw_times(paths: list[str], engine: str) -> list[np.ndarray]:
    times = []
    for path in paths:
        with _open_xr_dataset(path, engine=engine) as ds:
            times.append(ds.time.values)
    return times


def _repeat_and_increment_time(
    raw_times: list[np.ndarray], n_repeats: int, timestep: datetime.timedelta
) -> list[np.ndarray]:
    """Repeats and increments a collection of arrays of evenly spaced times."""
    n_timesteps = sum(len(times) for times in raw_times)
    timespan = timestep * n_timesteps

    repeated_and_incremented_time = []
    # precompute numpy ns increment per repeat
    try:
        base_inc_ns = pd.to_timedelta(timespan).to_timedelta64().astype('timedelta64[ns]').astype(int)
    except Exception:
        base_inc_ns = int(pd.to_timedelta(timespan).value)  # ns

    for repeats in range(n_repeats):
        inc_ns = repeats * base_inc_ns
        for time in raw_times:
            # Case A: numpy datetime64 array
            if isinstance(time, np.ndarray) and np.issubdtype(time.dtype, np.datetime64):
                t_ns = time.astype('datetime64[ns]')
                inc = np.timedelta64(inc_ns, 'ns')
                incremented_time = t_ns + inc
            else:
                # Fallback: object/cftime arrays -> use python timedelta
                increment = repeats * timespan
                try:
                    incremented_time = np.array([t + increment for t in time], dtype=object)
                except Exception:
                    # last resort: keep original
                    incremented_time = time
            repeated_and_incremented_time.append(incremented_time)
    return repeated_and_incremented_time


def _get_cumulative_timesteps(time: list[np.ndarray]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each item in a time coordinate."""
    num_timesteps_per_file = [0]
    for time_coord in time:
        num_timesteps_per_file.append(len(time_coord))
    return np.array(num_timesteps_per_file).cumsum()


def _get_file_local_index(index: int, start_indices: np.ndarray) -> tuple[int, int]:
    """
    Return a tuple of the index of the file containing the time point at `index`
    and the index of the time point within that file.
    """
    file_index = np.searchsorted(start_indices, index, side="right") - 1
    time_index = index - start_indices[file_index]
    return int(file_index), time_index


class StaticDerivedData:
    names = ("x", "y", "z")
    metadata = {
        "x": VariableMetadata(units="", long_name="Euclidean x-coordinate"),
        "y": VariableMetadata(units="", long_name="Euclidean y-coordinate"),
        "z": VariableMetadata(units="", long_name="Euclidean z-coordinate"),
    }

    def __init__(self, coordinates: HorizontalCoordinates):
        self._coords = coordinates
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._z: torch.Tensor | None = None

    def _get_xyz(self) -> TensorDict:
        if self._x is None or self._y is None or self._z is None:
            coords = self._coords
            x, y, z = coords.xyz

            self._x = torch.as_tensor(x)
            self._y = torch.as_tensor(y)
            self._z = torch.as_tensor(z)

        return {"x": self._x, "y": self._y, "z": self._z}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._get_xyz()[name]


def _get_protocol(path):
    return urlparse(str(path)).scheme


def _get_fs(path):
    protocol = _get_protocol(path)
    if not protocol:
        protocol = "file"
    proto_kw = _get_fs_protocol_kwargs(path)
    fs = fsspec.filesystem(protocol, **proto_kw)

    return fs


def _preserve_protocol(original_path, glob_paths):
    protocol = _get_protocol(str(original_path))
    if protocol:
        glob_paths = [f"{protocol}://{path}" for path in glob_paths]
    return glob_paths


def _get_fs_protocol_kwargs(path):
    protocol = _get_protocol(path)
    kwargs = {}
    if protocol == "gs":
        # https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem
        key_json = os.environ.get("FSSPEC_GS_KEY_JSON", None)
        key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

        if key_json is not None:
            token = json.loads(key_json)
        elif key_file is not None:
            token = key_file
        else:
            logger.warning(
                "GCS currently expects user credentials authenticated using"
                " `gcloud auth application-default login`. This is not recommended for "
                "production use."
            )
            token = "google_default"
        kwargs["token"] = token
    elif protocol == "s3":
        # https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage
        env_vars = [
            "FSSPEC_S3_KEY",
            "FSSPEC_S3_SECRET",
            "FSSPEC_S3_ENDPOINT_URL",
        ]
        for v in env_vars:
            if v not in os.environ:
                warnings.warn(
                    f"An S3 path was specified but environment variable {v} "
                    "was not found. This may cause authentication issues if not "
                    "set and no other defaults are present. See "
                    "https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage"
                    " for details."
                )

    return kwargs


def _open_xr_dataset(path: str, *args, **kwargs):
    # need the path to get protocol specific arguments for the backend
    protocol_kw = _get_fs_protocol_kwargs(path)
    if protocol_kw:
        kwargs.update({"storage_options": protocol_kw})

    return xr.open_dataset(
        path,
        *args,
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
        mask_and_scale=False,
        cache=False,
        chunks=None,
        **kwargs,
    )


_open_xr_dataset_lru = lru_cache()(_open_xr_dataset)


def _open_file_fh_cached(path, **kwargs):
    protocol = _get_protocol(path)
    if protocol:
        # add an LRU cache for remote zarrs
        return _open_xr_dataset_lru(
            path,
            **kwargs,
        )
    # netcdf4 and h5engine have a filehandle LRU cache in xarray
    # https://github.com/pydata/xarray/blob/cd3ab8d5580eeb3639d38e1e884d2d9838ef6aa1/xarray/backends/file_manager.py#L54 # noqa: E501
    return _open_xr_dataset(
        path,
        **kwargs,
    )


def get_raw_paths(path, file_pattern):
    fs = _get_fs(path)
    glob_paths = sorted(fs.glob(os.path.join(path, file_pattern)))
    raw_paths = _preserve_protocol(path, glob_paths)
    return raw_paths


def _get_mask_provider(ds: xr.Dataset, dtype: torch.dtype | None) -> MaskProvider:
    """
    Get mask provider from a dataset.

    If the dataset contains static variables that start with the string "mask_" or a
    variable named "surface_mask", then these variables will be used to instantiate
    a MaskProvider object. Otherwise, an empty MaskProvider is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.
    """
    masks: dict[str, torch.Tensor] = {
        name: torch.as_tensor(ds[name].values, dtype=dtype)
        for name in ds.data_vars
        if "mask_" in name
    }
    for name in masks:
        if "time" in ds[name].dims:
            raise ValueError("Masks must be time-independent.")
    mask_provider = MaskProvider(masks)
    logging.info(f"Initialized {mask_provider}.")
    return mask_provider


class XarrayDataset(torch.utils.data.Dataset):
    """Load data from a directory of files matching a pattern using xarray. The
    number of contiguous timesteps to load for each sample is specified by the
    n_timesteps argument.

    For example, if the file(s) have the time coordinate
    (t0, t1, t2, t3, t4) and n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4).
    """

    def __init__(self, config: XarrayDataConfig, names: list[str], n_timesteps: int):
        self._horizontal_coordinates: HorizontalCoordinates
        self._names = names
        self.path = config.data_path
        self.file_pattern = config.file_pattern
        self.engine = config.engine
        self.dtype = config.torch_dtype
        self.spatial_dimensions = config.spatial_dimensions
        self.fill_nans = config.fill_nans
        self.subset_config = config.subset
        self._raw_paths = get_raw_paths(self.path, self.file_pattern)
        if len(self._raw_paths) == 0:
            raise ValueError(
                f"No files found matching '{self.path}/{self.file_pattern}'."
            )
        self.full_paths = self._raw_paths * config.n_repeats
        self.sample_n_times = n_timesteps
        self._get_files_stats(config.n_repeats, config.infer_timestep)
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
            decode_timedelta=False,
            engine=self.engine,
            chunks=None,
        )
        # Apply ERA5-style preprocessing if applicable (dimension squeeze, renames, expansion, derived)
        first_dataset = self._preprocess_dataset(first_dataset, file_path=self.full_paths[0])
        
        self._mask_provider = _get_mask_provider(first_dataset, self.dtype)
        (
            self._horizontal_coordinates,
            self._static_derived_data,
            _loaded_horizontal_dims,
        ) = self.configure_horizontal_coordinates(first_dataset)
        (
            self._time_dependent_names,
            self._time_invariant_names,
            self._static_derived_names,
        ) = self._group_variable_names_by_time_type()
        

        self._vertical_coordinate = _get_vertical_coordinate(first_dataset, self.dtype)
        self.overwrite = config.overwrite
        # Determine non-spacetime dims from only the variables we actually load
        # to avoid introducing unused dims like 'lev' after multi-level expansion.
        try:
            names_for_dims = [
                n
                for n in (self._time_dependent_names + self._time_invariant_names)
                if n in first_dataset.data_vars
            ]
            if len(names_for_dims) > 0:
                subset_for_dims = first_dataset[names_for_dims]
            else:
                subset_for_dims = first_dataset
        except Exception:
            subset_for_dims = first_dataset
        self._nonspacetime_dims = get_nonspacetime_dimensions(
            subset_for_dims, _loaded_horizontal_dims
        )
        self._shape_excluding_time = [
            first_dataset.sizes[dim]
            for dim in (self._nonspacetime_dims + _loaded_horizontal_dims)
        ]
        self._loaded_dims = ["time"] + self._nonspacetime_dims + _loaded_horizontal_dims
        self.isel = {
            dim: v if isinstance(v, int) else v.slice for dim, v in config.isel.items()
        }
        self._isel_tuple = tuple(
            [self.isel.get(dim, SLICE_NONE) for dim in self._loaded_dims[1:]]
        )
        self._check_isel_dimensions(first_dataset.sizes)

    def _check_isel_dimensions(self, data_dim_sizes):
        # Horizontal dimensions are not currently supported, as the current isel code
        # does not adjust HorizonalCoordinates to match selection.
        if "time" in self.isel:
            raise ValueError("isel cannot be used to select time. Use subset instead.")

        for dim, selection in self.isel.items():
            if dim not in self._nonspacetime_dims:
                raise ValueError(
                    f"isel dimension {dim} must be a non-spacetime dimension "
                    f"of the dataset ({self._nonspacetime_dims})."
                )
            max_isel_index = (
                (selection.start or 0) if isinstance(selection, slice) else selection
            )
            if max_isel_index >= data_dim_sizes[dim]:
                raise ValueError(
                    f"isel index {max_isel_index} is out of bounds for dimension "
                    f"{dim} with size {data_dim_sizes[dim]}."
                )

    @property
    def _shape_excluding_time_after_selection(self):
        final_shape = []
        for orig_size, sel in zip(self._shape_excluding_time, self._isel_tuple):
            # if selecting a single index, dimension is squeezed
            # so it is not included in the final shape
            if isinstance(sel, slice):
                if sel.start is None and sel.stop is None and sel.step is None:
                    final_shape.append(orig_size)
                else:
                    final_shape.append(len(range(*sel.indices(orig_size))))
        return final_shape

    @property
    def dims(self) -> list[str]:
        # Final dimensions of returned data after dims that are selected
        # with a single index are dropped
        final_dims = ["time"]
        for dim, sel in zip(self._loaded_dims[1:], self._isel_tuple):
            if isinstance(sel, slice):
                final_dims.append(dim)
        return final_dims

    @property
    def properties(self) -> DatasetProperties:
        return DatasetProperties(
            self._variable_metadata,
            self._vertical_coordinate,
            self._horizontal_coordinates,
            self._mask_provider,
            self.timestep,
            self._is_remote,
        )

    @property
    def _is_remote(self) -> bool:
        protocol = _get_protocol(str(self.path))
        if not protocol or protocol == "file":
            return False
        return True

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """Time index of all available times in the data."""
        return self._all_times

    def _get_variable_metadata(self, ds):
        result = {}
        for name in self._names:
            if name in StaticDerivedData.names:
                result[name] = StaticDerivedData.metadata[name]
            elif name in ds.data_vars:
                da = ds[name]
                if hasattr(da, "units") and hasattr(da, "long_name"):
                    result[name] = VariableMetadata(
                        units=da.units,
                        long_name=da.long_name,
                    )
        self._variable_metadata = result

    def _get_files_stats(self, n_repeats: int, infer_timestep: bool):
        logging.info(f"Opening data at {os.path.join(self.path, self.file_pattern)}")
        raw_times = _get_raw_times(self._raw_paths, engine=self.engine)

        self._timestep: datetime.timedelta | None
        if infer_timestep:
            self._timestep = _get_timestep(np.concatenate(raw_times))
            time_coord = _repeat_and_increment_time(raw_times, n_repeats, self.timestep)
        else:
            self._timestep = None
            time_coord = raw_times

        cum_num_timesteps = _get_cumulative_timesteps(time_coord)
        self.start_indices = cum_num_timesteps[:-1]
        self.total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self.total_timesteps - self.sample_n_times + 1
        conc_time = np.concatenate(time_coord)
        # Build appropriate time index type depending on dtype
        try:
            import cftime  # type: ignore
            is_cftime = (
                conc_time.dtype == object and conc_time.size > 0 and isinstance(conc_time[0], cftime.DatetimeNoLeap | cftime.DatetimeGregorian | cftime.DatetimeProlepticGregorian)
            )
        except Exception:
            is_cftime = False
        if is_cftime:
            self._sample_start_times = xr.CFTimeIndex(
                conc_time[: self._n_initial_conditions]
            )
            self._all_times = xr.CFTimeIndex(conc_time)
        else:
            # Fall back to pandas DatetimeIndex for numpy datetime64 or mixed
            self._sample_start_times = pd.DatetimeIndex(conc_time[: self._n_initial_conditions])
            self._all_times = pd.DatetimeIndex(conc_time)

        del cum_num_timesteps, time_coord

        ds = self._open_file(0)
        self._get_variable_metadata(ds)

        logging.info(f"Found {self._n_initial_conditions} samples.")

    def _group_variable_names_by_time_type(self) -> VariableNames:
        """Returns lists of time-dependent variable names, time-independent
        variable names, and variables which are only present as an initial
        condition.
        """
        (
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        ) = ([], [], [])
        # Don't use open_mfdataset here, because it will give time-invariant
        # fields a time dimension. We assume that all fields are present in the
        # netcdf file corresponding to the first chunk of time.
        with _open_xr_dataset(self.full_paths[0], engine=self.engine) as ds:
            # Apply the same preprocessing used elsewhere so derived/renamed vars exist
            ds = self._preprocess_dataset(ds, file_path=self.full_paths[0])
            for name in self._names:
                if name in StaticDerivedData.names:
                    static_derived_names.append(name)
                else:
                    try:
                        da = ds[name]
                    except KeyError:
                        # Not present in this source; it may be provided by another dataset in the merge
                        continue
                    else:
                        dims = da.dims
                        if "time" in dims:
                            time_dependent_names.append(name)
                        else:
                            time_invariant_names.append(name)
            logging.info(
                f"The required variables have been found in the dataset: {self._names}."
            )

        return VariableNames(
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        )

    def configure_horizontal_coordinates(
        self, first_dataset
    ) -> tuple[HorizontalCoordinates, StaticDerivedData, list[str]]:
        horizontal_coordinates: HorizontalCoordinates
        static_derived_data: StaticDerivedData

        horizontal_coordinates, dim_names = get_horizontal_coordinates(
            first_dataset, self.spatial_dimensions, self.dtype
        )
        static_derived_data = StaticDerivedData(horizontal_coordinates)

        coords_sizes = {
            coord_name: len(coord)
            for coord_name, coord in horizontal_coordinates.coords.items()
        }
        logging.info(f"Horizontal coordinate sizes are {coords_sizes}.")
        return horizontal_coordinates, static_derived_data, dim_names

    @property
    def timestep(self) -> datetime.timedelta:
        if self._timestep is None:
            raise ValueError(
                "Timestep was not inferred in the data loader. Note "
                "XarrayDataConfig.infer_timestep must be set to True for this "
                "to occur."
            )
        else:
            return self._timestep

    def __len__(self):
        return self._n_initial_conditions

    def _open_file(self, idx):
        logger.debug(f"Opening file {self.full_paths[idx]}")
        ds = _open_file_fh_cached(self.full_paths[idx], engine=self.engine)
        return self._preprocess_dataset(ds, file_path=self.full_paths[idx])

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._sample_start_times

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        """Return a sample of data spanning the timesteps
        [idx, idx + self.sample_n_times).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a sample's data (i.e. a mapping from names to torch.Tensors) and
            its corresponding time coordinate.
        """
        time_slice = slice(idx, idx + self.sample_n_times)
        return self.get_sample_by_time_slice(time_slice)

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> tuple[TensorDict, xr.DataArray]:
        t0 = _time.time()
        input_file_idx, input_local_idx = _get_file_local_index(
            time_slice.start, self.start_indices
        )
        output_file_idx, output_local_idx = _get_file_local_index(
            time_slice.stop - 1, self.start_indices
        )

        # get the sequence of observations
        arrays: dict[str, list[torch.Tensor]] = {}
        idxs = range(input_file_idx, output_file_idx + 1)
        total_steps = 0
        for i, file_idx in enumerate(idxs):
            start = input_local_idx if i == 0 else 0
            if i == len(idxs) - 1:
                stop = output_local_idx
            else:
                stop = (
                    self.start_indices[file_idx + 1] - self.start_indices[file_idx] - 1
                )

            n_steps = stop - start + 1
            shape = [n_steps] + self._shape_excluding_time_after_selection
            total_steps += n_steps
            if self.engine == "zarr":
                tensor_dict = load_series_data_zarr_async(
                    idx=start,
                    n_steps=n_steps,
                    path=self.full_paths[file_idx],
                    names=self._time_dependent_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                    nontime_selection=self._isel_tuple,
                )
            else:
                ds = self._open_file(file_idx)
                ds = ds.isel(**self.isel)
                tensor_dict = load_series_data(
                    idx=start,
                    n_steps=n_steps,
                    ds=ds,
                    names=self._time_dependent_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                )
                ds.close()
                del ds
            for n in self._time_dependent_names:
                arrays.setdefault(n, []).append(tensor_dict[n])

        tensors: TensorDict = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays

        # Ensure surface masks remain 2D per timestep (time, H, W) without non-spacetime dims
        try:
            for _mask_name in ("ocean_fraction", "land_fraction"):
                if _mask_name in tensors:
                    _m = tensors[_mask_name]
                    # Expect at least (time, H, W). Remove any extra dims between time and spatial dims.
                    while _m.dim() > 3:
                        # remove the first non-time extra dim by taking slice 0
                        _m = _m.select(1, 0)
                    tensors[_mask_name] = _m
        except Exception:
            pass

        # load time-invariant variables from first dataset
        if len(self._time_invariant_names) > 0:
            ds = self._open_file(idxs[0])
            ds = ds.isel(**self.isel)
            shape = [total_steps] + self._shape_excluding_time_after_selection
            for name in self._time_invariant_names:
                variable = ds[name].variable
                if self.fill_nans is not None:
                    variable = variable.fillna(self.fill_nans.value)
                tensors[name] = as_broadcasted_tensor(variable, self.dims, shape)
            ds.close()
            del ds

        # load static derived variables
        for name in self._static_derived_names:
            tensor = self._static_derived_data[name]
            horizontal_dims = [1] * tensor.ndim
            tensors[name] = tensor.repeat((total_steps, *horizontal_dims))

        # cast to desired dtype
        tensors = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}

        # Apply field overwrites
        tensors = self.overwrite.apply(tensors)

        # Create a DataArray of times to return corresponding to the slice that
        # is valid even when n_repeats > 1.
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])

        # Logging: report build stats for this sample
        try:
            files_count = (output_file_idx - input_file_idx + 1)
            logging.debug(
                f"[Load] Built sample slice {time_slice.start}:{time_slice.stop} steps={total_steps} files={files_count} time_ms={int((_time.time()-t0)*1000)}"
            )
        except Exception:
            pass
        
        return tensors, time

    # ---------------------------
    # ERA5-style preprocessing
    # ---------------------------
    def _preprocess_dataset(self, ds: xr.Dataset, file_path: str | None = None) -> xr.Dataset:
        """Standardize ERA5 daily layout to model-expected variable names and dims.

        - Drop lev dimension when its size is 1
        - Rename single-level ERA5 variable names to ACE names
        - Expand multi-level variables (t,q,u,v,z) into per-level variables when requested
        - Add common derived variables if inputs exist (USWRFtoa, ULWRFtoa, USWRFsfc, ULWRFsfc, Q2m)
        """
        ds2 = ds
        try:
            logging.debug(f"[Preprocess] Start file={file_path}")
            _tpre = _time.time()
        except Exception:
            pass
        requested_names = set(getattr(self, "_names", []))
        # 1) Squeeze lev when size==1
        if "lev" in ds2.sizes and ds2.sizes.get("lev", None) == 1:
            # Drop lev from all data variables that include it
            for name, da in list(ds2.data_vars.items()):
                if "lev" in da.dims:
                    try:
                        ds2[name] = da.isel(lev=0, drop=True)
                    except TypeError:
                        # Older xarray: use squeeze
                        ds2[name] = da.isel(lev=0).squeeze()
            # Drop coordinate if present
            if "lev" in ds2.coords:
                ds2 = ds2.drop_vars("lev")

        # 2) Rename single-level ERA5 tokens to ACE
        rename_map = {
                # surface & near-surface
                "skt": "surface_temperature",
                "sst": "surface_temperature_ocean",
                "sp": "PRESsfc",
                "t2m": "TMP2m",
                "u10": "UGRD10m",
                "v10": "VGRD10m",
                # fluxes / rates
                "mslhf": "LHTFLsfc",
                "msshf": "SHTFLsfc",
                "mtpr": "PRATEsfc",
                # radiative avg/net
                "avg_sdswrf": "avg_sdswrf",
                "avg_sdlwrf": "avg_sdlwrf",
                "msnswrf": "msnswrf",
                "msnlwrf": "msnlwrf",
                # toa
                "tisr": "DSWRFtoa",
                "mtnswrf": "mtnswrf",
                "mtnlwrf": "mtnlwrf",
                # geopotential height (pre-derived)
                "z_over_g0": "HGTsfc",
        }
        keys_to_rename = {k: v for k, v in rename_map.items() if k in ds2.data_vars and v != k}
        if keys_to_rename:
            ds2 = ds2.rename(keys_to_rename)

        # 3) Derived variables with cross-folder sibling lookup where needed
        import os as _os
        def _open_sibling_variable(token: str) -> xr.DataArray | None:
            if file_path is None:
                return None
            # Known overrides (local post-processed outputs)
            OVERRIDE_SINGLE_ROOTS = {
                # Prefer local post-processed outputs; include both 350x180 and 360x180
                "avg_sdswrf": "/home/yjlee/ace-main/Ace_data_construction/out_360x180/daily/single_lev",
                "avg_sdlwrf": "/home/yjlee/ace-main/Ace_data_construction/out_360x180/daily/single_lev",
                "avg_sduvrf": "/home/yjlee/ace-main/Ace_data_construction/out_360x180/daily/single_lev",
            }
            base_name = _os.path.basename(file_path)
            parent = _os.path.dirname(file_path)
            # Extract YYYYMM from filename tail
            import re as _re
            m = _re.search(r"(\d{6})\.nc$", base_name)
            date_tag = m.group(1) if m else None
            candidate_dirs: list[str] = []
            # 1) If file resides under .../daily/single_lev or .../daily/multi_lev, derive daily root
            for tier in ("single_lev", "multi_lev"):
                idx = parent.find(f"/{tier}/")
                if idx != -1:
                    daily_root = parent[:idx]
                    # prefer matching-tier folders first
                    candidate_dirs.append(_os.path.join(daily_root, tier, token))
                    # also try single_lev/multi_lev counterpart folders
                    other = "multi_lev" if tier == "single_lev" else "single_lev"
                    candidate_dirs.append(_os.path.join(daily_root, other, token))
                    break
            # 2) Override roots for special tokens
            if token in OVERRIDE_SINGLE_ROOTS:
                candidate_dirs.insert(0, _os.path.join(OVERRIDE_SINGLE_ROOTS[token]))
                # Also try 350x180 sibling overrides if available
                candidate_dirs.insert(1, "/home/yjlee/ace-main/Ace_data_construction/out_350x180/daily/single_lev")
            # 3) Known global/local root fallbacks (try 350x180 then 360x180)
            candidate_dirs.append("/home/yjlee/ace-main/Ace_data_construction/out_350x180/daily/single_lev/" + token)
            candidate_dirs.append("/data1/DATA_ARCHIVE/Reanalysis/ERA5_new/data_360x180/daily/single_lev/" + token)
            # global fallbacks for multi-level variables
            candidate_dirs.append("/data1/DATA_ARCHIVE/Reanalysis/ERA5_new/data_360x180/daily/multi_lev/" + token)

            # Deduplicate while preserving order
            seen = set()
            cand_dirs = []
            for d in candidate_dirs:
                if d and d not in seen:
                    seen.add(d)
                    cand_dirs.append(d)

            import glob as _glob
            for sibling_dir in cand_dirs:
                if not _os.path.isdir(sibling_dir):
                    continue
                if date_tag is None:
                    pattern = _os.path.join(sibling_dir, "*.nc")
                else:
                    pattern = _os.path.join(sibling_dir, f"*_" + date_tag + ".nc")
                matches = sorted(_glob.glob(pattern))
                if not matches:
                    continue
                sibling_path = matches[0]
                try:
                    sib = xr.open_dataset(
                        sibling_path,
                        decode_times=False,
                        decode_timedelta=False,
                        engine=self.engine,
                        chunks=None,
                    )
                    # Reduced verbosity: avoid noisy sibling load logs
                    logging.debug(f"[SiblingLoad] Opened sibling for '{token}': {sibling_path}")
                    var_name = token if token in sib.data_vars else (list(sib.data_vars)[0] if len(sib.data_vars) > 0 else None)
                    if var_name is None:
                        sib.close()
                        continue
                    var_da = sib[var_name]
                    # Read only needed time window corresponding to current ds2 time length
                    try:
                        if "time" in var_da.dims and "time" in ds2.dims:
                            n_time = int(ds2.sizes.get("time", 0))
                            if n_time > 0:
                                m_time = int(var_da.sizes.get("time", 0))
                                n_sel = n_time if m_time >= n_time else m_time
                                var_da = var_da.isel(time=slice(0, n_sel))
                    except Exception:
                        pass
                    da = var_da.load()
                    sib.close()
                    return da
                except Exception:
                    continue
            return None

        # Helper: align a DataArray's time coordinate to this dataset's time to avoid dtype mismatches
        def _align_time_to_ds(da: xr.DataArray | None) -> xr.DataArray | None:
            if da is None:
                return None
            try:
                if "time" in da.dims and "time" in ds2.dims:
                    if da.sizes.get("time") == ds2.sizes.get("time"):
                        return da.assign_coords(time=ds2["time"])  # type: ignore
            except Exception:
                pass
            return da

        # Helper: drop any length-1 dims except time to avoid producing (time,1,lat,lon)
        def _drop_length1_non_time_dims(da: xr.DataArray | None) -> xr.DataArray | None:
            if da is None:
                return None
            try:
                for dim, size in list(da.sizes.items()):
                    if dim != "time" and size == 1:
                        try:
                            da = da.isel({dim: 0}, drop=True)
                        except TypeError:
                            da = da.isel({dim: 0}).squeeze()
            except Exception:
                pass
            return da

        # Helper: ensure ordering (time, lat, lon) when present
        def _order_time_lat_lon(da: xr.DataArray | None) -> xr.DataArray | None:
            if da is None:
                return None
            try:
                lat_cands = ["lat", "latitude", "grid_yt", "y"]
                lon_cands = ["lon", "longitude", "grid_xt", "x"]
                ds_lat = next((d for d in lat_cands if d in ds2.dims), None)
                ds_lon = next((d for d in lon_cands if d in ds2.dims), None)
                dims_order = []
                if "time" in da.dims:
                    dims_order.append("time")
                if ds_lat and ds_lat in da.dims:
                    dims_order.append(ds_lat)
                if ds_lon and ds_lon in da.dims:
                    dims_order.append(ds_lon)
                # Only transpose if we collected at least 2 dims in order
                if len(dims_order) >= 2 and all(d in da.dims for d in dims_order):
                    da = da.transpose(*dims_order)
            except Exception:
                pass
            return da

        # Helper: coerce spatial coordinates to exactly match ds2 to avoid label-based reindex -> NaNs
        def _coerce_spatial_coords_to_ds(da: xr.DataArray | None) -> xr.DataArray | None:
            if da is None:
                return None
            try:
                lat_cands = ["lat", "latitude", "grid_yt", "y"]
                lon_cands = ["lon", "longitude", "grid_xt", "x"]
                ds_lat = next((d for d in lat_cands if d in ds2.dims), None)
                ds_lon = next((d for d in lon_cands if d in ds2.dims), None)
                if ds_lat and ds_lon and (ds_lat in da.dims) and (ds_lon in da.dims):
                    if da.sizes.get(ds_lat) == ds2.sizes.get(ds_lat) and da.sizes.get(ds_lon) == ds2.sizes.get(ds_lon):
                        da = da.assign_coords({ds_lat: ds2[ds_lat], ds_lon: ds2[ds_lon]})  # type: ignore
            except Exception:
                pass
            return da

        # TOA shortwave up: DSWRFtoa - mtnswrf
        if "USWRFtoa" in requested_names and ("USWRFtoa" not in ds2.data_vars):
            lhs = ds2.data_vars.get("DSWRFtoa")
            rhs = ds2.data_vars.get("mtnswrf")
            if lhs is None:
                lhs = _open_sibling_variable("tisr")
                if lhs is not None:
                    # rename to DSWRFtoa semantics
                    lhs = lhs.rename("DSWRFtoa") if getattr(lhs, "name", None) else lhs
            if rhs is None:
                rhs = _open_sibling_variable("mtnswrf")
            lhs = _align_time_to_ds(lhs)
            rhs = _align_time_to_ds(rhs)
            if lhs is not None and rhs is not None:
                _lhs = _drop_length1_non_time_dims(lhs)
                _rhs = _drop_length1_non_time_dims(rhs)
                res = (_lhs - _rhs)
                res = _drop_length1_non_time_dims(res)
                res = _order_time_lat_lon(res)
                ds2["USWRFtoa"] = res.assign_attrs(description="Upward shortwave flux at TOA", source="DSWRFtoa,mtnswrf")
            else:
                logging.error(
                    f"[error] Cannot compute USWRFtoa in {file_path}: missing {'DSWRFtoa' if lhs is None else ''} {'mtnswrf' if rhs is None else ''}"
                )
                raise ValueError("Missing inputs for USWRFtoa")

        # Ensure TOA downward shortwave DSWRFtoa is available from tisr if requested
        if "DSWRFtoa" in requested_names and ("DSWRFtoa" not in ds2.data_vars):
            src = ds2.data_vars.get("DSWRFtoa")
            if src is None:
                # primary: tisr
                src = ds2.data_vars.get("tisr") or _open_sibling_variable("tisr")
            src = _align_time_to_ds(src)
            if src is not None:
                _src = _drop_length1_non_time_dims(src)
                _src = _order_time_lat_lon(_src)
                ds2["DSWRFtoa"] = _src.assign_attrs(description="Downward shortwave flux at TOA (from tisr)")
            else:
                # fallback: if USWRFtoa and mtnswrf are present, DSWRFtoa = USWRFtoa + mtnswrf
                uswrf_toa = ds2.data_vars.get("USWRFtoa")
                if uswrf_toa is None:
                    uswrf_toa = _open_sibling_variable("USWRFtoa")
                mtnswrf = ds2.data_vars.get("mtnswrf")
                if mtnswrf is None:
                    mtnswrf = _open_sibling_variable("mtnswrf")
                uswrf_toa = _align_time_to_ds(uswrf_toa)
                mtnswrf = _align_time_to_ds(mtnswrf)
                if uswrf_toa is not None and mtnswrf is not None:
                    _u = _drop_length1_non_time_dims(uswrf_toa)
                    _m = _drop_length1_non_time_dims(mtnswrf)
                    res = (_u + _m)
                    res = _drop_length1_non_time_dims(res)
                    res = _order_time_lat_lon(res)
                    ds2["DSWRFtoa"] = res.assign_attrs(description="Downward SW flux at TOA (USWRFtoa + mtnswrf)")
                else:
                    logging.debug(f"Cannot add DSWRFtoa in {file_path}: missing tisr or (USWRFtoa,mtnswrf)")

        # Surface temperature from sibling skt if missing
        if "surface_temperature" in requested_names and ("surface_temperature" not in ds2.data_vars):
            st = ds2.data_vars.get("skt")
            if st is None:
                st = _open_sibling_variable("skt")
            st = _align_time_to_ds(st)
            if st is not None:
                _st = _drop_length1_non_time_dims(st)
                _st = _order_time_lat_lon(_st)
                ds2["surface_temperature"] = _st.assign_attrs(description="Surface temperature (from skt)")
            else:
                logging.debug(f"Cannot add surface_temperature in {file_path}: missing skt")

        # Surface pressure PRESsfc from ERA5 sp if missing
        if "PRESsfc" in requested_names and ("PRESsfc" not in ds2.data_vars):
            ps = ds2.data_vars.get("PRESsfc")
            if ps is None:
                ps = ds2.data_vars.get("sp") or _open_sibling_variable("sp")
            ps = _align_time_to_ds(ps)
            if ps is not None:
                _ps = _drop_length1_non_time_dims(ps)
                _ps = _order_time_lat_lon(_ps)
                # Rename to PRESsfc
                ds2["PRESsfc"] = _ps.assign_attrs(description="Surface pressure (from ERA5 sp)")
            else:
                logging.debug(f"Cannot add PRESsfc in {file_path}: missing sp")

        # TOA longwave up: -mtnlwrf
        if "ULWRFtoa" in requested_names and ("ULWRFtoa" not in ds2.data_vars):
            src = ds2.data_vars.get("mtnlwrf")
            if src is None:
                src = _open_sibling_variable("mtnlwrf")
            if src is not None:
                _src = _drop_length1_non_time_dims(src)
                res = (-_src)
                res = _drop_length1_non_time_dims(res)
                res = _order_time_lat_lon(res)
                ds2["ULWRFtoa"] = res.assign_attrs(description="Upward longwave flux at TOA", source="mtnlwrf")
            else:
                logging.error(f"[error] Cannot compute ULWRFtoa in {file_path}: missing mtnlwrf")
                raise ValueError("Missing inputs for ULWRFtoa")

        # Surface shortwave up: avg_sdswrf - msnswrf
        if "USWRFsfc" in requested_names and ("USWRFsfc" not in ds2.data_vars):
            lhs = ds2.data_vars.get("avg_sdswrf")
            rhs = ds2.data_vars.get("msnswrf")
            if lhs is None:
                lhs = _open_sibling_variable("avg_sdswrf")
            # Additional fallbacks for downward SW at surface
            # Fallback: use ERA5 mean surface downward shortwave flux if local avg is missing
            if lhs is None:
                lhs = ds2.data_vars.get("msdwswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("msdwswrf")
            # Fallback: try total downward shortwave flux variants if available
            if lhs is None:
                lhs = ds2.data_vars.get("avg_tdswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("avg_tdswrf")
            if lhs is None:
                lhs = ds2.data_vars.get("tdswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("tdswrf")
            if rhs is None:
                rhs = _open_sibling_variable("msnswrf")
            lhs = _align_time_to_ds(lhs)
            rhs = _align_time_to_ds(rhs)
            if lhs is not None and rhs is not None:
                _lhs_name = getattr(lhs, "name", "avg_sdswrf")
                _rhs_name = getattr(rhs, "name", "msnswrf")
                _lhs = _drop_length1_non_time_dims(lhs)
                _rhs = _drop_length1_non_time_dims(rhs)
                res = (_lhs - _rhs)
                res = _drop_length1_non_time_dims(res)
                res = _order_time_lat_lon(res)
                ds2["USWRFsfc"] = res.assign_attrs(description="Upward shortwave flux at surface", source=f"{_lhs_name},{_rhs_name}")
            else:
                logging.error(
                    f"[error] Cannot compute USWRFsfc in {file_path}: missing "
                    f"{'avg_sdswrf/msdwswrf/avg_tdswrf/tdswrf' if lhs is None else ''} "
                    f"{'msnswrf' if rhs is None else ''}"
                )
                raise ValueError("Missing inputs for USWRFsfc")

        # Surface longwave up: avg_sdlwrf - msnlwrf
        if "ULWRFsfc" in requested_names and ("ULWRFsfc" not in ds2.data_vars):
            lhs = ds2.data_vars.get("avg_sdlwrf")
            rhs = ds2.data_vars.get("msnlwrf")
            if lhs is None:
                lhs = _open_sibling_variable("avg_sdlwrf")
            # Fallback: use ERA5 mean surface downward longwave flux if local avg is missing
            if lhs is None:
                lhs = ds2.data_vars.get("msdwlwrf")
                if lhs is None:
                    lhs = _open_sibling_variable("msdwlwrf")
            if rhs is None:
                rhs = _open_sibling_variable("msnlwrf")
            lhs = _align_time_to_ds(lhs)
            rhs = _align_time_to_ds(rhs)
            if lhs is not None and rhs is not None:
                _lhs_name = getattr(lhs, "name", "avg_sdlwrf")
                _rhs_name = getattr(rhs, "name", "msnlwrf")
                _lhs = _drop_length1_non_time_dims(lhs)
                _rhs = _drop_length1_non_time_dims(rhs)
                res = (_lhs - _rhs)
                res = _drop_length1_non_time_dims(res)
                res = _order_time_lat_lon(res)
                ds2["ULWRFsfc"] = res.assign_attrs(description="Upward longwave flux at surface", source=f"{_lhs_name},{_rhs_name}")
            else:
                logging.error(
                    f"[error] Cannot compute ULWRFsfc in {file_path}: missing "
                    f"{'avg_sdlwrf/msdwlwrf' if lhs is None else ''} "
                    f"{'msnlwrf' if rhs is None else ''}"
                )
                raise ValueError("Missing inputs for ULWRFsfc")

        # 2m humidity from dewpoint & surface pressure (Q2m)
        if "Q2m" in requested_names and ("Q2m" not in ds2.data_vars):
            d2m = ds2.data_vars.get("d2m")
            ps = ds2.data_vars.get("PRESsfc")
            if d2m is None:
                d2m = _open_sibling_variable("d2m")
            if ps is None:
                # Load raw sp then use rename behavior (PRESsfc)
                ps = _open_sibling_variable("sp")
                if ps is not None:
                    ps = ps.rename("PRESsfc") if getattr(ps, "name", None) else ps
            # Ensure both have the same time coordinate as ds2 to avoid dtype alignment issues
            d2m = _align_time_to_ds(d2m)
            ps = _align_time_to_ds(ps)
            if d2m is not None and ps is not None:
                d2m = _drop_length1_non_time_dims(d2m)
                ps = _drop_length1_non_time_dims(ps)
                Tc = d2m - 273.15
                es_hPa = xr.where(
                    Tc >= 0.0,
                    6.112 * np.exp(17.62 * Tc / (243.12 + Tc)),
                    6.112 * np.exp(22.46 * Tc / (272.62 + Tc)),
                )
                e = es_hPa * 100.0
                eps = 0.622
                q2m = (eps * e) / (ps - (1.0 - eps) * e)
                q2m = _drop_length1_non_time_dims(q2m)
                q2m = _order_time_lat_lon(q2m)
                ds2["Q2m"] = q2m.assign_attrs(description="2m specific humidity from d2m, PRESsfc")
            else:
                logging.error(f"[error] Cannot compute Q2m in {file_path}: missing {'d2m' if d2m is None else ''} {'PRESsfc/sp' if ps is None else ''}")
                raise ValueError("Missing inputs for Q2m")

        # Sea ice fraction from ERA5 sea ice concentration (siconc)
        if "sea_ice_fraction" in requested_names and ("sea_ice_fraction" not in ds2.data_vars):
            src = ds2.data_vars.get("siconc")
            if src is None:
                src = _open_sibling_variable("siconc")
            src = _align_time_to_ds(src)
            if src is not None:
                _src = _drop_length1_non_time_dims(src)
                _src = _order_time_lat_lon(_src)
                ds2["sea_ice_fraction"] = _src.assign_attrs(description="Sea ice fraction (from siconc)")
            else:
                logging.debug(f"Cannot add sea_ice_fraction in {file_path}: missing siconc")

        # Downward shortwave at surface: DSWRFsfc from avg_sdswrf or fallbacks
        if "DSWRFsfc" in requested_names and ("DSWRFsfc" not in ds2.data_vars):
            lhs = ds2.data_vars.get("avg_sdswrf")
            if lhs is None:
                lhs = _open_sibling_variable("avg_sdswrf")
            if lhs is None:
                lhs = ds2.data_vars.get("msdwswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("msdwswrf")
            if lhs is None:
                lhs = ds2.data_vars.get("avg_tdswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("avg_tdswrf")
            if lhs is None:
                lhs = ds2.data_vars.get("tdswrf")
                if lhs is None:
                    lhs = _open_sibling_variable("tdswrf")
            lhs = _align_time_to_ds(lhs)
            if lhs is not None:
                _lhs = _drop_length1_non_time_dims(lhs)
                _lhs = _order_time_lat_lon(_lhs)
                ds2["DSWRFsfc"] = _lhs.assign_attrs(description="Downward shortwave flux at surface")
            else:
                logging.debug(f"Cannot add DSWRFsfc in {file_path}: missing downward SW inputs")

        # Downward longwave at surface: DLWRFsfc from avg_sdlwrf or fallbacks
        if "DLWRFsfc" in requested_names and ("DLWRFsfc" not in ds2.data_vars):
            lhs = ds2.data_vars.get("avg_sdlwrf")
            if lhs is None:
                lhs = _open_sibling_variable("avg_sdlwrf")
            if lhs is None:
                lhs = ds2.data_vars.get("msdwlwrf")
                if lhs is None:
                    lhs = _open_sibling_variable("msdwlwrf")
            lhs = _align_time_to_ds(lhs)
            if lhs is not None:
                _lhs = _drop_length1_non_time_dims(lhs)
                _lhs = _order_time_lat_lon(_lhs)
                ds2["DLWRFsfc"] = _lhs.assign_attrs(description="Downward longwave flux at surface")
            else:
                logging.debug(f"Cannot add DLWRFsfc in {file_path}: missing downward LW inputs")

        # Terrain height HGTsfc from z_over_g0 or z (single-level) as fallback
        if "HGTsfc" in requested_names and ("HGTsfc" not in ds2.data_vars):
            src = ds2.data_vars.get("z_over_g0")
            if src is None:
                src = _open_sibling_variable("z_over_g0")
            if src is None:
                z = ds2.data_vars.get("z")
                if z is None:
                    z = _open_sibling_variable("z")
                if z is not None:
                    try:
                        src = (z / G0)
                    except Exception:
                        src = None
            src = _align_time_to_ds(src)
            if src is not None:
                _src = _drop_length1_non_time_dims(src)
                _src = _order_time_lat_lon(_src)
                ds2["HGTsfc"] = _src.assign_attrs(
                    description="Geopotential height at surface (fallback)",
                )
            else:
                logging.debug(f"Cannot add HGTsfc in {file_path}: missing z_over_g0/z sources")

        # 4) Expand multi-level (t,q,u,v,z) to per-level variables if present and not already expanded
        # Multi-level expansion: also support loading missing bases from sibling folders
        expansion = {
            "t": "air_temperature_{}",
            "q": "specific_total_water_{}",
            "u": "eastward_wind_{}",
            "v": "northward_wind_{}",
            # geopotential to geopotential height per level if desired by user logic
            # Provided here as 'geopotential_{}' if needed downstream
            "z": "geopotential_{}",
        }
        # Determine requested level indices per base
        requested_levels: dict[str, set[int]] = {"t": set(), "q": set(), "u": set(), "v": set(), "z": set()}
        for name in getattr(self, "_names", []):
            for base, tmpl in expansion.items():
                prefix = tmpl.split("{}")[0]
                if name.startswith(prefix):
                    try:
                        idx = int(name.rsplit("_", 1)[-1])
                        requested_levels[base].add(idx)
                    except Exception:
                        pass
        # Try to expand exactly the requested levels if specified; otherwise use available
        for base, tmpl in expansion.items():
            var_da = None
            base_level_dim = None
            if base in ds2.data_vars:
                var_da = ds2[base]
                for cand in ("lev", "level", "plev", "p"):
                    if cand in var_da.dims:
                        base_level_dim = cand
                        break
            if var_da is None or base_level_dim is None:
                sib = _open_sibling_variable(base)
                if sib is not None:
                    # Normalize sibling time and spatial coordinates to current dataset
                    sib = _align_time_to_ds(sib)
                    sib = _drop_length1_non_time_dims(sib)
                    sib = _order_time_lat_lon(sib)
                    sib = _coerce_spatial_coords_to_ds(sib)
                    var_da = sib
                    for cand in ("lev", "level", "plev", "p"):
                        if cand in var_da.dims:
                            base_level_dim = cand
                            break
            if var_da is None or base_level_dim is None:
                continue
            nlev = var_da.sizes[base_level_dim]
            req = sorted(requested_levels[base])
            if req:
                if any(i >= nlev for i in req):
                    logging.error(f"[error] {file_path or '<ds>'}: {base} has {nlev} levels but requested indices {req}.")
                    raise ValueError(f"Multi-level variable '{base}' missing requested indices.")
                idxs = req
            else:
                # if no specific request, use all available but cap at 13 for speed
                idxs = list(range(min(nlev, 13)))
            # Vectorized slice then split, instead of looping isel per level
            try:
                stacked = var_da.isel({base_level_dim: idxs})
                # Split along level axis and assign
                for i, level_i in enumerate(idxs):
                    name_i = tmpl.format(level_i)
                    if name_i not in ds2.data_vars:
                        try:
                            ds2[name_i] = stacked.isel({base_level_dim: i}, drop=True)
                        except TypeError:
                            ds2[name_i] = stacked.isel({base_level_dim: i}).squeeze()
            except Exception:
                # fallback to previous per-level loop if vectorized fails
                for level_i in idxs:
                    name_i = tmpl.format(level_i)
                    if name_i not in ds2.data_vars:
                        try:
                            ds2[name_i] = var_da.isel({base_level_dim: level_i}, drop=True)
                        except TypeError:
                            ds2[name_i] = var_da.isel({base_level_dim: level_i}).squeeze()

        # Fallback: ensure requested per-level variables exist for q/u/v by explicitly
        # loading siblings if they were not added above for any reason.
        try:
            requested_by_base: dict[str, list[int]] = {"q": [], "u": [], "v": []}
            for name in getattr(self, "_names", []):
                if name.startswith("specific_total_water_"):
                    try:
                        requested_by_base["q"].append(int(name.rsplit("_", 1)[-1]))
                    except Exception:
                        pass
                elif name.startswith("eastward_wind_"):
                    try:
                        requested_by_base["u"].append(int(name.rsplit("_", 1)[-1]))
                    except Exception:
                        pass
                elif name.startswith("northward_wind_"):
                    try:
                        requested_by_base["v"].append(int(name.rsplit("_", 1)[-1]))
                    except Exception:
                        pass
            for base, idxs in requested_by_base.items():
                if not idxs:
                    # If no explicit requests but no per-level fields present, expand defaults
                    tmpl = expansion[base]
                    any_present = any(n.startswith(tmpl.split("{}")[0]) for n in ds2.data_vars)
                    if any_present:
                        continue
                tmpl = expansion[base]
                # Identify which requested per-level variables are missing
                req = sorted(set(idxs)) if idxs else []
                if not req:
                    # default to first up to 13 levels
                    req = list(range(13))
                missing = [i for i in req if tmpl.format(i) not in ds2.data_vars]
                if not missing:
                    continue
                # Try sibling load directly
                sib = _open_sibling_variable(base)
                if sib is None:
                    logging.warning(
                        f"Failed to open sibling for base '{base}' to add missing per-level variables: {missing}"
                    )
                    continue
                # Align sibling time to current dataset and normalize dims order
                sib = _align_time_to_ds(sib)
                sib = _drop_length1_non_time_dims(sib)
                sib = _order_time_lat_lon(sib)
                sib = _coerce_spatial_coords_to_ds(sib)
                base_level_dim = None
                for cand in ("lev", "level", "plev", "p"):
                    if cand in sib.dims:
                        base_level_dim = cand
                        break
                if base_level_dim is None:
                    logging.warning(
                        f"Sibling for base '{base}' lacks a recognized level dimension; cannot expand {missing}"
                    )
                    continue
                nlev = sib.sizes[base_level_dim]
                for i in missing:
                    if i >= nlev:
                        logging.warning(
                            f"Requested {base} level {i} but sibling has only {nlev} levels"
                        )
                        continue
                    name_i = tmpl.format(i)
                    try:
                        ds2[name_i] = sib.isel({base_level_dim: i}, drop=True)
                    except TypeError:
                        ds2[name_i] = sib.isel({base_level_dim: i}).squeeze()
                logging.info(
                    f"Added per-level fields from sibling for base '{base}': "
                    f"{[tmpl.format(i) for i in missing if tmpl.format(i) in ds2.data_vars]}"
                )
        except Exception:
            pass
        # Drop original base multi-level variables to avoid leaking the 'lev' dimension
        # into downstream shape inference and broadcasting logic.
        try:
            dropped_bases = []
            for base in ("t", "q", "u", "v", "z"):
                if base in ds2.data_vars:
                    ds2 = ds2.drop_vars(base)
                    dropped_bases.append(base)
            if dropped_bases:
                logging.debug(
                    f"Dropped base multi-level variables after expansion: {dropped_bases}"
                )
        except Exception:
            pass

        # 4b) Single-level geopotential z -> HGTsfc (height) if needed
        if "HGTsfc" not in ds2.data_vars and "z" in ds2.data_vars:
            try:
                dims = ds2["z"].dims
                # if z has a level dimension, skip here (handled above); else compute height
                if not any(d in ("lev", "level", "plev", "p") for d in dims):
                    ds2["HGTsfc"] = (ds2["z"] / G0).assign_attrs(
                        units="m", description="Geopotential height at surface (z/g0)")
            except Exception:
                pass

        # 5) Add global_mean_co2 from annual means CSV, broadcast over time
        # CSV path: project-local annual means
        if "global_mean_co2" not in ds2.data_vars and "time" in ds2.coords:
            try:
                years = ds2["time"].dt.year.values
            except Exception:
                years = None
            if years is not None:
                try:
                    co2_vals = self._lookup_co2_by_year(years)
                    co2_da = xr.DataArray(co2_vals, dims=["time"]).assign_attrs(
                        units="ppm", long_name="global mean CO2 concentration"
                    )
                    # Broadcast to (time, lat, lon) to avoid downstream broadcasting limitations
                    lat_cands = ["lat", "latitude", "grid_yt", "y"]
                    lon_cands = ["lon", "longitude", "grid_xt", "x"]
                    ds_lat = next((d for d in lat_cands if d in ds2.dims), None)
                    ds_lon = next((d for d in lon_cands if d in ds2.dims), None)
                    if ds_lat and ds_lon:
                        co2_da = co2_da.expand_dims({ds_lat: ds2.sizes[ds_lat], ds_lon: ds2.sizes[ds_lon]}).transpose("time", ds_lat, ds_lon)
                    ds2["global_mean_co2"] = co2_da
                except Exception:
                    pass

        # 6) Add land_fraction and ocean_fraction from static LSM NetCDF, broadcast over time
        if ("land_fraction" not in ds2.data_vars or "ocean_fraction" not in ds2.data_vars) and "time" in ds2.dims:
            try:
                lsm_paths = [
                    "/home/yjlee/ace-main/Ace_data_construction/lsm_180x360.nc",
                ]
                lsm_path = None
                import os as _os
                for p in lsm_paths:
                    if _os.path.exists(p):
                        lsm_path = p
                        break
                if lsm_path is not None:
                    lsm_ds = xr.open_dataset(
                        lsm_path,
                        decode_times=False,
                        decode_timedelta=False,
                        engine=self.engine,
                        chunks=None,
                    )
                    # pick var
                    var_name = None
                    if "land_fraction" in lsm_ds.data_vars:
                        var_name = "land_fraction"
                    elif "lsm" in lsm_ds.data_vars:
                        var_name = "lsm"
                    elif len(lsm_ds.data_vars) == 1:
                        var_name = list(lsm_ds.data_vars)[0]
                    if var_name is not None:
                        lsm_da = lsm_ds[var_name]
                        # Drop singleton time dimension if present to allow broadcast later
                        try:
                            if "time" in lsm_da.dims and lsm_da.sizes.get("time", 0) == 1:
                                try:
                                    lsm_da = lsm_da.isel(time=0, drop=True)
                                except Exception:
                                    lsm_da = lsm_da.squeeze("time", drop=True)
                        except Exception:
                            pass
                        # detect lat/lon dims
                        lat_cands = ["lat", "latitude", "grid_yt", "y"]
                        lon_cands = ["lon", "longitude", "grid_xt", "x"]
                        ds_lat = next((d for d in lat_cands if d in ds2.dims), None)
                        ds_lon = next((d for d in lon_cands if d in ds2.dims), None)
                        lsm_lat = next((d for d in lat_cands if d in lsm_da.dims), None)
                        lsm_lon = next((d for d in lon_cands if d in lsm_da.dims), None)
                        if ds_lat and ds_lon and lsm_lat and lsm_lon:
                            if (lsm_lat != ds_lat) or (lsm_lon != ds_lon):
                                lsm_da = lsm_da.rename({lsm_lat: ds_lat, lsm_lon: ds_lon})
                            # Force coordinates to match exactly to avoid label-based reindex -> NaNs
                            try:
                                if ds2.sizes.get(ds_lat) == lsm_da.sizes.get(ds_lat) and ds2.sizes.get(ds_lon) == lsm_da.sizes.get(ds_lon):
                                    lsm_da = lsm_da.assign_coords({ds_lat: ds2[ds_lat], ds_lon: ds2[ds_lon]})  # type: ignore
                            except Exception:
                                pass
                            # broadcast over time
                            land_da = lsm_da.expand_dims({"time": ds2.sizes["time"]}).assign_coords(time=ds2["time"]).clip(0.0, 1.0)
                            if "land_fraction" not in ds2.data_vars:
                                ds2["land_fraction"] = land_da
                            if "ocean_fraction" not in ds2.data_vars:
                                ds2["ocean_fraction"] = (1.0 - land_da)
                    lsm_ds.close()
            except Exception:
                pass

        # Soft validations and diagnostics when names are known
        req = set(getattr(self, "_names", []))
        if ("land_fraction" in req) and ("land_fraction" not in ds2.data_vars):
            logging.debug(f"LSM not added in this dataset: {file_path}")
        if ("ocean_fraction" in req) and ("ocean_fraction" not in ds2.data_vars):
            logging.debug(f"LSM not added in this dataset: {file_path}")
        if ("global_mean_co2" in req) and ("global_mean_co2" not in ds2.data_vars):
            logging.debug(f"CO2 not added in this dataset: {file_path}")

        # Diagnostic: list injected per-level and mask variables with their dims (sample only)
        try:
            import itertools as _it
            check_prefixes = ("air_temperature_", "specific_total_water_", "eastward_wind_", "northward_wind_")
            per_level_names = [n for n in ds2.data_vars if any(n.startswith(p) for p in check_prefixes)]
            masks = [n for n in ("land_fraction", "ocean_fraction") if n in ds2.data_vars]
            sample_list = []
            for n in _it.islice(per_level_names, 0, 8):
                try:
                    sample_list.append((n, tuple(ds2[n].dims)))
                except Exception:
                    pass
            if sample_list:
                logging.debug(f"Per-level fields present (sample): {sample_list}")
            if masks:
                logging.debug(
                    f"Mask variables present: {[ (m, tuple(ds2[m].dims)) for m in masks ]}"
                )
        except Exception:
            pass

        try:
            logging.debug(
                f"[Preprocess] Done file={file_path} vars={len(ds2.data_vars)} time_ms={int((_time.time()-_tpre)*1000)}"
            )
        except Exception:
            pass
        return ds2

    # Cache CO2 loading once per process
    _CO2_CSV_PATH = "/home/yjlee/ace-main/Ace_data_construction/co2_annmean.csv"
    _co2_series_cache: pd.Series | None = None

    @classmethod
    def _get_co2_series(cls) -> pd.Series:
        if cls._co2_series_cache is not None:
            return cls._co2_series_cache
        df = pd.read_csv(cls._CO2_CSV_PATH)
        # Find year column
        year_col = None
        for c in df.columns:
            if str(c).strip().lower() in {"year", "yyyy"}:
                year_col = c
                break
        if year_col is None:
            # fallback: first int-like column
            for c in df.columns:
                if np.issubdtype(df[c].dtype, np.integer):
                    year_col = c
                    break
        if year_col is None:
            raise ValueError("co2_annmean.csv missing a year column")
        # Find value column
        value_col = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if c == year_col:
                continue
            if any(k in cl for k in ["co2", "value", "ppm"]):
                value_col = c
                break
        if value_col is None:
            # fallback: first non-year numeric column
            for c in df.columns:
                if c == year_col:
                    continue
                if np.issubdtype(df[c].dtype, np.number):
                    value_col = c
                    break
        if value_col is None:
            raise ValueError("co2_annmean.csv missing a numeric value column")
        s = pd.Series(df[value_col].to_numpy(), index=df[year_col].astype(int).to_numpy())
        cls._co2_series_cache = s
        return s

    @classmethod
    def _lookup_co2_by_year(cls, years: np.ndarray) -> np.ndarray:
        s = cls._get_co2_series()
        # Map exact years; if missing, attempt nearest within 1 year
        y = years.astype(int)
        vals = np.full_like(y, fill_value=np.nan, dtype=float)
        m = s.to_dict()
        for i, yy in enumerate(y):
            v = m.get(int(yy))
            if v is None:
                # nearest
                idx = (np.abs(s.index.values - yy)).argmin()
                v = float(s.iloc[idx])
            vals[i] = float(v)
        return vals


def _get_timestep(time: np.ndarray) -> datetime.timedelta:
    """Computes the timestep of an array of a time coordinate array.

    Raises an error if the times are not separated by a positive constant
    interval, or if the array has one or fewer times.
    """
    assert len(time.shape) == 1, "times must be a 1D array"

    if len(time) > 1:
        timesteps = np.diff(time)
        # Normalize timestep type
        timestep0 = timesteps[0]
        # Case 1: numpy timedelta64
        if np.issubdtype(getattr(timesteps, "dtype", np.array([timestep0]).dtype), np.timedelta64):
            zero = np.timedelta64(0, 'ns')
            if not np.all(timesteps > zero):
                raise ValueError("Timestep of data must be greater than zero.")
            if not np.all(timesteps == timesteps[0]):
                raise ValueError("Time coordinate does not have a uniform timestep.")
            # Convert to python timedelta (assume ns resolution)
            td_ns = timesteps[0].astype('timedelta64[ns]').astype(int)
            return datetime.timedelta(microseconds=td_ns / 1000)
        # Case 2: python datetime.timedelta
        if isinstance(timestep0, datetime.timedelta):
            if not (timestep0 > datetime.timedelta(days=0)):
                raise ValueError("Timestep of data must be greater than zero.")
            if not np.all([ts == timestep0 for ts in timesteps]):
                raise ValueError("Time coordinate does not have a uniform timestep.")
            return timestep0
        # Case 3: fall back to pandas to_timedelta if possible
        try:
            td_index = pd.to_timedelta(timesteps)
            if not (td_index > pd.Timedelta(0)).all():
                raise ValueError("Timestep of data must be greater than zero.")
            if not (td_index == td_index[0]).all():
                raise ValueError("Time coordinate does not have a uniform timestep.")
            return td_index[0].to_pytimedelta()
        except Exception:
            pass
        # Case 4: numeric increments (assume seconds)
        try:
            incr = float(timestep0)
            if not (incr > 0):
                raise ValueError("Timestep of data must be greater than zero.")
            if not np.allclose(timesteps, incr):
                raise ValueError("Time coordinate does not have a uniform timestep.")
            return datetime.timedelta(seconds=incr)
        except Exception:
            raise TypeError(
                f"Unsupported timestep type: {type(timestep0)}. Cannot infer uniform timestep."
            )
    else:
        raise ValueError(
            "Time coordinate does not have enough times to infer a timestep."
        )
