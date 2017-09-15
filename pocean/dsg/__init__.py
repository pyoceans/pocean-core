#!python
# coding=utf-8
from copy import copy

# Profile
from .profile.im import IncompleteMultidimensionalProfile
from .profile.om import OrthogonalMultidimensionalProfile

# Trajectory
from .trajectory.cr import ContiguousRaggedTrajectory
from .trajectory.ir import IndexedRaggedTrajectory
from .trajectory.im import IncompleteMultidimensionalTrajectory

# TrajectoryProfile
from .trajectoryProfile.cr import ContiguousRaggedTrajectoryProfile

# Timeseries
from .timeseries.cr import ContiguousRaggedTimeseries
from .timeseries.ir import IndexedRaggedTimeseries
from .timeseries.im import IncompleteMultidimensionalTimeseries
from .timeseries.om import OrthogonalMultidimensionalTimeseries

# TimeseriesProfile
from .timeseriesProfile.r import RaggedTimeseriesProfile
from .timeseriesProfile.im import IncompleteMultidimensionalTimeseriesProfile
from .timeseriesProfile.om import OrthogonalMultidimensionalTimeseriesProfile

from pocean.utils import dict_update
from pocean import logger as L


__all__ = [
    'IncompleteMultidimensionalProfile',
    'OrthogonalMultidimensionalProfile',
    'ContiguousRaggedTrajectory',
    'IndexedRaggedTrajectory',
    'IncompleteMultidimensionalTrajectory',
    'ContiguousRaggedTrajectoryProfile',
    'ContiguousRaggedTimeseries',
    'IndexedRaggedTimeseries',
    'IncompleteMultidimensionalTimeseries',
    'OrthogonalMultidimensionalTimeseries',
    'RaggedTimeseriesProfile',
    'IncompleteMultidimensionalTimeseriesProfile',
    'OrthogonalMultidimensionalTimeseriesProfile',
    'open_cfdsg_dataframe'
]


import xarray as xr


@xr.register_dataset_accessor('cfdsg')
class CfDsgAccessor(object):

    def __init__(self, xrobj):
        self._obj = xrobj

        # figure out the klass
        self._dsg_klass = IncompleteMultidimensionalProfile

    def update_attrs(self, meta_obj=None):

        if meta_obj is None:
            meta_obj = {}

        # Get dict metadata object
        dict_meta = self._dsg_klass.attrs(self._obj)

        # Override with any user supplied attributes
        dict_meta = dict_update(dict_meta, meta_obj)

        # Assign JSON metadata object via the 'jsmeta' dataset accessor
        self._obj.jsmeta.update(dict_meta)

    def reduce(self, reduce_map=None):

        saved_coords = [ v for v in self._obj.coords ]


        # Reset so we can repalce coordinate variables if need be
        self._obj.reset_coords(inplace=True)

        if reduce_map is None:
            reduce_map = {}

        reductions = dict_update(self._dsg_klass.reduce_variables(), reduce_map)
        L.info(reductions)

        # Reduce some variables to the correct dimensions
        for vname, dims in reductions.items():
            # Get the dims we want to get rid of
            reduce_dims = set(self._obj[vname].dims) - set(dims)

            # Reduce the unwanted dimension to zero
            to_remove = { x: 0 for x in reduce_dims}

            # Drop the dims and get the resulting Dataset
            t = self._obj[vname].isel(**to_remove, drop=True)
            t = t.to_dataset()

            # Merge with original Dataset, replacing the variable
            self._obj.update(t, inplace=True)

        # Add back in the coordinates
        self._obj.set_coords(saved_coords, inplace=True)

def open_cfdsg_dataframe(df, klass, flatten=False, axis_mapping=None):
    """Wrapper around xarray's open_dataset to load pandas DataFrames
    into CF DSG compliant xarray Datasets

    Parameters
    ----------
    df :
        Pandas dataframe
    klass:
        The CF DSG to interpret the dataframe as
    flatten :
        If the array should be flattened and re-indexed by the klass. Default: False
    axis_mapping:
        A mapping for axes. A dictionary with keys 't', 'z', 'x' and 'y' to defined
        what the axis names will be in the xarray Dataset.

    Returns
    -------
        xarray Dataset

    Raises
    ------
        ValueError:
            If no suitable klass is found for your dataset or it could not be converted
        KeyError:
            When a required column is not found in the DataFrame
    """

    axis_defaults = {
        't': 'time',
        'x': 'longitude',
        'y': 'latitude',
        'z': 'z',
    }
    axis_mapping = dict_update(axis_defaults, axis_mapping or {})
    coord_columns = list(axis_mapping.values())

    # Final list of required columns
    req_columns = klass.required_columns() + coord_columns
    L.info(req_columns)

    for rq in req_columns:
        if rq not in df and rq not in df.index.names:
            raise KeyError("Required column {} not found in DataFrame".format(rq))

    if flatten is True:
        df = klass.apply_indexes(df.reset_index())

    with xr.Dataset.from_dataframe(df) as ncd:
        ncd.set_coords(coord_columns, inplace=True)
        return ncd
