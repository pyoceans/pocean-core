#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd

from pocean import logger as L  # noqa
from pocean.cf import cf_safe_name, CFDataset
from pocean.dsg.trajectory import trajectory_calculated_metadata
from pocean.utils import (
    create_ncvar_from_series,
    dict_update,
    downcast_dataframe,
    generic_masked,
    get_default_axes,
    get_dtype,
    get_fill_value,
    get_mapped_axes_variables,
    get_masked_datetime_array,
    get_ncdata_from_series,
    nativize_times,
    normalize_array,
    normalize_countable_array,
)


class IncompleteMultidimensionalTrajectory(CFDataset):
    """
    When storing multiple trajectories in the same file, and the number of
    elements in each trajectory is the same, one can use the multidimensional
    array representation. This representation also allows one to have a
    variable number of elements in different trajectories, at the cost of some
    wasted space. In that case, any unused elements of the data and auxiliary
    coordinate variables must contain missing data values (section 9.6).
    """

    @classmethod
    def is_mine(cls, dsg, strict=False):
        try:
            tvars = dsg.filter_by_attrs(cf_role='trajectory_id')
            assert len(tvars) == 1
            assert dsg.featureType.lower() == 'trajectory'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            # Allow for string variables
            tvar = tvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(tvar.dimensions) <= 2

            ts = normalize_array(tvar)
            is_single = False

            if tvar.ndim == 0:
                is_single = True
            elif tvar.ndim == 2:
                is_single = False
            elif isinstance(ts, str):
                # Non-dimensioned string variable
                is_single = True
            elif tvar.ndim == 1 and hasattr(ts, 'dtype') and ts.dtype.kind in ['U', 'S']:
                is_single = True

            t = dsg.t_axes()[0]
            x = dsg.x_axes()[0]
            y = dsg.y_axes()[0]
            z = dsg.z_axes()[0]

            assert t.dimensions == x.dimensions == y.dimensions == z.dimensions
            assert t.size == x.size == y.size == z.size

            if is_single:
                assert len(t.dimensions) == 1
                t_dim = dsg.dimensions[t.dimensions[0]]
                for dv in dsg.data_vars():
                    assert len(dv.dimensions) == 1
                    assert t_dim.name in dv.dimensions
                    assert dv.size == t_dim.size
            else:
                # This `time` being two dimensional is unique to IncompleteMultidimensionalTrajectory
                assert len(t.dimensions) == 2
                t_dim = dsg.dimensions[t.dimensions[0]]
                o_dim = dsg.dimensions[t.dimensions[1]]
                for dv in dsg.data_vars():
                    assert dv.size == t.size
                    assert len(dv.dimensions) == 2
                    assert t_dim.name in dv.dimensions
                    assert o_dim.name in dv.dimensions
                    assert dv.size == t_dim.size * o_dim.size

        except BaseException:
            if strict is True:
                raise
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        daxes = axes
        data_columns = [ d for d in df.columns if d not in axes ]

        reduce_dims = kwargs.pop('reduce_dims', False)
        unlimited = kwargs.pop('unlimited', False)

        unique_dims = kwargs.pop('unique_dims', False)
        if unique_dims is True:
            # Rename the dimension to avoid a dimension and coordinate having the same name
            # which is not support in xarray
            changed_axes = { k: f'{v}_dim' for k, v in axes._asdict().items() }
            daxes = get_default_axes(changed_axes)

        # Downcast anything from int64 to int32
        # Convert any timezone aware datetimes to native UTC times
        df = downcast_dataframe(nativize_times(df))

        with IncompleteMultidimensionalTrajectory(output, 'w') as nc:

            trajectory_group = df.groupby(axes.trajectory)

            if unlimited is True:
                max_obs = None
            else:
                max_obs = trajectory_group.size().max()
            nc.createDimension(daxes.sample, max_obs)

            num_trajectories = len(trajectory_group)
            if reduce_dims is True and num_trajectories == 1:
                # If a singular trajectory, we can reduce that dimension if it is of size 1
                def ts(t_index, size):
                    return np.s_[0:size]
                default_dimensions = (daxes.sample,)
                trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]))
            else:
                def ts(t_index, size):
                    return np.s_[t_index, 0:size]
                default_dimensions = (daxes.trajectory, daxes.sample)
                nc.createDimension(daxes.trajectory, num_trajectories)
                trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]), (daxes.trajectory,))

            # Create all of the variables
            time = nc.createVariable(axes.t, 'f8', default_dimensions, fill_value=np.dtype('f8').type(cls.default_fill_value))
            z = nc.createVariable(axes.z, get_dtype(df[axes.z]), default_dimensions, fill_value=df[axes.z].dtype.type(cls.default_fill_value))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), default_dimensions, fill_value=df[axes.y].dtype.type(cls.default_fill_value))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), default_dimensions, fill_value=df[axes.x].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            # Create vars based on full dataframe (to get all variables)
            for c in data_columns:
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    v = create_ncvar_from_series(
                        nc,
                        var_name,
                        default_dimensions,
                        df[c],
                    )
                    attributes[var_name] = dict_update(attributes.get(var_name, {}), {
                        'coordinates': '{} {} {} {}'.format(
                            axes.t, axes.z, axes.x, axes.y
                        )
                    })

            for i, (uid, gdf) in enumerate(trajectory_group):
                trajectory[i] = uid

                times = get_ncdata_from_series(gdf[axes.t], time).astype('f8')
                time[ts(i, times.size)] = times

                lats = get_ncdata_from_series(gdf[axes.y], latitude)
                latitude[ts(i, lats.size)] = lats

                lons = get_ncdata_from_series(gdf[axes.x], longitude)
                longitude[ts(i, lons.size)] = lons

                zs = gdf[axes.z].fillna(get_fill_value(z)).values
                z[ts(i, zs.size)] = zs

                for c in data_columns:
                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    v = nc.variables[var_name]

                    vvalues = get_ncdata_from_series(gdf[c], v)
                    slicer = ts(i, vvalues.size)
                    v[slicer] = vvalues

            # Metadata variables
            if 'crs' not in nc.variables:
                nc.createVariable('crs', 'i4')

            # Set attributes
            nc.update_attributes(attributes)

        return IncompleteMultidimensionalTrajectory(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        return trajectory_calculated_metadata(df, axes, geometries)

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes, skip=[axes.profile, axes.station])

        # T
        t = get_masked_datetime_array(axv.t[:], axv.t).flatten()

        # X
        x = generic_masked(axv.x[:], attrs=self.vatts(axv.x.name)).flatten()

        # Y
        y = generic_masked(axv.y[:], attrs=self.vatts(axv.y.name)).flatten()

        # Z
        z = generic_masked(axv.z[:], attrs=self.vatts(axv.z.name)).flatten()

        # Trajectories
        rvar = axv.trajectory
        p = normalize_countable_array(rvar)

        # The Dimension that the trajectory id variable doesn't have is what
        # the trajectory data needs to be repeated by
        dim_diff = self.dimensions[list(set(axv.t.dimensions).difference(set(rvar.dimensions)))[0]]
        if dim_diff:
            p = p.repeat(dim_diff.size)

        df_data = OrderedDict([
            (axes.t, t),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.trajectory, p)
        ])

        building_index_to_drop = np.ones(t.size, dtype=bool)

        # Axes variables are already processed so skip them
        extract_vars = copy(self.variables)
        for ncvar in axv._asdict().values():
            if ncvar is not None and ncvar.name in extract_vars:
                del extract_vars[ncvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):
            vdata = generic_masked(dvar[:].flatten().astype(dvar.dtype), attrs=self.vatts(dnam))

            # Carry through size 1 variables
            if vdata.size == 1:
                if vdata[0] is np.ma.masked:
                    L.warning(f"Skipping variable {dnam} that is completely masked")
                    continue
            else:
                if dvar[:].flatten().size != t.size:
                    L.warning(f"Variable {dnam} is not the correct size, skipping.")
                    continue

            # Mark rows with data so we don't remove them with clear_rows
            if vdata.size == building_index_to_drop.size:
                building_index_to_drop = (building_index_to_drop == True) & (vdata.mask == True)  # noqa

            # Handle scalars here at the end
            if vdata.size == 1:
                vdata = vdata[0]

            df_data[dnam] = vdata

        df = pd.DataFrame(df_data)

        # Drop all data columns with no data
        if clean_cols:
            df = df.dropna(axis=1, how='all')

        # Drop all data rows with no data variable data
        if clean_rows:
            df = df.iloc[~building_index_to_drop]

        return df

    def nc_attributes(self, axes, daxes):
        atts = super().nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'trajectory',
                'cdm_data_type': 'Trajectory'
            },
            axes.trajectory : {
                'cf_role': 'trajectory_id',
                'long_name' : 'trajectory identifier'
            },
            axes.x: {
                'axis': 'X'
            },
            axes.y: {
                'axis': 'Y'
            },
            axes.z: {
                'axis': 'Z'
            },
            axes.t: {
                'units': self.default_time_unit,
                'standard_name': 'time',
                'axis': 'T'
            }
        })
