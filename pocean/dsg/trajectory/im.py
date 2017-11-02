#!python
# coding=utf-8
from collections import OrderedDict

import six
import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    dict_update,
    generic_masked,
    get_default_axes,
    get_dtype,
    get_masked_datetime_array,
    normalize_array,
)
from pocean.cf import CFDataset, cf_safe_name
from pocean.dsg.trajectory import trajectory_calculated_metadata

from pocean import logger as L  # noqa


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
    def is_mine(cls, dsg):
        try:
            tvars = dsg.filter_by_attrs(cf_role='trajectory_id')
            assert len(tvars) == 1
            assert dsg.featureType.lower() == 'trajectory'

            assert len(dsg.t_axes()) == 1
            assert len(dsg.x_axes()) == 1
            assert len(dsg.y_axes()) == 1
            assert len(dsg.z_axes()) == 1

            # Allow for string variables
            tvar = tvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(tvar.dimensions) <= 2

            ts = normalize_array(tvar)
            is_single = False

            if isinstance(ts, six.string_types):
                # Non-dimensioned string variable
                is_single = True
            elif ts.size == 1 and tvar.dtype != str:
                # Other non-string types
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
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        data_columns = [ d for d in df.columns if d not in axes ]

        reduce_dims = kwargs.pop('reduce_dims', False)
        unlimited = kwargs.pop('unlimited', False)

        with IncompleteMultidimensionalTrajectory(output, 'w') as nc:

            trajectory_group = df.groupby(axes.trajectory)

            if unlimited is True:
                max_obs = None
            else:
                max_obs = trajectory_group.size().max()
            nc.createDimension('obs', max_obs)

            unique_trajectories = df[axes.trajectory].unique()
            if reduce_dims is True and len(unique_trajectories) == 1:
                # If a singlular trajectory, we can reduce that dimension if it is of size 1
                def ts(t_index, size):
                    return np.s_[0:size]
                default_dimensions = ('obs',)
                trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]))
            else:
                def ts(t_index, size):
                    return np.s_[t_index, 0:size]
                default_dimensions = (axes.trajectory, 'obs')
                nc.createDimension(axes.trajectory, unique_trajectories.size)
                trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]), (axes.trajectory,))

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            time = nc.createVariable(axes.t, 'f8', default_dimensions, fill_value=np.dtype('f8').type(cls.default_fill_value))
            z = nc.createVariable(axes.z, get_dtype(df[axes.z]), default_dimensions, fill_value=df[axes.z].dtype.type(cls.default_fill_value))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), default_dimensions, fill_value=df[axes.y].dtype.type(cls.default_fill_value))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), default_dimensions, fill_value=df[axes.x].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes), kwargs.pop('attributes', {}))

            for i, (uid, gdf) in enumerate(trajectory_group):
                trajectory[i] = uid

                # tolist() converts to a python datetime object without timezone and has NaTs.
                g = gdf[axes.t].tolist()
                # date2num convers NaTs to np.nan
                gg = nc4.date2num(g, units=cls.default_time_unit)
                # masked_invalid moves np.nan to a masked value
                time[ts(i, gg.size)] = np.ma.masked_invalid(gg)

                lats = gdf[axes.y].fillna(latitude._FillValue).values
                latitude[ts(i, lats.size)] = lats

                lons = gdf[axes.x].fillna(longitude._FillValue).values
                longitude[ts(i, lons.size)] = lons

                zs = gdf[axes.z].fillna(z._FillValue).values
                z[ts(i, zs.size)] = zs

                for c in data_columns:
                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        if np.issubdtype(gdf[c].dtype, 'S') or gdf[c].dtype == object:
                            # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                            v = nc.createVariable(var_name, get_dtype(gdf[c]), default_dimensions)
                        else:
                            v = nc.createVariable(var_name, get_dtype(gdf[c]), default_dimensions, fill_value=gdf[c].dtype.type(cls.default_fill_value))

                        if var_name not in attributes:
                            attributes[var_name] = {}
                        attributes[var_name] = dict_update(attributes[var_name], {
                            'coordinates' : '{} {} {} {}'.format(
                                axes.t, axes.z, axes.x, axes.y
                            )
                        })
                    else:
                        v = nc.variables[var_name]

                    if hasattr(v, '_FillValue'):
                        vvalues = gdf[c].fillna(v._FillValue).values
                    else:
                        # Use an empty string... better than nothing!
                        vvalues = gdf[c].fillna('').values

                    # Why do we need a slice object?
                    # sl = slice(0, vvalues.size)
                    v[ts(i, vvalues.size)] = vvalues

            # Set global attributes
            nc.update_attributes(attributes)

        return IncompleteMultidimensionalTrajectory(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        return trajectory_calculated_metadata(df, axes, geometries)

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        # Z
        zvar = self.z_axes()[0]
        z = generic_masked(zvar[:], attrs=self.vatts(zvar.name)).flatten()
        L.debug(['z data size: ', z.size])

        # T
        tvar = self.t_axes()[0]
        t = get_masked_datetime_array(tvar[:], tvar).flatten()
        L.debug(['time data size: ', t.size])

        # X
        xvar = self.x_axes()[0]
        x = generic_masked(xvar[:], attrs=self.vatts(xvar.name)).flatten()
        L.debug(['x data size: ', x.size])

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:], attrs=self.vatts(yvar.name)).flatten()
        L.debug(['y data size: ', y.size])

        # Trajectories
        pvar = self.filter_by_attrs(cf_role='trajectory_id')[0]
        try:
            p = normalize_array(pvar)
            if isinstance(p, six.string_types):
                p = np.asarray([p])
        except BaseException:
            L.exception('Could not pull trajectory values from the variable, using indexes.')
            p = np.asarray(list(range(len(pvar))), dtype=np.integer)

        # The Dimension that the trajectory id variable doesn't have is what
        # the trajectory data needs to be repeated by
        dim_diff = self.dimensions[list(set(tvar.dimensions).difference(set(pvar.dimensions)))[0]]
        if dim_diff:
            p = p.repeat(dim_diff.size)
        L.debug(['trajectory data size: ', p.size])

        df_data = OrderedDict([
            (axes.t, t),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.trajectory, p)
        ])

        building_index_to_drop = np.ones(t.size, dtype=bool)
        extract_vars = list(set(self.data_vars() + self.ancillary_vars()))
        for i, dvar in enumerate(extract_vars):
            vdata = np.ma.fix_invalid(np.ma.MaskedArray(dvar[:].round(3).flatten()))
            building_index_to_drop = (building_index_to_drop == True) & (vdata.mask == True)  # noqa
            df_data[dvar.name] = vdata

        df = pd.DataFrame(df_data)

        # Drop all data columns with no data
        if clean_cols:
            df = df.dropna(axis=1, how='all')

        # Drop all data rows with no data variable data
        if clean_rows:
            df = df.iloc[~building_index_to_drop]

        return df

    def nc_attributes(self, axes):
        atts = super(IncompleteMultidimensionalTrajectory, self).nc_attributes()
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
