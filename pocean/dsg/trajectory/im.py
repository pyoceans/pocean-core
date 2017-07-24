#!python
# coding=utf-8
from collections import namedtuple, OrderedDict

import six
import numpy as np
import pandas as pd
import netCDF4 as nc4
from pygc import great_distance
from shapely.geometry import Point, LineString

from pocean.utils import unique_justseen, normalize_array, get_dtype, dict_update
from pocean.cf import CFDataset
from pocean.cf import cf_safe_name
from pocean import logger


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
            is_single = isinstance(ts, six.string_types) or ts.size == 1

            t = dsg.t_axes()[0]
            x = dsg.x_axes()[0]
            y = dsg.y_axes()[0]
            z = dsg.z_axes()[0]

            assert t.dimensions == x.dimensions == y.dimensions == z.dimensions
            assert t.size == x.size == y.size == z.size

            if is_single:
                assert len(t.dimensions) == 1
                time_dim = dsg.dimensions[t.dimensions[0]]
                for dv in dsg.data_vars():
                    assert len(dv.dimensions) == 1
                    assert time_dim.name in dv.dimensions
                    assert dv.size == time_dim.size
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
        reserved_columns = ['trajectory', 't', 'x', 'y', 'z', 'distance']
        data_columns = [ d for d in df.columns if d not in reserved_columns ]

        reduce_dims = kwargs.pop('reduce_dims', False)

        with IncompleteMultidimensionalTrajectory(output, 'w') as nc:

            trajectory_group = df.groupby('trajectory')
            max_obs = trajectory_group.size().max()

            nc.createDimension('obs', max_obs)

            unique_trajectories = df.trajectory.unique()
            if reduce_dims is True and len(unique_trajectories) == 1:
                # If a singlular trajectory, we can reduce that dimension if it is of size 1
                def ts(index):
                    return np.s_[:]
                default_dimensions = ('obs',)
                trajectory = nc.createVariable('trajectory', get_dtype(df.trajectory))
            else:
                def ts(t_index):
                    return np.s_[t_index, :]
                default_dimensions = ('trajectory', 'obs')
                nc.createDimension('trajectory', unique_trajectories.size)
                trajectory = nc.createVariable('trajectory', get_dtype(df.trajectory), ('trajectory',))

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            time = nc.createVariable('time', 'i4', default_dimensions, fill_value=int(cls.default_fill_value))
            z = nc.createVariable('z', get_dtype(df.z), default_dimensions, fill_value=df.z.dtype.type(cls.default_fill_value))
            latitude = nc.createVariable('latitude', get_dtype(df.y), default_dimensions, fill_value=df.y.dtype.type(cls.default_fill_value))
            longitude = nc.createVariable('longitude', get_dtype(df.x), default_dimensions, fill_value=df.x.dtype.type(cls.default_fill_value))
            if 'distance' in df:
                distance = nc.createVariable('distance', get_dtype(df.distance), default_dimensions, fill_value=df.distance.dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(), kwargs.pop('attributes', {}))

            for i, (uid, gdf) in enumerate(trajectory_group):
                trajectory[i] = uid

                # tolist() converts to a python datetime object without timezone
                g = gdf.t.fillna(999999).tolist()   # 999999 is a dummy value
                NaTs = gdf.t.isnull()
                timenums = np.ma.MaskedArray(nc4.date2num(g, units=cls.default_time_unit))
                timenums.mask = NaTs
                time[ts(i)] = timenums

                latitude[ts(i)] = gdf.y.fillna(latitude._FillValue).values
                longitude[ts(i)] = gdf.x.fillna(longitude._FillValue).values
                z[ts(i)] = gdf.z.fillna(z._FillValue).values
                if 'distance' in gdf:
                    distance[ts(i)] = gdf.distance.fillna(distance._FillValue).values

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
                            'coordinates' : 'time latitude longitude z',
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
                    v[ts(i)] = vvalues

            # Set global attributes
            nc.update_attributes(attributes)

        return IncompleteMultidimensionalTrajectory(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)

        trajectories = {}
        for tid, tgroup in df.groupby('trajectory'):
            tgroup = tgroup.sort_values('t')
            first_row = tgroup.iloc[0]
            first_loc = Point(first_row.x, first_row.y)

            geometry = None
            if geometries:
                null_coordinates = tgroup.x.isnull() | tgroup.y.isnull()
                coords = list(unique_justseen(zip(
                    tgroup.x[~null_coordinates].tolist(),
                    tgroup.y[~null_coordinates].tolist()
                )))
                if len(coords) > 1:
                    geometry = LineString(coords)
                elif coords == 1:
                    geometry = first_loc

            trajectory = namedtuple('Trajectory', ['min_z', 'max_z', 'min_t', 'max_t', 'first_loc', 'geometry'])
            trajectories[tid] = trajectory(
                min_z=tgroup.z.min(),
                max_z=tgroup.z.max(),
                min_t=tgroup.t.min(),
                max_t=tgroup.t.max(),
                first_loc=first_loc,
                geometry=geometry
            )

        meta = namedtuple('Metadata', ['min_t', 'max_t', 'trajectories'])
        return meta(
            min_t=df.t.min(),
            max_t=df.t.max(),
            trajectories=trajectories
        )

    def to_dataframe(self, clean_cols=True, clean_rows=True):
        # Z
        zvar = self.z_axes()[0]
        z = np.ma.fix_invalid(np.ma.MaskedArray(zvar[:]))
        z = z.flatten().round(5)
        logger.debug(['z data size: ', z.size])

        # T
        tvar = self.t_axes()[0]
        t = np.ma.MaskedArray(nc4.num2date(tvar[:], tvar.units, getattr(tvar, 'calendar', 'standard'))).flatten()
        # Patch the time variable back to its original mask, since num2date
        # breaks any missing/fill values
        if hasattr(tvar[0], 'mask'):
            t.mask = tvar[:].mask
        logger.debug(['time data size: ', t.size])

        # X
        xvar = self.x_axes()[0]
        x = np.ma.fix_invalid(np.ma.MaskedArray(xvar[:])).flatten().round(5)
        logger.debug(['x data size: ', x.size])

        # Y
        yvar = self.y_axes()[0]
        y = np.ma.fix_invalid(np.ma.MaskedArray(yvar[:])).flatten().round(5)
        logger.debug(['y data size: ', y.size])

        # Trajectories
        pvar = self.filter_by_attrs(cf_role='trajectory_id')[0]
        try:
            p = normalize_array(pvar)
            if isinstance(p, six.string_types):
                p = np.asarray([p])
        except BaseException:
            logger.exception('Could not pull trajectory values from the variable, using indexes.')
            p = np.asarray(list(range(len(pvar))), dtype=np.integer)

        # The Dimension that the trajectory id variable doesn't have is what
        # the trajectory data needs to be repeated by
        dim_diff = self.dimensions[list(set(tvar.dimensions).difference(set(pvar.dimensions)))[0]]
        if dim_diff:
            p = p.repeat(dim_diff.size)
        logger.debug(['trajectory data size: ', p.size])

        # Distance
        d = np.append([0], great_distance(start_latitude=y[0:-1], end_latitude=y[1:], start_longitude=x[0:-1], end_longitude=x[1:])['distance'])
        d = np.ma.fix_invalid(np.ma.MaskedArray(np.cumsum(d)).astype(np.float64).round(2))
        logger.debug(['distance data size: ', d.size])

        df_data = OrderedDict([
            ('t', t),
            ('x', x),
            ('y', y),
            ('z', z),
            ('trajectory', p),
            ('distance', d),
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

    def nc_attributes(self):
        atts = super(IncompleteMultidimensionalTrajectory, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'trajectory',
                'cdm_data_type': 'Trajectory'
            },
            'trajectory' : {
                'cf_role': 'trajectory_id',
                'long_name' : 'trajectory identifier'
            },
            'distance' : {
                'long_name': 'Great circle distance between trajectory points',
                'standard_name': 'distance_between_trajectory_points',
                'units': 'm'
            },
            'time' : {
                'axis': 'T'
            },
            'latitude' : {
                'axis': 'Y'
            },
            'longitude' : {
                'axis': 'X'
            },
            'z' : {
                'axis': 'Z'
            }
        })
