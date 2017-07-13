#!python
# coding=utf-8
from copy import copy
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    normalize_array,
    get_dtype,
    dict_update,
    generic_masked
)
from pocean.cf import CFDataset
from pocean.cf import cf_safe_name
from pocean import logger  # noqa


class OrthogonalMultidimensionalTimeseriesProfile(CFDataset):

    @classmethod
    def is_mine(cls, dsg):
        try:
            assert dsg.featureType.lower() == 'timeseriesprofile'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            # If there is only a single set of levels and a single set of
            # times, then it is orthogonal.
            tvar = dsg.t_axes()[0]
            assert len(tvar.dimensions) == 1

            zvar = dsg.z_axes()[0]
            assert len(zvar.dimensions) == 1

            assert tvar.dimensions != zvar.dimensions

            # Not ragged
            o_index_vars = dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )
            assert len(o_index_vars) == 0

            r_index_vars = dsg.filter_by_attrs(
                instance_dimension=lambda x: x is not None
            )
            assert len(r_index_vars) == 0

        except BaseException:
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        reserved_columns = ['station', 't', 'x', 'y', 'z']
        data_columns = [ d for d in df.columns if d not in reserved_columns ]

        with OrthogonalMultidimensionalTimeseriesProfile(output, 'w') as nc:
            time_group = df.groupby('t')
            station_group = df.groupby('station')
            z_group = df.groupby(['station', 't'])

            n_times = len(time_group)
            n_stations = len(station_group)
            # all time/stations have same number of zs
            _, zdf = list(z_group)[0]
            n_z = len(zdf)

            nc.createVariable('crs', 'i4')

            # create ortho dimensions
            nc.createDimension('station', n_stations)
            nc.createDimension('time', n_times)
            nc.createDimension('z', n_z)

            # create nondata variables
            station = nc.createVariable('station', get_dtype(df.station), ('station',))
            time = nc.createVariable('time', 'f8', ('time',))
            latitude = nc.createVariable('latitude', get_dtype(df.y), ('station',))
            longitude = nc.createVariable('longitude', get_dtype(df.x), ('station',))
            z = nc.createVariable('z', get_dtype(df.z), ('z',))

            attributes = dict_update(nc.nc_attributes(), kwargs.pop('attributes', {}))

            for itime, (t, tdf) in enumerate(time_group):
                time[itime] = nc4.date2num(t, units=cls.default_time_unit)

                ts_group = tdf.groupby('station')

                for istation, (s, sdf) in enumerate(ts_group):
                    station[istation] = s
                    latitude[istation] = sdf.y.iloc[0]
                    longitude[istation] = sdf.x.iloc[0]
                    # assume z is all same length, FIXME don't repeat assignment
                    z[:] = np.array(sdf.z)  # FIXME deal with fill values

                    for c in data_columns:
                        # Create variable if it doesn't exist
                        var_name = cf_safe_name(c)
                        if var_name not in nc.variables:
                            if np.issubdtype(sdf[c].dtype, 'S') or sdf[c].dtype == object:
                                # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                                v = nc.createVariable(var_name, get_dtype(sdf[c]), ('time', 'z', 'station'))
                            else:
                                v = nc.createVariable(var_name, get_dtype(sdf[c]), ('time', 'z', 'station'), fill_value=sdf[c].dtype.type(cls.default_fill_value))

                            if var_name not in attributes:
                                attributes[var_name] = {}
                            attributes[var_name] = dict_update(attributes[var_name], {
                                'coordinates' : 'time latitude longitude z',
                            })
                        else:
                            v = nc.variables[var_name]

                        if hasattr(v, '_FillValue'):
                            vvalues = sdf[c].fillna(v._FillValue).values
                        else:
                            # Use an empty string... better than nothing!
                            vvalues = sdf[c].fillna('').values

                        v[itime, :, istation] = vvalues

            nc.update_attributes(attributes)

        return OrthogonalMultidimensionalTimeseriesProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self):
        svar = self.filter_by_attrs(cf_role='timeseries_id')[0]
        try:
            s = normalize_array(svar)
        except ValueError:
            s = np.asarray(list(range(len(svar))), dtype=np.integer)

        # T
        tvar = self.t_axes()[0]
        t = nc4.num2date(tvar[:], tvar.units, getattr(tvar, 'calendar', 'standard'))
        if isinstance(t, datetime):
            # Size one
            t = np.array([t.isoformat()], dtype='datetime64')

        # X
        xvar = self.x_axes()[0]
        x = generic_masked(xvar[:], attrs=self.vatts(xvar.name)).round(5)

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:], attrs=self.vatts(yvar.name)).round(5)

        # Z
        zvar = self.z_axes()[0]
        z = generic_masked(zvar[:], attrs=self.vatts(zvar.name))

        # dimensions
        n_times = len(t)
        n_stations = len(s)
        n_z = len(z)

        # denormalize table structure
        t = np.repeat(t, n_stations * n_z)
        z = np.tile(np.repeat(z, n_stations), n_times)
        s = np.tile(s, n_z * n_times)
        y = np.tile(y, n_times * n_z)
        x = np.tile(x, n_times * n_z)

        df_data = OrderedDict([
            ('t', t),
            ('x', x),
            ('y', y),
            ('z', z),
            ('station', s),
        ])

        extract_vars = copy(self.variables)
        del extract_vars[svar.name]
        del extract_vars[xvar.name]
        del extract_vars[yvar.name]
        del extract_vars[zvar.name]
        del extract_vars[tvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):
            if dvar[:].flatten().size != n_stations * n_times * n_z:
                logger.warning("Variable {} is not the correct size, skipping.".format(dnam))
                continue

            vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dnam))
            if vdata.size == 1:
                vdata = vdata[0]
            df_data[dnam] = vdata

        df = pd.DataFrame(df_data)

        return df

    def nc_attributes(self):
        atts = super(OrthogonalMultidimensionalTimeseriesProfile, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'timeseriesProfile',
                'cdm_data_type': 'TimeseriesProfile'
            },
            'station' : {
                'cf_role': 'timeseries_id',
                'long_name' : 'station identifier'
            },
            'longitude': {
                'axis': 'X'
            },
            'latitude': {
                'axis': 'Y'
            },
            'z': {
                'axis': 'Z'
            },
            'time': {
                'units': self.default_time_unit,
                'standard_name': 'time',
                'axis': 'T'
            }
        })
