#!python
# coding=utf-8
import re
from copy import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    normalize_array,
    get_dtype,
    dict_update,
    generic_masked,
    get_masked_datetime_array
)
from pocean.cf import CFDataset
from pocean.cf import cf_safe_name

from pocean import logger  # noqa


class OrthogonalMultidimensionalTimeseries(CFDataset):
    """
    H.2.1. Orthogonal multidimensional array representation of time series

    If the time series instances have the same number of elements and the time values are identical for all instances, you may use the orthogonal multidimensional array representation. This has either a one-dimensional coordinate variable, time(time), provided the time values are ordered monotonically, or a one-dimensional auxiliary coordinate variable, time(o), where o is the element dimension. In the former case, listing the time variable in the coordinates attributes of the data variables is optional.
    """

    @classmethod
    def is_mine(cls, dsg):
        try:
            rvars = dsg.filter_by_attrs(cf_role='timeseries_id')
            assert len(rvars) == 1
            assert dsg.featureType.lower() == 'timeseries'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1

            # Not a CR
            assert not dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )

            # Not an IR
            assert not dsg.filter_by_attrs(
                instance_dimension=lambda x: x is not None
            )

            # OM files will always have a time variable with one dimension.
            assert len(dsg.t_axes()[0].dimensions) == 1

            # Allow for string variables
            rvar = rvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(rvar.dimensions) <= 2

        except AssertionError:
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        reserved_columns = ['station', 't', 'x', 'y', 'z']
        data_columns = [ d for d in df.columns if d not in reserved_columns ]

        with OrthogonalMultidimensionalTimeseries(output, 'w') as nc:

            station_group = df.groupby('station')
            num_stations = len(station_group)

            # assume all groups are the same size and have identical times
            _, sdf = list(station_group)[0]
            t = sdf.t

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            nc.createDimension('time', t.size)
            nc.createDimension('station', num_stations)
            station = nc.createVariable('station', get_dtype(df.station), ('station',))

            time = nc.createVariable('time', 'f8', ('time',))
            latitude = nc.createVariable('latitude', get_dtype(df.y), ('station',))
            longitude = nc.createVariable('longitude', get_dtype(df.x), ('station',))
            z = nc.createVariable('z', get_dtype(df.z), ('station',), fill_value=df.z.dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(), kwargs.pop('attributes', {}))

            time[:] = nc4.date2num(t.tolist(), units=cls.default_time_unit)

            for i, (uid, sdf) in enumerate(station_group):
                station[i] = uid
                latitude[i] = sdf.y.iloc[0]
                longitude[i] = sdf.x.iloc[0]

                # TODO: write a test for a Z with a _FillValue
                z[i] = sdf.z.iloc[0]

                for c in data_columns:

                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        if var_name not in attributes:
                            attributes[var_name] = {}
                        if sdf[c].dtype == np.dtype('datetime64[ns]'):
                            fv = np.dtype('f8').type(cls.default_fill_value)
                            v = nc.createVariable(var_name, 'f8', ('station', 'time',), fill_value=fv)
                            tvalues = pd.Series(nc4.date2num(sdf[c].tolist(), units=cls.default_time_unit))
                            attributes[var_name] = dict_update(attributes[var_name], {
                                'units': cls.default_time_unit
                            })
                        elif np.issubdtype(sdf[c].dtype, 'S') or sdf[c].dtype == object:
                            # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                            v = nc.createVariable(var_name, get_dtype(sdf[c]), ('station', 'time',))
                        else:
                            v = nc.createVariable(var_name, get_dtype(sdf[c]), ('station', 'time',), fill_value=sdf[c].dtype.type(cls.default_fill_value))

                        attributes[var_name] = dict_update(attributes[var_name], {
                            'coordinates' : 'time latitude longitude z',
                        })
                    else:
                        v = nc.variables[var_name]

                    if sdf[c].dtype == np.dtype('datetime64[ns]'):
                        vvalues = tvalues.fillna(v._FillValue).values
                    elif hasattr(v, '_FillValue'):
                        vvalues = sdf[c].fillna(v._FillValue).values
                    else:
                        # Use an empty string... better than nothing!
                        vvalues = sdf[c].fillna('').values

                    try:
                        v[i, :] = vvalues
                    except BaseException:
                        logger.debug('{} was not written. Likely a metadata variable'.format(v.name))

            # Set global attributes
            nc.update_attributes(attributes)

        return OrthogonalMultidimensionalTimeseries(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self, clean_cols=False, clean_rows=False):

        # Don't pass around the attributes store them in the class

        # T
        tvar = self.t_axes()[0]
        t = get_masked_datetime_array(tvar[:], tvar)
        logger.debug(['time data size: ', t.size])

        svar = self.filter_by_attrs(cf_role='timeseries_id')[0]
        # Stations
        # TODO: Make sure there is a test for a file with multiple time variables
        try:
            s = normalize_array(svar)
        except ValueError:
            s = np.asarray(list(range(len(svar))), dtype=np.integer)
        s = np.repeat(s, t.size)
        logger.debug(['station data size: ', s.size])

        # X
        xvar = self.x_axes()[0]
        x = generic_masked(xvar[:].repeat(t.size), attrs=self.vatts(xvar.name))
        logger.debug(['x data size: ', x.size])

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:].repeat(t.size), attrs=self.vatts(yvar.name))
        logger.debug(['y data size: ', y.size])

        # Z
        zvar = self.z_axes()[0]
        z = generic_masked(zvar[:].repeat(t.size), attrs=self.vatts(zvar.name))
        logger.debug(['z data size: ', z.size])

        # now repeat t per station

        # figure out if this is a single-station file
        # do this by checking the dimensions of the Z var
        if zvar.ndim == 1:
            t = np.repeat(t, len(svar))

        df_data = OrderedDict([
            ('t', t),
            ('x', x),
            ('y', y),
            ('z', z),
            ('station', s),
        ])

        building_index_to_drop = np.ma.zeros(t.size, dtype=bool)
        extract_vars = copy(self.variables)
        del extract_vars[svar.name]
        del extract_vars[xvar.name]
        del extract_vars[yvar.name]
        del extract_vars[zvar.name]
        del extract_vars[tvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):
            if dvar[:].flatten().size > t.size:
                logger.warning("Variable {} is not the correct size, skipping.".format(dnam))
                continue

            vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dnam))
            building_index_to_drop = (building_index_to_drop == True) & (vdata.mask == True)  # noqa
            try:
                if re.match(r'.* since .*', dvar.units):
                    vdata = nc4.num2date(vdata.data[:], dvar.units, getattr(dvar, 'calendar', 'standard'))
            except AttributeError:
                pass

            if vdata.size == 1:
                vdata = vdata[0]
            df_data[dnam] = vdata

        df = pd.DataFrame()
        for k, v in df_data.items():
            try:
                df[k] = v
            except ValueError as e:
                logger.exception("Could not write {} as {}. {}".format(k, v, e))
                raise

        # Drop all data columns with no data
        if clean_cols:
            df = df.dropna(axis=1, how='all')

        # Drop all data rows with no data variable data
        if clean_rows:
            df = df.iloc[~building_index_to_drop]

        return df

    def nc_attributes(self):
        atts = super(OrthogonalMultidimensionalTimeseries, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'timeseries',
                'cdm_data_type': 'Timeseries'
            },
            'station' : {
                'cf_role': 'timeseries_id',
                'long_name' : 'station identifier'
            },
            'time': {
                'units': self.default_time_unit,
                'standard_name': 'time',
                'axis': 'T'
            },
            'latitude': {
                'axis': 'Y'
            },
            'longitude': {
                'axis': 'X'
            },
            'z': {
                'axis': 'Z'
            }
        })
