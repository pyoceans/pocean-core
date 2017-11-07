#!python
# coding=utf-8
import re
from copy import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    dict_update,
    generic_masked,
    get_default_axes,
    get_dtype,
    get_mapped_axes_variables,
    get_masked_datetime_array,
    normalize_countable_array,
)
from pocean.cf import CFDataset
from pocean.cf import cf_safe_name

from pocean import logger as L  # noqa


class OrthogonalMultidimensionalTimeseries(CFDataset):
    """
    H.2.1. Orthogonal multidimensional array representation of time series

    If the time series instances have the same number of elements and the time values are identical
    for all instances, you may use the orthogonal multidimensional array representation. This has
    either a one-dimensional coordinate variable, time(time), provided the time values are ordered
    monotonically, or a one-dimensional auxiliary coordinate variable, time(o), where o is the
    element dimension. In the former case, listing the time variable in the coordinates attributes
    of the data variables is optional.
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
        axes = get_default_axes(kwargs.pop('axes', {}))
        data_columns = [ d for d in df.columns if d not in axes ]

        with OrthogonalMultidimensionalTimeseries(output, 'w') as nc:

            station_group = df.groupby(axes.station)
            num_stations = len(station_group)

            # assume all groups are the same size and have identical times
            _, sdf = list(station_group)[0]
            t = sdf[axes.t]

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            nc.createDimension(axes.t, t.size)
            nc.createDimension(axes.station, num_stations)
            station = nc.createVariable(axes.station, get_dtype(df.station), (axes.station,))

            time = nc.createVariable(axes.t, 'f8', (axes.t,))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), (axes.station,))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), (axes.station,))
            z = nc.createVariable(axes.z, get_dtype(df[axes.z]), (axes.station,), fill_value=df[axes.z].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes), kwargs.pop('attributes', {}))

            # tolist() converts to a python datetime object without timezone and has NaTs.
            g = t.tolist()
            # date2num convers NaTs to np.nan
            gg = nc4.date2num(g, units=cls.default_time_unit)
            # masked_invalid moves np.nan to a masked value
            time[:] = np.ma.masked_invalid(gg)

            for i, (uid, sdf) in enumerate(station_group):
                station[i] = uid
                latitude[i] = sdf[axes.y].iloc[0]
                longitude[i] = sdf[axes.x].iloc[0]

                # TODO: write a test for a Z with a _FillValue
                z[i] = sdf[axes.z].iloc[0]

                for c in data_columns:

                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        if var_name not in attributes:
                            attributes[var_name] = {}
                        if sdf[c].dtype == np.dtype('datetime64[ns]'):
                            fv = np.dtype('f8').type(cls.default_fill_value)
                            v = nc.createVariable(var_name, 'f8', (axes.station, axes.t,), fill_value=fv)
                            tvalues = pd.Series(nc4.date2num(sdf[c].tolist(), units=cls.default_time_unit))
                            attributes[var_name] = dict_update(attributes[var_name], {
                                'units': cls.default_time_unit
                            })
                        elif np.issubdtype(sdf[c].dtype, 'S') or sdf[c].dtype == object:
                            # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                            v = nc.createVariable(var_name, get_dtype(sdf[c]), (axes.station, axes.t,))
                        else:
                            v = nc.createVariable(var_name, get_dtype(sdf[c]), (axes.station, axes.t,), fill_value=sdf[c].dtype.type(cls.default_fill_value))

                        attributes[var_name] = dict_update(attributes[var_name], {
                            'coordinates' : '{} {} {} {}'.format(
                                axes.t, axes.z, axes.x, axes.y
                            )
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
                        L.debug('{} was not written. Likely a metadata variable'.format(v.name))

            # Set global attributes
            nc.update_attributes(attributes)

        return OrthogonalMultidimensionalTimeseries(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        # axes = get_default_axes(kwargs.pop('axes', {}))
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        raise NotImplementedError

    def to_dataframe(self, clean_cols=False, clean_rows=False, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes)

        # T
        t = get_masked_datetime_array(axv.t[:], axv.t)
        L.debug(['time data size: ', t.size])

        # X
        x = generic_masked(axv.x[:].repeat(t.size), attrs=self.vatts(axv.x.name))
        L.debug(['x data size: ', x.size])

        # Y
        y = generic_masked(axv.y[:].repeat(t.size), attrs=self.vatts(axv.y.name))
        L.debug(['y data size: ', y.size])

        # Z
        z = generic_masked(axv.z[:].repeat(t.size), attrs=self.vatts(axv.z.name))
        L.debug(['z data size: ', z.size])

        svar = axv.station
        s = normalize_countable_array(svar)
        s = np.repeat(s, t.size)
        L.debug(['station data size: ', s.size])

        # now repeat t per station
        # figure out if this is a single-station file
        # do this by checking the dimensions of the Z var
        if axv.z.ndim == 1:
            t = np.repeat(t, len(svar))

        df_data = OrderedDict([
            (axes.t, t),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.station, s),
        ])

        building_index_to_drop = np.ma.zeros(t.size, dtype=bool)

        # Axes variables are already processed so skip them
        extract_vars = copy(self.variables)
        for ncvar in axv._asdict().values():
            if ncvar is not None and ncvar.name in extract_vars:
                del extract_vars[ncvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):
            vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dnam))

            # Carry through size 1 variables
            if vdata.size == 1:
                vdata = vdata[0]
            else:
                if dvar[:].flatten().size != t.size:
                    L.warning("Variable {} is not the correct size, skipping.".format(dnam))
                    continue

                building_index_to_drop = (building_index_to_drop == True) & (vdata.mask == True)  # noqa

            # Convert to datetime objects for identified time variables
            try:
                if re.match(r'.* since .*', dvar.units):
                    vdata = nc4.num2date(vdata.data[:], dvar.units, getattr(dvar, 'calendar', 'standard'))
            except AttributeError:
                pass

            df_data[dnam] = vdata

        df = pd.DataFrame(df_data)

        # Drop all data columns with no data
        if clean_cols:
            df = df.dropna(axis=1, how='all')

        # Drop all data rows with no data variable data
        if clean_rows:
            df = df.iloc[~building_index_to_drop]

        return df

    def nc_attributes(self, axes):
        atts = super(OrthogonalMultidimensionalTimeseries, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'timeseries',
                'cdm_data_type': 'Timeseries'
            },
            axes.station : {
                'cf_role': 'timeseries_id',
                'long_name' : 'station identifier'
            },
            axes.t: {
                'units': self.default_time_unit,
                'standard_name': 'time',
                'axis': 'T'
            },
            axes.y: {
                'axis': 'Y'
            },
            axes.x: {
                'axis': 'X'
            },
            axes.z: {
                'axis': 'Z'
            }
        })
