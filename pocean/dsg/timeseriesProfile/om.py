#!python
# coding=utf-8
from copy import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    dict_update,
    downcast_dataframe,
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
        axes = get_default_axes(kwargs.pop('axes', {}))
        data_columns = [ d for d in df.columns if d not in axes ]

        reduce_dims = kwargs.pop('reduce_dims', False)
        unlimited = kwargs.pop('unlimited', False)

        # Downcast anything from int64 to int32
        df = downcast_dataframe(df)

        # Make a new index that is the Cartesian product of all of the values from all of the
        # values of the old index. This is so don't have to iterate over anything. The full column
        # of data will be able to be shaped to the size of the final unique sized dimensions.
        index_order = [axes.t, axes.z, axes.station]
        df = df.set_index(index_order)
        df = df.reindex(
            pd.MultiIndex.from_product(df.index.levels, names=index_order)
        )

        unique_z = df.index.get_level_values(axes.z).unique().values
        unique_t = df.index.get_level_values(axes.t).unique().tolist()  # tolist converts to Timestamp
        all_stations = df.index.get_level_values(axes.station)
        unique_s = all_stations.unique()

        with OrthogonalMultidimensionalTimeseriesProfile(output, 'w') as nc:

            if reduce_dims is True and unique_s.size == 1:
                # If a singlular trajectory, we can reduce that dimension if it is of size 1
                def ts():
                    return np.s_[:, :]
                default_dimensions = (axes.t, axes.z)
                station_dimensions = ()
            else:
                def ts():
                    return np.s_[:, :, :]
                default_dimensions = (axes.t, axes.z, axes.station)
                station_dimensions = (axes.station,)
                nc.createDimension(axes.station, unique_s.size)

            station = nc.createVariable(axes.station, get_dtype(unique_s), station_dimensions)
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), station_dimensions)
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), station_dimensions)
            # Assign over loop because VLEN variables (strings) have to be assigned by integer index
            # and we need to find the lat/lon based on station index
            for si, st in enumerate(unique_s):
                station[si] = st
                latitude[si] = df[axes.y][all_stations == st].dropna().iloc[0]
                longitude[si] = df[axes.x][all_stations == st].dropna().iloc[0]

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            if unlimited is True:
                nc.createDimension(axes.t, None)
            else:
                nc.createDimension(axes.t, len(unique_t))
            time = nc.createVariable(axes.t, 'f8', (axes.t,))
            time[:] = nc4.date2num(unique_t, units=cls.default_time_unit)

            nc.createDimension(axes.z, unique_z.size)
            z = nc.createVariable(axes.z, get_dtype(unique_z), (axes.z,))
            z[:] = unique_z

            attributes = dict_update(nc.nc_attributes(axes), kwargs.pop('attributes', {}))

            for c in data_columns:
                # Create variable if it doesn't exist
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    if np.issubdtype(df[c].dtype, 'S') or df[c].dtype == object:
                        # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                        v = nc.createVariable(var_name, get_dtype(df[c]), default_dimensions)
                    else:
                        v = nc.createVariable(var_name, get_dtype(df[c]), default_dimensions, fill_value=df[c].dtype.type(cls.default_fill_value))

                    if var_name not in attributes:
                        attributes[var_name] = {}
                    attributes[var_name] = dict_update(attributes[var_name], {
                        'coordinates' : 'time latitude longitude z',
                    })
                else:
                    v = nc.variables[var_name]

                if hasattr(v, '_FillValue'):
                    vvalues = df[c].fillna(v._FillValue).values
                else:
                    # Use an empty string... better than nothing!
                    vvalues = df[c].fillna('').values

                v[ts()] = vvalues.reshape(len(unique_t), unique_z.size, unique_s.size)

            nc.update_attributes(attributes)

        return OrthogonalMultidimensionalTimeseriesProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        # axes = get_default_axes(kwargs.pop('axes', {}))
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        raise NotImplementedError

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes)

        svar = axv.station
        s = normalize_countable_array(svar)

        # T
        t = get_masked_datetime_array(axv.t[:], axv.t)
        n_times = t.size

        # X
        x = generic_masked(axv.x[:], attrs=self.vatts(axv.x.name))

        # Y
        y = generic_masked(axv.y[:], attrs=self.vatts(axv.y.name))

        # Z
        z = generic_masked(axv.z[:], attrs=self.vatts(axv.z.name))
        n_z = z.size

        # denormalize table structure
        t = np.repeat(t, s.size * n_z)
        z = np.tile(np.repeat(z, s.size), n_times)
        s = np.tile(s, n_z * n_times)
        y = np.tile(y, n_times * n_z)
        x = np.tile(x, n_times * n_z)

        df_data = OrderedDict([
            (axes.t, t),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.station, s),
        ])

        building_index_to_drop = np.ones(t.size, dtype=bool)

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
        atts = super(OrthogonalMultidimensionalTimeseriesProfile, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'timeSeriesProfile',
                'cdm_data_type': 'TimeseriesProfile'
            },
            axes.station : {
                'cf_role': 'timeseries_id',
                'long_name' : 'station identifier'
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
