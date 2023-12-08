#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd

from pocean import logger as L  # noqa
from pocean.cf import cf_safe_name, CFDataset
from pocean.utils import (
    create_ncvar_from_series,
    dict_update,
    downcast_dataframe,
    generic_masked,
    get_default_axes,
    get_dtype,
    get_mapped_axes_variables,
    get_masked_datetime_array,
    get_ncdata_from_series,
    nativize_times,
    normalize_countable_array,
)


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
    def is_mine(cls, dsg, strict=False):
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
        _ = kwargs.pop('unlimited', False)

        unique_dims = kwargs.pop('unique_dims', False)
        if unique_dims is True:
            # Rename the dimension to avoid a dimension and coordinate having the same name
            # which is not support in xarray
            changed_axes = { k: f'{v}_dim' for k, v in axes._asdict().items() }
            daxes = get_default_axes(changed_axes)

        # Downcast anything from int64 to int32
        # Convert any timezone aware datetimes to native UTC times
        df = downcast_dataframe(nativize_times(df))

        with OrthogonalMultidimensionalTimeseries(output, 'w') as nc:

            station_group = df.groupby(axes.station)
            num_stations = len(station_group)
            has_z = axes.z is not None

            if reduce_dims is True and num_stations == 1:
                # If a station, we can reduce that dimension if it is of size 1
                def ts(i):
                    return np.s_[:]
                default_dimensions = (daxes.t,)
                station_dimensions = ()
            else:
                def ts(i):
                    return np.s_[i, :]
                default_dimensions = (daxes.station, daxes.t)
                station_dimensions = (daxes.station,)
                nc.createDimension(daxes.station, num_stations)

            # Set the coordinates attribute correctly
            coordinates = [axes.t, axes.x, axes.y]
            if has_z is True:
                coordinates.insert(1, axes.z)
            coordinates = ' '.join(coordinates)

            # assume all groups are the same size and have identical times
            _, sdf = list(station_group)[0]
            t = sdf[axes.t]

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Create all of the variables
            nc.createDimension(daxes.t, t.size)
            time = nc.createVariable(axes.t, 'f8', (daxes.t,))
            station = nc.createVariable(axes.station, get_dtype(df[axes.station]), station_dimensions)
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), station_dimensions)
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), station_dimensions)
            if has_z is True:
                z = nc.createVariable(axes.z, get_dtype(df[axes.z]), station_dimensions, fill_value=df[axes.z].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            time[:] = get_ncdata_from_series(t, time).astype('f8')

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
                        'coordinates': coordinates
                    })

            for i, (uid, sdf) in enumerate(station_group):
                station[i] = uid
                latitude[i] = sdf[axes.y].iloc[0]
                longitude[i] = sdf[axes.x].iloc[0]

                if has_z is True:
                    # TODO: write a test for a Z with a _FillValue
                    z[i] = sdf[axes.z].iloc[0]

                for c in data_columns:
                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    v = nc.variables[var_name]

                    vvalues = get_ncdata_from_series(sdf[c], v)
                    try:
                        v[ts(i)] = vvalues
                    except BaseException:
                        L.debug(f'{v.name} was not written. Likely a metadata variable')

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

        # X
        x = generic_masked(axv.x[:].repeat(t.size), attrs=self.vatts(axv.x.name))

        # Y
        y = generic_masked(axv.y[:].repeat(t.size), attrs=self.vatts(axv.y.name))

        # Z
        if axv.z is not None:
            z = generic_masked(axv.z[:].repeat(t.size), attrs=self.vatts(axv.z.name))
        else:
            z = None

        svar = axv.station
        s = normalize_countable_array(svar)
        s = np.repeat(s, t.size)

        # now repeat t per station
        # figure out if this is a single-station file by checking
        # the dimension size of the x dimension
        if axv.x.ndim == 1:
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
