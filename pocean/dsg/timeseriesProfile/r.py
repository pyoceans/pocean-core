#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd

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
    normalize_array,
    normalize_countable_array,
)
from pocean.utils import logger as L  # noqa


class RaggedTimeseriesProfile(CFDataset):

    @classmethod
    def is_mine(cls, dsg, strict=False):
        try:
            assert dsg.featureType.lower() == 'timeseriesprofile'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            o_index_vars = dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )
            assert len(o_index_vars) == 1
            assert o_index_vars[0].sample_dimension in dsg.dimensions  # Sample dimension

            _ = dsg.filter_by_attrs(
                cf_role='profile_id'
            )[0]

            svar = dsg.filter_by_attrs(
                cf_role='timeseries_id'
            )[0]
            sdata = normalize_array(svar)
            if not isinstance(sdata, str) and len(sdata.shape) > 0:
                r_index_vars = dsg.filter_by_attrs(
                    instance_dimension=lambda x: x is not None
                )
                assert len(r_index_vars) == 1
                assert r_index_vars[0].instance_dimension in dsg.dimensions  # Station dimension

        except BaseException:
            if strict is True:
                raise
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        daxes = axes

        reduce_dims = kwargs.pop('reduce_dims', False)
        unlimited = kwargs.pop('unlimited', False)

        unique_dims = kwargs.pop('unique_dims', False)
        if unique_dims is True:
            # Rename the dimension to avoid a dimension and coordinate having the same name
            # which is not supported in xarray
            changed_axes = { k: f'{v}_dim' for k, v in axes._asdict().items() }
            daxes = get_default_axes(changed_axes)

        # Downcast anything from int64 to int32
        # Convert any timezone aware datetimes to native UTC times
        df = downcast_dataframe(nativize_times(df))

        with RaggedTimeseriesProfile(output, 'w') as nc:

            station_groups = df.groupby(axes.station)
            unique_stations = list(station_groups.groups.keys())
            num_stations = len(unique_stations)

            # Calculate the max number of profiles
            profile_groups = df.groupby(axes.profile)
            unique_profiles = list(profile_groups.groups.keys())
            num_profiles = len(unique_profiles)
            nc.createDimension(daxes.profile, num_profiles)

            if reduce_dims is True and num_stations == 1:
                # If a singular station, remove the dimension
                station_dimensions = ()
                s_ind = None
            else:
                station_dimensions = (daxes.station,)
                nc.createDimension(daxes.station, num_stations)
                # The station this profile belongs to
                s_ind = nc.createVariable('stationIndex', 'i4', (daxes.profile,))

            station = nc.createVariable(axes.station, get_dtype(unique_stations), station_dimensions)
            profile = nc.createVariable(axes.profile, get_dtype(df[axes.profile]), (daxes.profile,))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), station_dimensions)
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), station_dimensions)

            # Get unique obs by grouping on traj and profile and getting the max size
            if unlimited is True:
                nc.createDimension(daxes.sample, None)
            else:
                nc.createDimension(daxes.sample, len(df))

            # Number of observations in each profile
            row_size = nc.createVariable('rowSize', 'i4', (daxes.profile,))

            # Axes variables are already processed so skip them
            data_columns = [ d for d in df.columns if d not in axes ]
            data_columns += [axes.t, axes.z]  # time isn't really special, its dimensioned by obs
            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            for i, (sname, srg) in enumerate(station_groups):
                station[i] = sname
                latitude[i] = df[axes.y][df[axes.station] == sname].dropna().iloc[0]
                longitude[i] = df[axes.x][df[axes.station] == sname].dropna().iloc[0]

            for j, (pname, pfg) in enumerate(profile_groups):
                profile[j] = pname
                row_size[j] = len(pfg)
                if s_ind is not None:
                    s_ind[j] = np.argwhere(station[:] == pfg[axes.station].dropna().iloc[0]).item()

            # Add back in the z axes that was removed when calculating data_columns
            # and ignore variables that were stored in the profile index
            skips = ['stationIndex', 'rowSize']
            for c in [ d for d in data_columns if d not in skips ]:
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    v = create_ncvar_from_series(
                        nc,
                        var_name,
                        (daxes.sample,),
                        df[c],
                    )
                else:
                    v = nc.variables[var_name]
                vvalues = get_ncdata_from_series(df[c], v)
                try:
                    if unlimited is True:
                        v[:] = vvalues
                    else:
                        v[:] = vvalues.reshape(v.shape)
                except BaseException:
                    L.exception(f'Failed to add {c}')
                    continue

            # Metadata variables
            nc.createVariable('crs', 'i4')

            # Set attributes
            nc.update_attributes(attributes)

        return RaggedTimeseriesProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        axv = get_mapped_axes_variables(self, axes)

        # Profile dimension
        p_var = self.filter_by_attrs(cf_role='profile_id')[0]
        p_dim = self.dimensions[p_var.dimensions[0]]

        # Station dimension
        s_var = self.filter_by_attrs(cf_role='timeseries_id')[0]
        if s_var.ndim == 1:
            s_dim = self.dimensions[s_var.dimensions[0]]
        elif s_var.ndim == 0:
            s_dim = None
        else:
            raise ValueError('Number of dimension on the station (timeseries_id) must be 0 or 1')

        # Station index
        r_index_var = self.filter_by_attrs(instance_dimension=lambda x: x is not None)
        if not r_index_var:
            # A reduced netCDF file, set station to 0 so it pulls the first value
            # of the variable that identifies the stations
            r_index_var = [0]
        else:
            r_index_var = r_index_var[0]

        # Sample (obs) dimension
        o_index_var = self.filter_by_attrs(sample_dimension=lambda x: x is not None)
        if not o_index_var:
            raise ValueError(
                'Could not find the "sample_dimension" attribute on any variables, '
                'is this a valid {}?'.format(self.__class__.__name__)
            )
        else:
            o_index_var = o_index_var[0]

        # Sample dimension
        # Since this is a flat dataframe, everything is the length of the obs dimension
        row_sizes = o_index_var[:]
        o_dim = self.dimensions[o_index_var.sample_dimension]

        profile_indexes = normalize_countable_array(p_var, count_if_none=p_dim.size)
        p = np.repeat(profile_indexes, row_sizes)

        stat_indexes = normalize_countable_array(s_var, count_if_none=s_dim.size)
        r = np.ma.masked_all(o_dim.size, dtype=stat_indexes.dtype)

        # Lat and Lon are on the station dimension
        xvar = axv.x
        x = np.ma.masked_all(o_dim.size, dtype=xvar.dtype)
        yvar = axv.y
        y = np.ma.masked_all(o_dim.size, dtype=yvar.dtype)
        si = 0
        for i in np.arange(stat_indexes.size):
            ei = si + o_index_var[i]
            r[si:ei] = np.array(stat_indexes[r_index_var[i]])
            x[si:ei] = xvar[i]
            y[si:ei] = yvar[i]
            si = ei
        x = generic_masked(x, minv=-180, maxv=180)
        y = generic_masked(y, minv=-90, maxv=90)

        # Time and Z are on the sample (obs) dimension
        tvar = axv.t
        t = get_masked_datetime_array(
            generic_masked(tvar[:].flatten(), attrs=self.vatts(tvar.name)),
            tvar
        )
        z = generic_masked(axv.z[:].flatten(), attrs=self.vatts(axv.z.name))

        df_data = OrderedDict([
            (axes.t, t),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.station, r),
            (axes.profile, p)
        ])

        building_index_to_drop = np.ones(o_dim.size, dtype=bool)

        extract_vars = copy(self.variables)
        # Skip the station and row index variables
        del extract_vars[o_index_var.name]
        del extract_vars[r_index_var.name]
        # Axes variables are already processed so skip them
        for ncvar in axv._asdict().values():
            if ncvar is not None and ncvar.name in extract_vars:
                del extract_vars[ncvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):

            # Profile dimensions
            if dvar.dimensions == (p_dim.name,):
                vdata = generic_masked(
                    np.repeat(
                        dvar[:].flatten().astype(dvar.dtype),
                        row_sizes
                    ),
                    attrs=self.vatts(dnam)
                )

            # Sample dimensions
            elif dvar.dimensions == (o_dim.name,):
                vdata = generic_masked(dvar[:].flatten().astype(dvar.dtype), attrs=self.vatts(dnam))

            else:
                vdata = generic_masked(dvar[:].flatten().astype(dvar.dtype), attrs=self.vatts(dnam))
                # Carry through size 1 variables
                if vdata.size == 1:
                    if vdata[0] is np.ma.masked:
                        L.warning(f"Skipping variable {dnam} that is completely masked")
                        continue
                else:
                    L.warning(f"Skipping variable {dnam} since it didn't match any dimension sizes")
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
                'featureType': 'timeSeriesProfile',
                'cdm_data_type': 'TimeseriesProfile',
                'cdm_timeseries_variables': axes.station,
                'cdm_profile_variables': axes.profile,
                'subsetVariables': '{x},{y},{t},{station}'.format(**axes._asdict())
            },
            axes.station : {
                'cf_role': 'timeseries_id',
                'long_name' : 'station identifier',
                'ioos_category': 'identifier'
            },
            axes.profile : {
                'cf_role': 'profile_id',
                'long_name' : 'profile identifier',
                'ioos_category': 'identifier'
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
            },
            'stationIndex': {
                'long_name': 'which station this profile belongs to',
                'instance_dimension': daxes.station
            },
            'rowSize': {
                'long_name': 'number of obs in this profile',
                'sample_dimension': daxes.sample
            }
        })
