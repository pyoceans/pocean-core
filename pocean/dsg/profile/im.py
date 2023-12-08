#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd
from cftime import date2num

from pocean import logger as L  # noqa
from pocean.cf import cf_safe_name, CFDataset
from pocean.dsg.profile import profile_calculated_metadata
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


class IncompleteMultidimensionalProfile(CFDataset):
    """
    If there are the same number of levels in each profile, but they do not
    have the same set of vertical coordinates, one can use the incomplete
    multidimensional array representation, which the vertical coordinate
    variable is two-dimensional e.g. replacing z(z) in Example H.8,
    "Atmospheric sounding profiles for a common set of vertical coordinates
    stored in the orthogonal multidimensional array representation." with
    alt(profile,z).  This representation also allows one to have a variable
    number of elements in different profiles, at the cost of some wasted space.
    In that case, any unused elements of the data and auxiliary coordinate
    variables must contain missing data values (section 9.6).
    """

    @classmethod
    def is_mine(cls, dsg, strict=False):
        try:
            pvars = dsg.filter_by_attrs(cf_role='profile_id')
            assert len(pvars) == 1
            assert dsg.featureType.lower() == 'profile'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            # Allow for string variables
            pvar = pvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(pvar.dimensions) <= 2

            t = dsg.t_axes()[0]
            x = dsg.x_axes()[0]
            y = dsg.y_axes()[0]
            z = dsg.z_axes()[0]
            assert len(z.dimensions) == 2

            assert t.size == pvar.size
            assert x.size == pvar.size
            assert y.size == pvar.size
            p_dim = dsg.dimensions[pvar.dimensions[0]]
            z_dim = dsg.dimensions[[ d for d in z.dimensions if d != p_dim.name ][0]]
            for dv in dsg.data_vars():
                assert len(dv.dimensions) in [1, 2]  # dimensioned by profile or profile, z
                assert z_dim.name in dv.dimensions or p_dim.name in dv.dimensions
                assert dv.size in [z_dim.size, p_dim.size, z_dim.size * p_dim.size]

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

        with IncompleteMultidimensionalProfile(output, 'w') as nc:

            profile_group = df.groupby(axes.profile)

            if unlimited is True:
                max_profiles = None
            else:
                max_profiles = df[axes.profile].unique().size
            nc.createDimension(daxes.profile, max_profiles)

            max_zs = profile_group.size().max()
            nc.createDimension(daxes.z, max_zs)

            # Metadata variables
            nc.createVariable('crs', 'i4')

            profile = nc.createVariable(axes.profile, get_dtype(df[axes.profile]), (daxes.profile,))

            # Create all of the variables
            time = nc.createVariable(axes.t, 'f8', (daxes.profile,))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), (daxes.profile,))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), (daxes.profile,))
            z = nc.createVariable(axes.z, get_dtype(df[axes.z]), (daxes.profile, daxes.z), fill_value=df[axes.z].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            # Create vars based on full dataframe (to get all variables)
            for c in data_columns:
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    v = create_ncvar_from_series(
                        nc,
                        var_name,
                        (daxes.profile, daxes.z),
                        df[c],
                    )
                    attributes[var_name] = dict_update(attributes.get(var_name, {}), {
                        'coordinates': '{} {} {} {}'.format(
                            axes.t, axes.z, axes.x, axes.y
                        )
                    })

            # Write values for each profile within profile_group
            for i, (uid, pdf) in enumerate(profile_group):
                profile[i] = uid

                time[i] = date2num(pdf[axes.t].iloc[0], units=cls.default_time_unit)
                latitude[i] = pdf[axes.y].iloc[0]
                longitude[i] = pdf[axes.x].iloc[0]

                zvalues = pdf[axes.z].fillna(z._FillValue).values
                sl = slice(0, zvalues.size)
                z[i, sl] = zvalues

                for c in data_columns:
                    var_name = cf_safe_name(c)
                    v = nc.variables[var_name]

                    vvalues = get_ncdata_from_series(pdf[c], v)

                    sl = slice(0, vvalues.size)
                    v[i, sl] = vvalues

            # Set global attributes
            nc.update_attributes(attributes)

        return IncompleteMultidimensionalProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        return profile_calculated_metadata(df, axes, geometries)

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes)

        # Multiple profiles in the file
        pvar = axv.profile
        p_dim = self.dimensions[pvar.dimensions[0]]

        zvar = axv.z
        zs = len(self.dimensions[[ d for d in zvar.dimensions if d != p_dim.name ][0]])

        # Profiles
        p = normalize_countable_array(pvar)
        p = p.repeat(zs)

        # Z
        z = generic_masked(zvar[:].flatten(), attrs=self.vatts(zvar.name))

        # T
        tvar = axv.t
        t = tvar[:].repeat(zs)
        nt = get_masked_datetime_array(t, tvar).flatten()

        # X
        xvar = axv.x
        x = generic_masked(xvar[:].repeat(zs), attrs=self.vatts(xvar.name))

        # Y
        yvar = axv.y
        y = generic_masked(yvar[:].repeat(zs), attrs=self.vatts(yvar.name))

        df_data = OrderedDict([
            (axes.t, nt),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.profile, p)
        ])

        building_index_to_drop = np.ones(t.size, dtype=bool)

        extract_vars = copy(self.variables)
        for ncvar in axv._asdict().values():
            if ncvar is not None and ncvar.name in extract_vars:
                del extract_vars[ncvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):

            # Profile dimension
            if dvar.dimensions == pvar.dimensions:
                vdata = generic_masked(dvar[:].repeat(zs).astype(dvar.dtype), attrs=self.vatts(dnam))

            # Profile, z dimension
            elif dvar.dimensions == zvar.dimensions:
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
                'featureType': 'profile',
                'cdm_data_type': 'Profile'
            },
            axes.profile : {
                'cf_role': 'profile_id',
                'long_name' : 'profile identifier'
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
