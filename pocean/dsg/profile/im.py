#!python
# coding=utf-8
from collections import OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4

from pocean.utils import (
    create_ncvar_from_series,
    dict_update,
    generic_masked,
    get_default_axes,
    get_dtype,
    get_masked_datetime_array,
    get_ncdata_from_series,
    normalize_array,
)
from pocean.cf import CFDataset, cf_safe_name
from pocean.dsg.profile import profile_calculated_metadata

from pocean import logger


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
    def is_mine(cls, dsg):
        try:
            pvars = dsg.filter_by_attrs(cf_role='profile_id')
            assert len(pvars) == 1
            assert dsg.featureType.lower() == 'profile'
            assert len(dsg.t_axes()) == 1
            assert len(dsg.x_axes()) == 1
            assert len(dsg.y_axes()) == 1
            assert len(dsg.z_axes()) == 1

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
            assert t.size == pvar.size
            assert x.size == pvar.size
            assert y.size == pvar.size
            p_dim = dsg.dimensions[pvar.dimensions[0]]
            z_dim = dsg.dimensions[[ d for d in z.dimensions if d != p_dim.name ][0]]
            for dv in dsg.data_vars() + [z]:
                assert len(dv.dimensions) == 2
                assert z_dim.name in dv.dimensions
                assert p_dim.name in dv.dimensions
                assert dv.size == z_dim.size * p_dim.size

        except BaseException:
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        data_columns = [ d for d in df.columns if d not in axes ]

        unlimited = kwargs.pop('unlimited', False)

        with IncompleteMultidimensionalProfile(output, 'w') as nc:

            profile_group = df.groupby(axes.profile)

            if unlimited is True:
                max_profiles = None
            else:
                max_profiles = df[axes.profile].unique().size
            nc.createDimension(axes.profile, max_profiles)

            max_zs = profile_group.size().max()
            nc.createDimension(axes.z, max_zs)

            # Metadata variables
            nc.createVariable('crs', 'i4')

            profile = nc.createVariable(axes.profile, get_dtype(df[axes.profile]), (axes.profile,))

            # Create all of the variables
            time = nc.createVariable(axes.t, 'f8', (axes.profile,))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), (axes.profile,))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), (axes.profile,))
            z = nc.createVariable(axes.z, get_dtype(df[axes.z]), (axes.profile, axes.z), fill_value=df[axes.z].dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(axes), kwargs.pop('attributes', {}))

            for i, (uid, pdf) in enumerate(profile_group):
                profile[i] = uid

                time[i] = nc4.date2num(pdf[axes.t].iloc[0], units=cls.default_time_unit)
                latitude[i] = pdf[axes.y].iloc[0]
                longitude[i] = pdf[axes.x].iloc[0]

                zvalues = pdf[axes.z].fillna(z._FillValue).values
                sl = slice(0, zvalues.size)
                z[i, sl] = zvalues
                for c in data_columns:
                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        v = create_ncvar_from_series(nc, var_name, (axes.profile, axes.z), pdf[c])
                        attributes[var_name] = dict_update(attributes.get(var_name, {}), {
                            'coordinates' : '{} {} {} {}'.format(
                                axes.t, axes.z, axes.x, axes.y
                            )
                        })
                    else:
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
        # Multiple profiles in the file
        pvar = self.filter_by_attrs(cf_role='profile_id')[0]
        p_dim = self.dimensions[pvar.dimensions[0]]

        zvar = self.z_axes()[0]
        zs = len(self.dimensions[[ d for d in zvar.dimensions if d != p_dim.name ][0]])

        # Profiles
        try:
            p = normalize_array(pvar)
        except ValueError:
            p = np.asarray(list(range(len(pvar))), dtype=np.integer)
        p = p.repeat(zs)

        # Z
        z = generic_masked(zvar[:].flatten(), attrs=self.vatts(zvar.name))

        # T
        tvar = self.t_axes()[0]
        t = tvar[:].repeat(zs)
        nt = get_masked_datetime_array(t, tvar).flatten()

        # X
        xvar = self.x_axes()[0]
        x = generic_masked(xvar[:].repeat(zs), attrs=self.vatts(xvar.name))

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:].repeat(zs), attrs=self.vatts(yvar.name))

        df_data = OrderedDict([
            (axes.t, nt),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.profile, p)
        ])

        building_index_to_drop = np.ones(t.size, dtype=bool)
        extract_vars = list(set(self.data_vars() + self.ancillary_vars()))
        for i, dvar in enumerate(extract_vars):

            # Profile dimension
            if dvar.dimensions == pvar.dimensions:
                vdata = generic_masked(dvar[:].repeat(zs).flatten(), attrs=self.vatts(dvar.name))

            # Profile, z dimension
            elif dvar.dimensions == zvar.dimensions:
                vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dvar.name))

            else:
                logger.warning("Skipping variable {}... it didn't seem like a data variable".format(dvar))
                continue

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
        atts = super(IncompleteMultidimensionalProfile, self).nc_attributes()
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
