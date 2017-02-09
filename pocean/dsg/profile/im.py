# -*- coding: utf-8 -*-
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import netCDF4 as nc4
from pygc import great_distance
from shapely.geometry import Point, LineString

from pocean.utils import (
    unique_justseen,
    normalize_array,
    get_dtype,
    dict_update,
    generic_masked
)
from pocean.cf import CFDataset, cf_safe_name
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
        reserved_columns = ['trajectory', 'profile', 't', 'x', 'y', 'z', 'distance']
        data_columns = [ d for d in df.columns if d not in reserved_columns ]

        with IncompleteMultidimensionalProfile(output, 'w') as nc:

            profile_group = df.groupby('profile')
            max_zs = profile_group.size().max()

            unique_profiles = df.profile.unique()
            nc.createDimension('profile', unique_profiles.size)
            nc.createDimension('z', max_zs)

            # Metadata variables
            nc.createVariable('crs', 'i4')

            profile = nc.createVariable('profile', get_dtype(df.profile), ('profile',))

            # Create all of the variables
            time = nc.createVariable('time', 'i4', ('profile',))
            latitude = nc.createVariable('latitude', get_dtype(df.y), ('profile',))
            longitude = nc.createVariable('longitude', get_dtype(df.x), ('profile',))
            if 'distance' in df:
                distance = nc.createVariable('distance', get_dtype(df.distance), ('profile',))
            z = nc.createVariable('z', get_dtype(df.z), ('profile', 'z'), fill_value=df.z.dtype.type(cls.default_fill_value))

            attributes = dict_update(nc.nc_attributes(), kwargs.pop('attributes', {}))

            for i, (uid, pdf) in enumerate(profile_group):
                profile[i] = uid

                time[i] = nc4.date2num(pdf.t.iloc[0], units=cls.default_time_unit)
                latitude[i] = pdf.y.iloc[0]
                longitude[i] = pdf.x.iloc[0]
                if 'distance' in pdf:
                    distance[i] = pdf.distance.iloc[0]

                zvalues = pdf.z.fillna(z._FillValue).values
                sl = slice(0, zvalues.size)
                z[i, sl] = zvalues
                for c in data_columns:
                    # Create variable if it doesn't exist
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        if np.issubdtype(pdf[c].dtype, 'S') or pdf[c].dtype == object:
                            # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
                            v = nc.createVariable(var_name, get_dtype(pdf[c]), ('profile', 'z'))
                        else:
                            v = nc.createVariable(var_name, get_dtype(pdf[c]), ('profile', 'z'), fill_value=pdf[c].dtype.type(cls.default_fill_value))

                        if var_name not in attributes:
                            attributes[var_name] = {}
                        attributes[var_name] = dict_update(attributes[var_name], {
                            'coordinates' : 'time latitude longitude z',
                        })
                    else:
                        v = nc.variables[var_name]

                    if hasattr(v, '_FillValue'):
                        vvalues = pdf[c].fillna(v._FillValue).values
                    else:
                        # Use an empty string... better than nothing!
                        vvalues = pdf[c].fillna('').values

                    sl = slice(0, vvalues.size)
                    v[i, sl] = vvalues

            # Set global attributes
            nc.update_attributes(attributes)

        return IncompleteMultidimensionalProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)

        profiles = {}
        for pid, pgroup in df.groupby('profile'):
            pgroup = pgroup.sort_values('t')
            first_row = pgroup.iloc[0]
            profile = namedtuple('Profile', ['min_z', 'max_z', 't', 'x', 'y', 'loc'])
            profiles[pid] = profile(
                min_z=pgroup.z.min(),
                max_z=pgroup.z.max(),
                t=first_row.t,
                x=first_row.x,
                y=first_row.y,
                loc=Point(first_row.x, first_row.y)
            )

        geometry = None
        first_row = df.iloc[0]
        first_loc = Point(first_row.x, first_row.y)
        if geometries:
            coords = list(unique_justseen(zip(df.x, df.y)))
            if len(coords) > 1:
                geometry = LineString(coords)  # noqa
            elif len(coords) == 1:
                geometry = first_loc  # noqa

        meta = namedtuple('Metadata', ['min_z', 'max_z', 'min_t', 'max_t', 'profiles', 'first_loc', 'geometry'])
        return meta(
            min_z=df.z.min(),
            max_z=df.z.max(),
            min_t=df.t.min(),
            max_t=df.t.max(),
            profiles=profiles,
            first_loc=first_loc,
            geometry=geometry
        )

    def to_dataframe(self, clean_cols=True, clean_rows=True):
        pvar = self.filter_by_attrs(cf_role='profile_id')[0]
        # Multiple profiles in the file
        p_dim = self.dimensions[pvar.dimensions[0]]
        ps = p_dim.size
        logger.debug(['# profiles: ', ps])

        zvar = self.z_axes()[0]

        z_dim = self.dimensions[[ d for d in zvar.dimensions if d != p_dim.name ][0]]
        zs = z_dim.size

        # Profiles
        try:
            p = normalize_array(pvar)
        except ValueError:
            p = np.asarray(list(range(len(pvar))), dtype=np.integer)
        p = p.repeat(zs)
        logger.debug(['profile data size: ', p.size])

        # Z
        z = generic_masked(zvar[:].flatten(), attrs=self.vatts(zvar.name)).round(5)
        logger.debug(['z data size: ', z.size])

        # T
        tvar = self.t_axes()[0]
        t = nc4.num2date(tvar[:], tvar.units, getattr(tvar, 'calendar', 'standard'))
        if isinstance(t, datetime):
            # Size one
            t = np.array([t.isoformat()], dtype='datetime64')
        t = t.repeat(zs)
        logger.debug(['time data size: ', t.size])

        # X
        xvar = self.x_axes()[0]
        x = generic_masked(xvar[:].repeat(zs), attrs=self.vatts(xvar.name)).round(5)
        logger.debug(['x data size: ', x.size])

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:].repeat(zs), attrs=self.vatts(yvar.name)).round(5)
        logger.debug(['y data size: ', y.size])

        # Distance
        d = np.ma.zeros(y.size, dtype=np.float64)
        d[1:] = great_distance(start_latitude=y[0:-1], end_latitude=y[1:], start_longitude=x[0:-1], end_longitude=x[1:])['distance']
        d = generic_masked(np.cumsum(d), minv=0).round(2)
        logger.debug(['distance data size: ', d.size])

        df_data = {
            't': t,
            'x': x,
            'y': y,
            'z': z,
            'profile': p,
            'distance': d
        }

        building_index_to_drop = np.ones(t.size, dtype=bool)
        extract_vars = list(set(self.data_vars() + self.ancillary_vars()))
        for i, dvar in enumerate(extract_vars):
            vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dvar.name)).round(3)
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
        atts = super(IncompleteMultidimensionalProfile, self).nc_attributes()
        return dict_update(atts, {
            'global' : {
                'featureType': 'profile',
                'cdm_data_type': 'Profile'
            },
            'profile' : {
                'cf_role': 'profile_id',
                'long_name' : 'profile identifier'
            },
            'distance' : {
                'long_name': 'Great circle distance between trajectory points',
                'standard_name': 'distance_between_trajectory_points',
                'units': 'm'
            }
        })
