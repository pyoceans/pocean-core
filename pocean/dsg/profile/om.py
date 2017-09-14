# -*- coding: utf-8 -*-
from datetime import datetime
from collections import namedtuple

import six
import numpy as np
import pandas as pd
import netCDF4 as nc4

from shapely.geometry import Point, LineString

from pocean.utils import unique_justseen, normalize_array, generic_masked
from pocean.cf import CFDataset
from pocean import logger


class OrthogonalMultidimensionalProfile(CFDataset):
    """
    If the profile instances have the same number of elements and the vertical
    coordinate values are identical for all instances, you may use the
    orthogonal multidimensional array representation. This has either a
    one-dimensional coordinate variable, z(z), provided the vertical coordinate
    values are ordered monotonically, or a one-dimensional auxiliary coordinate
    variable, alt(o), where o is the element dimension. In the former case,
    listing the vertical coordinate variable in the coordinates attributes of
    the data variables is optional.
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

            ps = normalize_array(pvar)
            is_single = isinstance(ps, six.string_types) or ps.size == 1

            t = dsg.t_axes()[0]
            x = dsg.x_axes()[0]
            y = dsg.y_axes()[0]
            z = dsg.z_axes()[0]
            assert len(z.dimensions) == 1
            z_dim = dsg.dimensions[z.dimensions[0]]

            if is_single:
                assert t.size == 1
                assert x.size == 1
                assert y.size == 1
                for dv in dsg.data_vars():
                    assert len(dv.dimensions) == 1
                    assert z_dim.name in dv.dimensions
                    assert dv.size == z_dim.size
            else:
                assert t.size == pvar.size
                assert x.size == pvar.size
                assert y.size == pvar.size
                p_dim = dsg.dimensions[pvar.dimensions[0]]
                for dv in dsg.data_vars():
                    assert len(dv.dimensions) == 2
                    assert z_dim.name in dv.dimensions
                    assert p_dim.name in dv.dimensions
                    assert dv.size == z_dim.size * p_dim.size

        except BaseException:
            return False

        return True

    def from_dataframe(cls, df, output, **kwargs):
        raise NotImplementedError

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
            null_coordinates = df.x.isnull() | df.y.isnull()
            coords = list(unique_justseen(zip(
                df.loc[~null_coordinates, 'x'].tolist(),
                df.loc[~null_coordinates, 'y'].tolist()
            )))
            if len(coords) > 1:
                geometry = LineString(coords)
            elif len(coords) == 1:
                geometry = first_loc

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

        zvar = self.z_axes()[0]
        zs = len(self.dimensions[zvar.dimensions[0]])

        # Profiles
        pvar = self.filter_by_attrs(cf_role='profile_id')[0]
        try:
            p = normalize_array(pvar)
        except ValueError:
            p = np.asarray(list(range(len(pvar))), dtype=np.integer)
        ps = p.size
        p = p.repeat(zs)
        logger.debug(['profile data size: ', p.size])

        # Z
        z = generic_masked(zvar[:], attrs=self.vatts(zvar.name))
        try:
            z = np.tile(z, ps)
        except ValueError:
            z = z.flatten()
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
        x = generic_masked(xvar[:].repeat(zs), attrs=self.vatts(xvar.name))
        logger.debug(['x data size: ', x.size])

        # Y
        yvar = self.y_axes()[0]
        y = generic_masked(yvar[:].repeat(zs), attrs=self.vatts(yvar.name))
        logger.debug(['y data size: ', y.size])

        df_data = {
            't': t,
            'x': x,
            'y': y,
            'z': z,
            'profile': p
        }

        building_index_to_drop = np.ones(t.size, dtype=bool)
        extract_vars = list(set(self.data_vars() + self.ancillary_vars()))
        for i, dvar in enumerate(extract_vars):

            # Profile dimension
            if dvar.dimensions == pvar.dimensions:
                vdata = generic_masked(dvar[:].repeat(zs).flatten(), attrs=self.vatts(dvar.name))

            # Z dimension
            elif dvar.dimensions == zvar.dimensions:
                vdata = generic_masked(np.tile(dvar[:], ps).flatten(), attrs=self.vatts(dvar.name))

            # Profile, z dimension
            elif dvar.dimensions == pvar.dimensions + zvar.dimensions:
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
