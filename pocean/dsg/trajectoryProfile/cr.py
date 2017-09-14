#!python
# coding=utf-8
from collections import namedtuple, OrderedDict

import numpy as np
import pandas as pd
import netCDF4 as nc4
from shapely.geometry import Point, LineString

from pocean.utils import (
    unique_justseen,
    normalize_array,
    get_fill_value,
    generic_masked,
    get_masked_datetime_array
)
from pocean.cf import CFDataset
from pocean import logger  # noqa


class ContiguousRaggedTrajectoryProfile(CFDataset):

    @classmethod
    def is_mine(cls, dsg):
        try:
            rvars = dsg.filter_by_attrs(cf_role='trajectory_id')
            assert len(rvars) == 1
            assert dsg.featureType.lower() == 'trajectoryprofile'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            r_index_vars = dsg.filter_by_attrs(
                instance_dimension=lambda x: x is not None
            )
            assert len(r_index_vars) == 1
            assert r_index_vars[0].instance_dimension in dsg.dimensions  # Trajectory dimension

            o_index_vars = dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )
            assert len(o_index_vars) == 1
            assert o_index_vars[0].sample_dimension in dsg.dimensions  # Sample dimension

            # Allow for string variables
            rvar = rvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(rvar.dimensions) <= 2

        except BaseException:
            return False

        return True

    def from_dataframe(cls, df, output, **kwargs):
        raise NotImplementedError

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)

        trajectories = {}
        for tid, tgroup in df.groupby('trajectory'):
            tgroup = tgroup.sort_values('t')

            profiles = {}
            for pid, pgroup in tgroup.groupby('profile'):
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
            first_row = tgroup.iloc[0]
            first_loc = Point(first_row.x, first_row.y)
            if geometries:
                null_coordinates = tgroup.x.isnull() | tgroup.y.isnull()
                coords = list(unique_justseen(zip(
                    tgroup.loc[~null_coordinates, 'x'].tolist(),
                    tgroup.loc[~null_coordinates, 'y'].tolist()
                )))
                if len(coords) > 1:
                    geometry = LineString(coords)
                elif coords == 1:
                    geometry = first_loc

            trajectory = namedtuple('Trajectory', ['min_z', 'max_z', 'min_t', 'max_t', 'profiles', 'first_loc', 'geometry'])
            trajectories[tid] = trajectory(
                min_z=tgroup.z.min(),
                max_z=tgroup.z.max(),
                min_t=tgroup.t.min(),
                max_t=tgroup.t.max(),
                profiles=profiles,
                first_loc=first_loc,
                geometry=geometry
            )

        meta = namedtuple('Metadata', ['min_z', 'max_z', 'min_t', 'max_t', 'trajectories'])
        return meta(
            min_z=df.z.min(),
            max_z=df.z.max(),
            min_t=df.t.min(),
            max_t=df.t.max(),
            trajectories=trajectories
        )

    def to_dataframe(self, clean_cols=True, clean_rows=True):
        # The index variable (trajectory_index) is identified by having an
        # attribute with name of instance_dimension whose value is the instance
        # dimension name (trajectory in this example). The index variable must
        # have the profile dimension as its sole dimension, and must be type
        # integer. Each value in the index variable is the zero-based trajectory
        # index that the profile belongs to i.e. profile p belongs to trajectory
        # i=trajectory_index(p), as in section H.2.5.
        r_index_var = self.filter_by_attrs(instance_dimension=lambda x: x is not None)[0]
        p_dim = self.dimensions[r_index_var.dimensions[0]]       # Profile dimension
        r_dim = self.dimensions[r_index_var.instance_dimension]  # Trajectory dimension

        # The count variable (row_size) contains the number of elements for
        # each profile, which must be written contiguously. The count variable
        # is identified by having an attribute with name sample_dimension whose
        # value is the sample dimension (obs in this example) being counted. It
        # must have the profile dimension as its sole dimension, and must be
        # type integer
        o_index_var = self.filter_by_attrs(sample_dimension=lambda x: x is not None)[0]
        o_dim = self.dimensions[o_index_var.sample_dimension]  # Sample dimension

        try:
            rvar = self.filter_by_attrs(cf_role='trajectory_id')[0]
            traj_indexes = normalize_array(rvar)
            if hasattr(traj_indexes, 'mask') and np.all(traj_indexes.mask == True):  # noqa
                raise ValueError  # If they are all fill values, create an integer index
            assert traj_indexes.size == r_dim.size
        except BaseException:
            logger.warning('Could not pull trajectory values a variable with "cf_role=trajectory_id", using a computed range.')
            traj_indexes = np.arange(r_dim.size)
        try:
            pvar = self.filter_by_attrs(cf_role='profile_id')[0]
            profile_indexes = normalize_array(pvar)
            if hasattr(profile_indexes, 'mask') and np.all(profile_indexes.mask == True):  # noqa
                raise ValueError  # If they are all fill values, create an integer index
            assert profile_indexes.size == p_dim.size
        except BaseException:
            logger.warning('Could not pull profile values from a variable with "cf_role=profile_id", using a computed range.')
            profile_indexes = np.arange(p_dim.size)

        # Profile dimension
        tvars = self.t_axes()
        if len(tvars) > 1:
            tvar = [ v for v in self.t_axes() if v.dimensions == (p_dim.name,) and getattr(v, 'axis', '').lower() == 't' ][0]
        else:
            tvar = tvars[0]

        xvars = self.x_axes()
        if len(xvars) > 1:
            xvar = [ v for v in self.x_axes() if v.dimensions == (p_dim.name,) and getattr(v, 'axis', '').lower() == 'x' ][0]
        else:
            xvar = xvars[0]

        yvars = self.y_axes()
        if len(yvars) > 1:
            yvar = [ v for v in self.y_axes() if v.dimensions == (p_dim.name,) and getattr(v, 'axis', '').lower() == 'y' ][0]
        else:
            yvar = yvars[0]

        zvars = self.z_axes()
        if len(zvars) > 1:
            zvar = [ v for v in self.z_axes() if v.dimensions == (o_dim.name,) and getattr(v, 'axis', '').lower() == 'z' ][0]
        else:
            zvar = zvars[0]

        p = np.ma.masked_all(o_dim.size, dtype=profile_indexes.dtype)
        r = np.ma.masked_all(o_dim.size, dtype=traj_indexes.dtype)
        t = np.ma.masked_all(o_dim.size, dtype=tvar.dtype)
        x = np.ma.masked_all(o_dim.size, dtype=xvar.dtype)
        y = np.ma.masked_all(o_dim.size, dtype=yvar.dtype)
        si = 0

        for i in np.arange(profile_indexes.size):
            ei = si + o_index_var[i]
            p[si:ei] = profile_indexes[i]
            r[si:ei] = np.array(traj_indexes[r_index_var[i]])
            t[si:ei] = tvar[i]
            x[si:ei] = xvar[i]
            y[si:ei] = yvar[i]
            si = ei

        #  T
        nt = get_masked_datetime_array(t, tvar).flatten()

        # X and Y
        x = generic_masked(x, minv=-180, maxv=180)
        y = generic_masked(y, minv=-90, maxv=90)

        # Sample dimension
        z = generic_masked(zvar[:].flatten(), attrs=self.vatts(zvar.name))

        df_data = OrderedDict([
            ('t', nt),
            ('x', x),
            ('y', y),
            ('z', z),
            ('trajectory', r),
            ('profile', p)
        ])

        building_index_to_drop = np.ones(o_dim.size, dtype=bool)
        extract_vars = list(set(self.data_vars() + self.ancillary_vars()))
        for i, dvar in enumerate(extract_vars):

            # Profile dimensions
            if dvar.dimensions == (p_dim.name,):
                vdata = np.ma.masked_all(o_dim.size, dtype=dvar.dtype)
                si = 0
                for j in np.arange(profile_indexes.size):
                    ei = si + o_index_var[j]
                    vdata[si:ei] = dvar[j]
                    si = ei

            # Sample dimensions
            elif dvar.dimensions == (o_dim.name,):
                vdata = generic_masked(dvar[:].flatten(), attrs=self.vatts(dvar.name))

            else:
                logger.warning("Skipping variable {}... it didn't seem like a data variable".format(dvar))

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
