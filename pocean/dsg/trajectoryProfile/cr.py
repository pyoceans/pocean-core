#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd

from pocean import logger as L  # noqa
from pocean.cf import cf_safe_name, CFDataset
from pocean.dsg.trajectoryProfile import trajectory_profile_calculated_metadata
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


class ContiguousRaggedTrajectoryProfile(CFDataset):

    @classmethod
    def is_mine(cls, dsg, strict=False):
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
            if strict is True:
                raise
            return False

        return True

    @classmethod
    def from_dataframe(cls, df, output, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        daxes = axes

        _ = kwargs.pop('reduce_dims', False)
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

        with ContiguousRaggedTrajectoryProfile(output, 'w') as nc:

            trajectory_groups = df.groupby(axes.trajectory)
            unique_trajectories = list(trajectory_groups.groups.keys())
            num_trajectories = len(unique_trajectories)

            nc.createDimension(daxes.trajectory, num_trajectories)
            trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]), (daxes.trajectory,))
            trajectory[:] = np.array(unique_trajectories)

            # Calculate the max number of profiles
            unique_profiles = df[axes.profile].unique()
            num_profiles = len(unique_profiles)

            nc.createDimension(daxes.profile, num_profiles)
            profile = nc.createVariable(axes.profile, get_dtype(df[axes.profile]), (daxes.profile,))
            profile[:] = np.array(unique_profiles)

            # Get unique obs by grouping on traj and profile and getting the max size
            num_obs = len(df)
            nc.createDimension(daxes.sample, num_obs)

            # The trajectory this profile belongs to
            t_ind = nc.createVariable('trajectoryIndex', 'i4', (daxes.profile,))
            # Number of observations in each profile
            row_size = nc.createVariable('rowSize', 'i4', (daxes.profile,))

            # Create all of the axis variables
            time = nc.createVariable(axes.t, 'f8', (daxes.profile,), fill_value=np.dtype('f8').type(cls.default_fill_value))
            latitude = nc.createVariable(axes.y, get_dtype(df[axes.y]), (daxes.profile,), fill_value=df[axes.y].dtype.type(cls.default_fill_value))
            longitude = nc.createVariable(axes.x, get_dtype(df[axes.x]), (daxes.profile,), fill_value=df[axes.x].dtype.type(cls.default_fill_value))

            # Axes variables are already processed so skip them
            data_columns = [ d for d in df.columns if d not in axes ]
            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            # Variables defined on only the profile axis
            profile_vars = kwargs.pop('profile_vars', [])
            profile_columns = [ p for p in profile_vars if p in data_columns ]
            for c in profile_columns:
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    create_ncvar_from_series(
                        nc,
                        var_name,
                        (daxes.profile,),
                        df[c],
                    )

            for i, (_, trg) in enumerate(trajectory_groups):
                for j, (_, pfg) in enumerate(trg.groupby(axes.profile)):
                    time[j] = get_ncdata_from_series(pfg[axes.t], time).astype('f8')[0]
                    latitude[j] = get_ncdata_from_series(pfg[axes.y], latitude)[0]
                    longitude[j] = get_ncdata_from_series(pfg[axes.x], longitude)[0]
                    row_size[j] = len(pfg)
                    t_ind[j] = i

                    # Save any profile variables on the "profile" index using the first value found
                    # in the column.
                    for c in profile_columns:
                        var_name = cf_safe_name(c)
                        if var_name not in nc.variables:
                            continue
                        v = nc.variables[var_name]
                        vvalues = get_ncdata_from_series(pfg[c], v)[0]
                        try:
                            v[j] = vvalues
                        except BaseException:
                            L.exception(f'Failed to add {c}')
                            continue

            # Add back in the z axes that was removed when calculating data_columns
            # and ignore variables that were stored in the profile index
            sample_columns = [ f for f in data_columns + [axes.z] if f not in profile_columns ]
            skips = ['trajectoryIndex', 'rowSize']
            for c in [ d for d in sample_columns if d not in skips ]:
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
                    v[:] = vvalues.reshape(v.shape)
                except BaseException:
                    L.exception(f'Failed to add {c}')
                    continue

            # Metadata variables
            if 'crs' not in nc.variables:
                nc.createVariable('crs', 'i4')

            # Set attributes
            nc.update_attributes(attributes)

        return ContiguousRaggedTrajectoryProfile(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        return trajectory_profile_calculated_metadata(df, axes, geometries)

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes)

        # The index variable (trajectory_index) is identified by having an
        # attribute with name of instance_dimension whose value is the instance
        # dimension name (trajectory in this example). The index variable must
        # have the profile dimension as its sole dimension, and must be type
        # integer. Each value in the index variable is the zero-based trajectory
        # index that the profile belongs to i.e. profile p belongs to trajectory
        # i=trajectory_index(p), as in section H.2.5.
        r_index_var = self.filter_by_attrs(instance_dimension=lambda x: x is not None)
        if not r_index_var:
            raise ValueError(
                'Could not find the "instance_dimension" attribute on any variables, '
                'is this a valid {}?'.format(self.__class__.__name__)
            )
        else:
            r_index_var = r_index_var[0]
        p_dim = self.dimensions[r_index_var.dimensions[0]]       # Profile dimension

        # We should probably use this below to test for dimensionality of variables?
        # r_dim = self.dimensions[r_index_var.instance_dimension]  # Trajectory dimension

        # The count variable (row_size) contains the number of elements for
        # each profile, which must be written contiguously. The count variable
        # is identified by having an attribute with name sample_dimension whose
        # value is the sample dimension (obs in this example) being counted. It
        # must have the profile dimension as its sole dimension, and must be
        # type integer
        o_index_var = self.filter_by_attrs(sample_dimension=lambda x: x is not None)
        if not o_index_var:
            raise ValueError(
                'Could not find the "sample_dimension" attribute on any variables, '
                'is this a valid {}?'.format(self.__class__.__name__)
            )
        else:
            o_index_var = o_index_var[0]
        o_dim = self.dimensions[o_index_var.sample_dimension]  # Sample dimension

        profile_indexes = normalize_countable_array(axv.profile, count_if_none=p_dim.size)
        p = np.ma.masked_all(o_dim.size, dtype=profile_indexes.dtype)

        traj_indexes = normalize_countable_array(axv.trajectory)
        r = np.ma.masked_all(o_dim.size, dtype=traj_indexes.dtype)

        tvar = axv.t
        t = np.ma.masked_all(o_dim.size, dtype=tvar.dtype)

        xvar = axv.x
        x = np.ma.masked_all(o_dim.size, dtype=xvar.dtype)

        yvar = axv.y
        y = np.ma.masked_all(o_dim.size, dtype=yvar.dtype)
        si = 0

        # Sample (obs) dimension
        zvar = axv.z
        z = generic_masked(zvar[:].flatten(), attrs=self.vatts(zvar.name))

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

        df_data = OrderedDict([
            (axes.t, nt),
            (axes.x, x),
            (axes.y, y),
            (axes.z, z),
            (axes.trajectory, r),
            (axes.profile, p)
        ])

        building_index_to_drop = np.ones(o_dim.size, dtype=bool)

        extract_vars = copy(self.variables)
        # Skip the traj and row index variables
        del extract_vars[o_index_var.name]
        del extract_vars[r_index_var.name]
        # Axes variables are already processed so skip them
        for ncvar in axv._asdict().values():
            if ncvar is not None and ncvar.name in extract_vars:
                del extract_vars[ncvar.name]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):

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
                'featureType': 'trajectoryProfile',
                'cdm_data_type': 'TrajectoryProfile'
            },
            axes.trajectory: {
                'cf_role': 'trajectory_id',
                'long_name' : 'trajectory identifier',
                'ioos_category': 'identifier'
            },
            axes.profile: {
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
            'trajectoryIndex': {
                'instance_dimension': daxes.trajectory
            },
            'rowSize': {
                'sample_dimension': daxes.sample
            }
        })
