#!python
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd

from pocean import logger as L  # noqa
from pocean.cf import cf_safe_name, CFDataset
from pocean.dsg.trajectory import trajectory_calculated_metadata
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


class ContiguousRaggedTrajectory(CFDataset):

    @classmethod
    def is_mine(cls, dsg, strict=False):
        try:
            rvars = dsg.filter_by_attrs(cf_role='trajectory_id')
            assert len(rvars) == 1
            assert dsg.featureType.lower() == 'trajectory'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

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

        # Should never be a CR file with one trajectory so we ignore the "reduce_dims" attribute
        _ = kwargs.pop('reduce_dims', False)  # noqa
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

        with ContiguousRaggedTrajectory(output, 'w') as nc:

            trajectory_groups = df.groupby(axes.trajectory)
            unique_trajectories = list(trajectory_groups.groups.keys())
            num_trajectories = len(unique_trajectories)
            nc.createDimension(daxes.trajectory, num_trajectories)
            trajectory = nc.createVariable(axes.trajectory, get_dtype(df[axes.trajectory]), (daxes.trajectory,))

            # Get unique obs by grouping on traj getting the max size
            if unlimited is True:
                nc.createDimension(daxes.sample, None)
            else:
                nc.createDimension(daxes.sample, len(df))

            # Number of observations in each trajectory
            row_size = nc.createVariable('rowSize', 'i4', (daxes.trajectory,))

            attributes = dict_update(nc.nc_attributes(axes, daxes), kwargs.pop('attributes', {}))

            # Variables defined on only the trajectory axis
            traj_vars = kwargs.pop('traj_vars', [])
            traj_columns = [ p for p in traj_vars if p in df.columns ]
            for c in traj_columns:
                var_name = cf_safe_name(c)
                if var_name not in nc.variables:
                    create_ncvar_from_series(
                        nc,
                        var_name,
                        (daxes.trajectory,),
                        df[c],
                    )

            for i, (trajid, trg) in enumerate(trajectory_groups):
                trajectory[i] = trajid
                row_size[i] = len(trg)

                # Save any trajectory variables using the first value found
                # in the column.
                for c in traj_columns:
                    var_name = cf_safe_name(c)
                    if var_name not in nc.variables:
                        continue
                    v = nc.variables[var_name]
                    vvalues = get_ncdata_from_series(trg[c], v)[0]
                    try:
                        v[i] = vvalues
                    except BaseException:
                        L.exception(f'Failed to add {c}')
                        continue

            # Add all of the columns based on the sample dimension. Take all columns and remove the
            # trajectory, rowSize and other trajectory based columns.
            sample_columns = [
                f for f in df.columns if f not in traj_columns + ['rowSize', axes.trajectory]
            ]
            for c in sample_columns:
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
            if 'crs' not in nc.variables:
                nc.createVariable('crs', 'i4')

            # Set attributes
            nc.update_attributes(attributes)

        return ContiguousRaggedTrajectory(output, **kwargs)

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))
        if df is None:
            df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows, axes=axes)
        return trajectory_calculated_metadata(df, axes, geometries)

    def to_dataframe(self, clean_cols=True, clean_rows=True, **kwargs):
        axes = get_default_axes(kwargs.pop('axes', {}))

        axv = get_mapped_axes_variables(self, axes)

        o_index_var = self.filter_by_attrs(sample_dimension=lambda x: x is not None)
        if not o_index_var:
            raise ValueError(
                'Could not find the "sample_dimension" attribute on any variables, '
                'is this a valid {}?'.format(self.__class__.__name__)
            )
        else:
            o_index_var = o_index_var[0]
            o_dim = self.dimensions[o_index_var.sample_dimension]  # Sample dimension
            t_dim = o_index_var.dimensions

        # Trajectory
        row_sizes = o_index_var[:]
        traj_data = normalize_countable_array(axv.trajectory)
        traj_data = np.repeat(traj_data, row_sizes)

        # time
        time_data = get_masked_datetime_array(axv.t[:], axv.t).flatten()

        df_data = OrderedDict([
            (axes.t, time_data),
            (axes.trajectory, traj_data)
        ])

        building_index_to_drop = np.ones(o_dim.size, dtype=bool)

        extract_vars = copy(self.variables)
        # Skip the time and row index variables
        del extract_vars[o_index_var.name]
        del extract_vars[axes.t]

        for i, (dnam, dvar) in enumerate(extract_vars.items()):

            # Trajectory dimensions
            if dvar.dimensions == t_dim:
                vdata = np.repeat(generic_masked(dvar[:], attrs=self.vatts(dnam)), row_sizes)

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
                'featureType': 'trajectory',
                'cdm_data_type': 'Trajectory'
            },
            axes.trajectory: {
                'cf_role': 'trajectory_id',
                'long_name' : 'trajectory identifier',
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
            'rowSize': {
                'sample_dimension': daxes.sample
            }
        })
