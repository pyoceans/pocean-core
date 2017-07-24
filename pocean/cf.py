#!python
# coding=utf-8
import os
from datetime import datetime

import six
from six import u as astr

from .utils import all_subclasses
from .dataset import EnhancedDataset
from . import logger


class CFDataset(EnhancedDataset):

    default_fill_value = -9999.9
    default_time_unit = 'seconds since 1990-01-01 00:00:00Z'

    @classmethod
    def load(cls, path):
        """Attempt to load a netCDF file as a CF compatible dataset

        Extended description of function.

        Parameters
        ----------
        path :
            Path to netCDF file

        Returns
        -------
            CFDataset subclass for your netCDF file

        Raises
        ------
            ValueError:
                If no suitable class is found for your dataset

        """

        fpath = os.path.realpath(path)
        subs = list(all_subclasses(cls))

        try:
            dsg = cls(fpath)
            for klass in subs:
                logger.debug('Trying {}...'.format(klass.__name__))
                if hasattr(klass, 'is_mine'):
                    if klass.is_mine(dsg):
                        return klass(path)
        finally:
            dsg.close()

        subnames = ', '.join([ s.__name__ for s in subs ])
        raise ValueError(
            'Could not open {} as any type of CF Dataset. Tried: {}.'.format(
                fpath,
                subnames
            )
        )

    def axes(self, name):
        return getattr(self, '{}_axes'.format(name.lower()))()

    def t_axes(self):
        tvars = list(set(
            self.filter_by_attrs(
                axis=lambda x: x and x.lower() == 't'
            ) +
            self.filter_by_attrs(
                standard_name=lambda x: x in ['time', 'forecast_reference_time']
            )
        ))
        return tvars

    def x_axes(self):
        """
        CF X axis will have one of the following:
          * The `axis` property has the value ``'X'``
          * Units of longitude (see `cf.Units.islongitude` for details)
          * The `standard_name` property is one of ``'longitude'``,
            ``'projection_x_coordinate'`` or ``'grid_longitude'``
        """
        xnames = ['longitude', 'grid_longitude', 'projection_x_coordinate']
        xunits = [
            'degrees_east',
            'degree_east',
            'degree_E',
            'degrees_E',
            'degreeE',
            'degreesE'
        ]
        xvars = list(set(
            self.filter_by_attrs(
                axis=lambda x: x and x.lower() == 'x'
            ) +
            self.filter_by_attrs(
                standard_name=lambda x: x and x.lower() in xnames
            ) +
            self.filter_by_attrs(
                units=lambda x: x and x.lower() in xunits
            )
        ))
        return xvars

    def y_axes(self):
        ynames = ['latitude', 'grid_latitude', 'projection_y_coordinate']
        yunits = [
            'degrees_north',
            'degree_north',
            'degree_N',
            'degrees_N',
            'degreeN',
            'degreesN'
        ]
        yvars = list(set(
            self.filter_by_attrs(
                axis=lambda x: x and x.lower() == 'y'
            ) +
            self.filter_by_attrs(
                standard_name=lambda x: x and x.lower() in ynames
            ) +
            self.filter_by_attrs(
                units=lambda x: x and x.lower() in yunits
            )
        ))
        return yvars

    def z_axes(self):
        znames = [
            'atmosphere_ln_pressure_coordinate',
            'atmosphere_sigma_coordinate',
            'atmosphere_hybrid_sigma_pressure_coordinate',
            'atmosphere_hybrid_height_coordinate',
            'atmosphere_sleve_coordinate',
            'ocean_sigma_coordinate',
            'ocean_s_coordinate',
            'ocean_s_coordinate_g1',
            'ocean_s_coordinate_g2',
            'ocean_sigma_z_coordinate',
            'ocean_double_sigma_coordinate'
        ]
        zvars = list(set(
            self.filter_by_attrs(
                axis=lambda x: x and x.lower() == 'z'
            ) +
            self.filter_by_attrs(
                positive=lambda x: x and x.lower() in ['up', 'down']
            ) +
            self.filter_by_attrs(
                standard_name=lambda x: x and x.lower() in znames
            )
        ))
        return zvars

    def data_vars(self):
        return self.filter_by_attrs(
            coordinates=lambda x: x is not None,
            units=lambda x: x is not None,
            standard_name=lambda x: x is not None,
            flag_values=lambda x: x is None,
            flag_masks=lambda x: x is None,
            flag_meanings=lambda x: x is None
        )

    def ancillary_vars(self):
        ancillary_variables = []
        for rv in self.filter_by_attrs(
            ancillary_variables=lambda x: x is not None
        ):
            # Space separated ancillary variables
            for av in rv.ancillary_variables.split(' '):
                if av in self.variables:
                    ancillary_variables.append(self.variables[av])
        return list(set(ancillary_variables))

    def nc_attributes(self):
        return {
            'global' : {
                'Conventions': 'CF-1.6',
                'date_created': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:00Z"),
            }
        }


def cf_safe_name(name):
    import re
    if isinstance(name, six.string_types):
        if re.match(astr('^[0-9_]'), name):
            # Add a letter to the front
            name = astr("v_{}".format(name))
        return re.sub(r'[^_a-zA-Z0-9]', astr("_"), name)

    raise ValueError(astr('Could not convert "{}" to a safe name'.format(name)))
