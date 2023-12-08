#!python
import itertools
import os
import re
from datetime import datetime

from . import logger
from .dataset import EnhancedDataset
from .utils import all_subclasses, is_url


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

        if not is_url(path):
            path = os.path.realpath(path)

        subs = list(all_subclasses(cls))

        dsg = None
        try:
            dsg = cls(path)
            for klass in subs:
                logger.debug(f'Trying {klass.__name__}...')
                if hasattr(klass, 'is_mine'):
                    if klass.is_mine(dsg):
                        return klass(path)
        except OSError:
            raise
        finally:
            if hasattr(dsg, 'close'):
                dsg.close()

        subnames = ', '.join([ s.__name__ for s in subs ])
        raise ValueError(
            'Could not open {} as any type of CF Dataset. Tried: {}.'.format(
                path,
                subnames
            )
        )

    def axes(self, name):
        return getattr(self, f'{name.lower()}_axes')()

    def t_axes(self):

        # If there is only one variable with the axis parameter, return it
        hasaxis = self.filter_by_attrs(axis=lambda x: x and str(x).lower() == 't')
        if len(hasaxis) == 1:
            return hasaxis

        tvars = list(set(itertools.chain(
            hasaxis,
            self.filter_by_attrs(standard_name=lambda x: x in ['time', 'forecast_reference_time'])
        )))
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

        # If there is only one variable with the axis parameter, return it
        hasaxis = self.filter_by_attrs(axis=lambda x: x and str(x).lower() == 'x')
        if len(hasaxis) == 1:
            return hasaxis

        xvars = list(set(itertools.chain(
            hasaxis,
            self.filter_by_attrs(standard_name=lambda x: x and str(x).lower() in xnames),
            self.filter_by_attrs(units=lambda x: x and str(x).lower() in xunits)
        )))
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

        # If there is only one variable with the axis parameter, return it
        hasaxis = self.filter_by_attrs(axis=lambda x: x and str(x).lower() == 'y')
        if len(hasaxis) == 1:
            return hasaxis

        yvars = list(set(itertools.chain(
            hasaxis,
            self.filter_by_attrs(standard_name=lambda x: x and str(x).lower() in ynames),
            self.filter_by_attrs(units=lambda x: x and str(x).lower() in yunits)
        )))
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

        # If there is only one variable with the axis parameter, return it
        hasaxis = self.filter_by_attrs(axis=lambda x: x and str(x).lower() == 'z')
        if len(hasaxis) == 1:
            return hasaxis

        zvars = list(set(itertools.chain(
            hasaxis,
            self.filter_by_attrs(positive=lambda x: x and str(x).lower() in ['up', 'down']),
            self.filter_by_attrs(standard_name=lambda x: x and str(x).lower() in znames)
        )))
        return zvars

    def is_valid(self, *args, **kwargs):
        return self.__class__.is_mine(self, *args, **kwargs)

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
    if isinstance(name, str):
        if re.match('^[0-9_]', name):
            # Add a letter to the front
            name = f"v_{name}"
        return re.sub(r'[^_a-zA-Z0-9]', "_", name)

    raise ValueError(f'Could not convert "{name}" to a safe name')
