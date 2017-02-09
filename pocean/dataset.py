#!python
# coding=utf-8
import numpy as np
import simplejson as json
from netCDF4 import Dataset

from .utils import BasicNumpyEncoder
from . import logger


class EnhancedDataset(Dataset):

    def __del__(self):
        try:
            self.close()
        except RuntimeError:
            pass

    def close(self):
        if not self.isopen():
            return

        super(EnhancedDataset, self).close()

    def vatts(self, vname):
        d = {}
        var = self.variables[vname]
        for k in var.ncattrs():
            d[k] = var.getncattr(k)
        return d

    def filter_by_attrs(self, *args, **kwargs):
        return self.get_variables_by_attributes(*args, **kwargs)

    def json_attributes(self, vfuncs=None):
        """
        vfuncs can be any callable that accepts a single argument, the
        Variable object, and returns a dictionary of new attributes to
        set. These will overwrite existing attributes
        """

        vfuncs = vfuncs or []

        js = {'global': {}}

        for k in self.ncattrs():
            js['global'][k] = self.getncattr(k)

        for varname, var in self.variables.items():
            js[varname] = {}
            for k in var.ncattrs():
                z = var.getncattr(k)
                try:
                    assert not np.isnan(z).all()
                    js[varname][k] = z
                except AssertionError:
                    js[varname][k] = None
                except TypeError:
                    js[varname][k] = z

            for vf in vfuncs:
                try:
                    js[varname].update(vfuncs(var))
                except BaseException:
                    logger.exception("Could not apply custom variable attribue function")

        return json.loads(json.dumps(js, cls=BasicNumpyEncoder))

    def update_attributes(self, attributes):
        for k, v in attributes.pop('global', {}).items():
            try:
                self.setncattr(k, v)
            except BaseException:
                logger.warning('Could not set global attribute {}: {}'.format(k, v))

        for k, v in attributes.items():
            if k in self.variables:
                for n, z in v.items():
                    try:
                        self.variables[k].setncattr(n, z)
                    except BaseException:
                        logger.warning('Could not set attribute {} on {}'.format(n, k))
        self.sync()
