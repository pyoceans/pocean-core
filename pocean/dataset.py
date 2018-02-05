#!python
# coding=utf-8
import warnings
from collections import OrderedDict

import numpy as np
import simplejson as json
from netCDF4 import Dataset

from .utils import (
    BasicNumpyEncoder,
)
from .meta import (
    MetaInterface,
    ncpyattributes,
    string_to_dtype,
    untype_attributes
)
from . import logger as L


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

    def __apply_meta_interface__(self, meta):
        warnings.warn(
            '`__apply_meta_interface__` is deprecated. Use `apply_meta()` instead',
            DeprecationWarning
        )
        return self.apply_meta(meta)

    def __getattr__(self, name):
        if name in ['__meta_interface__', '_meta']:
            warnings.warn(
                '`__meta_interface__` and `_meta` are deprecated. Use `meta()` instead',
                DeprecationWarning
            )
            return self.meta()
        else:
            return super().__getattr__(name)

    def apply_meta(self, meta, create_vars=True, create_dims=True):
        """Apply a meta interface object to a netCDF4 compatible object"""
        ds = meta.get('dimensions', OrderedDict())
        gs = meta.get('attributes', OrderedDict())
        vs = meta.get('variables', OrderedDict())

        # Dimensions
        for dname, dsize in ds.items():
            # Ignore dimension sizes less than 0
            if dsize and dsize < 0:
                continue
            if dname not in self.dimensions:

                # Don't create new dimensions
                if create_dims is False:
                    continue

                self.createDimension(dname, size=dsize)
            else:
                dfilesize = self.dimensions[dname].size
                if dfilesize != dsize:
                    L.warning("Not changing size of dimension {}. file: {}, meta: {}".format(
                        dname, dfilesize, dsize
                    ))

        # Global attributes
        typed_gs = untype_attributes(gs)
        self.setncatts(typed_gs)

        # Variables
        for vname, vvalue in vs.items():

            vatts = untype_attributes(vvalue.get('attributes', {}))

            if vname not in self.variables:

                # Don't create new variables
                if create_vars is False:
                    continue

                if 'shape' not in vvalue and 'type' not in vvalue:
                    L.debug("Skipping {} creation, no shape or no type defined".format(vname))
                    continue
                shape = vvalue.get('shape', [])  # Dimension names
                dtype = string_to_dtype(vvalue.get('type'))
                # I'm fairly certain that using 'np.ma.masked' here is safe and will always
                # translate to a NaN value for the dtype. We can't use `np.nan` or `math.nan`
                # since they are reserved for floats.
                fillmiss = vatts.get('_FillValue', vatts.get('missing_value', np.ma.masked))
                newvar = self.createVariable(
                    vname,
                    dtype,
                    dimensions=shape,
                    fill_value=dtype.type(fillmiss)
                )
            else:
                newvar = self.variables[vname]

            # Don't re-assign fill value attributes
            if '_FillValue' in vatts:
                del vatts['_FillValue']
            if 'missing_value' in vatts:
                del vatts['missing_value']

            newvar.setncatts(vatts)

    def meta(self):
        ds = OrderedDict()
        vs = OrderedDict()
        gs = ncpyattributes({ ga: self.getncattr(ga) for ga in self.ncattrs() })

        # Dimensions
        for dname, dim in self.dimensions.items():
            if dim.isunlimited():
                ds[dname] = None
            else:
                ds[dname] = dim.size

        # Variables
        for k, v in self.variables.items():

            typed = v.dtype
            if isinstance(typed, np.dtype):
                typed = str(typed.name)
            elif isinstance(typed, type):
                typed = typed.__name__

            vs[k] = {
                'attributes': ncpyattributes({ va: v.getncattr(va) for va in v.ncattrs() }),
                'shape': v.dimensions,
                'type': typed
            }

        return MetaInterface(
            dimensions=ds,
            variables=vs,
            attributes=gs
        )

    def to_json(self, *args, **kwargs):
        return json.dumps(self.to_dict(), *args, **kwargs)

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
                    L.exception("Could not apply custom variable attribue function")

        return json.loads(json.dumps(js, cls=BasicNumpyEncoder))

    def update_attributes(self, attributes):
        for k, v in attributes.pop('global', {}).items():
            try:
                self.setncattr(k, v)
            except BaseException:
                L.warning('Could not set global attribute {}: {}'.format(k, v))

        for k, v in attributes.items():
            if k in self.variables:
                for n, z in v.items():
                    try:
                        self.variables[k].setncattr(n, z)
                    except BaseException:
                        L.warning('Could not set attribute {} on {}'.format(n, k))
        self.sync()
