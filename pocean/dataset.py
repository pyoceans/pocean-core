#!python
import warnings
from collections import OrderedDict

import numpy as np
import simplejson as json
from netCDF4 import Dataset

from . import logger as L
from .meta import (
    MetaInterface,
    ncpyattributes,
    string_to_dtype,
    untype_attributes,
)
from .utils import (
    generic_masked,
    JSONEncoder,
    safe_attribute_typing,
    safe_issubdtype,
)

# Attribute that need to be of the same type as the variables
_TYPE_SENSITIVE_ATTRIBUTES = [
    '_FillValue',
    'missing_value',
    'valid_min',
    'valid_max',
    'valid_range',
    'display_min',
    'display_max',
    'display_range',
    'colorBarMinimum',
    'colorBarMaximum',
]


class EnhancedDataset(Dataset):

    def __del__(self):
        try:
            self.close()
        except RuntimeError:
            pass

    def close(self):
        if not self.isopen():
            return

        super().close()

    def vatts(self, vname):
        d = {}
        var = self.variables[vname]
        for k in var.ncattrs():
            d[k] = var.getncattr(k)
        return d

    def filter_by_attrs(self, *args, **kwargs):
        return self.get_variables_by_attributes(*args, **kwargs)

    def __apply_meta_interface__(self, meta, **kwargs):
        warnings.warn(
            '`__apply_meta_interface__` is deprecated. Use `apply_meta()` instead',
            DeprecationWarning
        )
        return self.apply_meta(meta, **kwargs)

    def __getattr__(self, name):
        if name in ['__meta_interface__', '_meta']:
            warnings.warn(
                '`__meta_interface__` and `_meta` are deprecated. Use `meta()` instead',
                DeprecationWarning
            )
            return self.meta()
        else:
            return super().__getattr__(name)

    def apply_meta(self, *args, **kwargs):
        """ Shortcut to the JSON object without writing any data"""
        kwargs['create_data'] = False
        return self.apply_json(*args, **kwargs)

    def meta(self, *args, **kwargs):
        """ Shortcut to the JSON object without any data"""
        kwargs['return_data'] = False
        return self.json(*args, **kwargs)

    def json(self, return_data=True, fill_data=True):
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

            vattrs = { va: v.getncattr(va) for va in v.ncattrs() }
            vardict = {
                'attributes': ncpyattributes(vattrs),
                'shape': v.dimensions,
                'type': typed
            }
            if return_data is True:
                vdata = generic_masked(v[:], attrs=vattrs)
                if fill_data is True:
                    vdata = vdata.filled()
                vardict['data'] = vdata.tolist()

            vs[k] = vardict

        return MetaInterface(
            dimensions=ds,
            variables=vs,
            attributes=gs
        )

    def apply_json(self, meta, create_vars=True, create_dims=True, create_data=True):
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
                    L.debug(f"Skipping {vname} creation, no shape or no type defined")
                    continue
                shape = vvalue.get('shape', [])  # Dimension names
                vardtype = string_to_dtype(vvalue.get('type'))

                if safe_issubdtype(vardtype, np.floating):
                    defaultfill = vardtype.type(np.nan)  # We can use `nan` for floats
                elif vardtype.kind in ['U', 'S']:
                    defaultfill = None  # No fillvalue on VLENs
                else:
                    # Use a masked value which evaluates to different things depending on the dtype
                    # For integers is resolves to `0`.
                    defaultfill = vardtype.type(np.ma.masked)

                fillmiss = vatts.get('_FillValue', vatts.get('missing_value', defaultfill))
                newvar = self.createVariable(
                    vname,
                    vardtype,
                    dimensions=shape,
                    fill_value=fillmiss
                )
            else:
                newvar = self.variables[vname]

            # Now assign the data if is exists
            if create_data is True and 'data' in vvalue:
                # Because the JSON format can be flattened already we are just
                # going to always reshape the data to the variable shape
                data = generic_masked(
                    np.array(vvalue['data'], dtype=newvar.dtype).flatten()
                ).reshape(newvar.shape)
                newvar[:] = data

            # Don't re-assign fill value attributes
            if '_FillValue' in vatts:
                del vatts['_FillValue']
            if 'missing_value' in vatts:
                del vatts['missing_value']

            # Convert any attribute that need to match the variables dtype to that dtype
            for sattr in _TYPE_SENSITIVE_ATTRIBUTES:
                if sattr in vatts:
                    vatts[sattr] = safe_attribute_typing(newvar.dtype, vatts[sattr])

            newvar.setncatts(vatts)

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
                    L.exception("Could not apply custom variable attribute function")

        return json.loads(json.dumps(js, cls=JSONEncoder))

    def update_attributes(self, attributes):
        for k, v in attributes.pop('global', {}).items():
            try:
                self.setncattr(k, v)
            except BaseException:
                L.warning(f'Could not set global attribute {k}: {v}')

        for k, v in attributes.items():
            if k in self.variables:
                for n, z in v.items():

                    # Don't re-assign fill value attributes
                    if n in ['_FillValue', 'missing_value']:
                        L.warning(f'Refusing to set {n} on {k}')
                        continue

                    try:
                        self.variables[k].setncattr(n, z)
                    except BaseException:
                        L.warning(f'Could not set attribute {n} on {k}')
        self.sync()
