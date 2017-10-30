#!python
# coding=utf-8
import base64
import operator
import itertools
import simplejson as json
from datetime import datetime

import numpy as np
import netCDF4 as nc4

from . import logger


def downcast_dataframe(df):
    for column in df:
        if np.issubdtype(df[column].dtype, np.int64):
            df[column] = df[column].astype(np.int32)
    return df


def all_subclasses(cls, skips=None):
    """ Recursively generate of all the subclasses of class cls. """
    if skips is None:
        skips = []

    for subclass in cls.__subclasses__():
        if subclass not in skips:
            yield subclass
        for subc in all_subclasses(subclass):
            if subclass not in skips:
                yield subc


def unique_justseen(iterable, key=None):
    "List unique elements, preserving order. Remember only the element just seen."
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    try:
        # PY2 support
        from itertools import imap as map
    except ImportError:
        from builtins import map

    return map(next, map(operator.itemgetter(1), itertools.groupby(iterable, key)))


def normalize_array(var):
    """
    Returns a normalized data array from a NetCDF4 variable. This is mostly
    used to normalize string types between py2 and py3 as well as netcdf3 and
    netcdf4. It has no effect on types other than chars/strings
    """
    if np.issubdtype(var.dtype, 'S1'):
        if var.dtype == str:
            # Python 2 on netCDF4 'string' variables needs this.
            # Python 3 returns false for np.issubdtype(var.dtype, 'S1')
            return var[:]

        def decoder(x):
            if hasattr(x, 'decode'):
                return str(x.decode('utf-8'))
            else:
                return str(x)
        vfunc = np.vectorize(decoder)
        return vfunc(nc4.chartostring(var[:]))
    else:
        return var[:]


def safe_attribute_typing(zdtype, value):
    try:
        return zdtype.type(value)
    except ValueError:
        logger.warning("Could not convert {} to type {}".format(value, zdtype))
        return None


def generic_masked(arr, attrs=None, minv=None, maxv=None, mask_nan=True):
    """
    Returns a masked array with anything outside of values masked.
    The minv and maxv parameters take precendence over any dict values.
    The valid_range attribute takes precendence over the valid_min and
    valid_max attributes.
    """
    if np.issubdtype('S', arr.dtype):
        return np.ma.masked_array(arr)

    attrs = attrs or {}

    if 'valid_min' in attrs:
        minv = safe_attribute_typing(arr.dtype, attrs['valid_min'])
    if 'valid_max' in attrs:
        maxv = safe_attribute_typing(arr.dtype, attrs['valid_max'])
    if 'valid_range' in attrs:
        vr = attrs['valid_range']
        minv = safe_attribute_typing(arr.dtype, vr[0])
        maxv = safe_attribute_typing(arr.dtype, vr[1])

    # Get the min/max of values that the hardware supports
    if np.issubdtype(arr.dtype, int):
        ifunc = np.iinfo
    elif np.issubdtype(arr.dtype, float):
        ifunc = np.finfo

    try:
        info = ifunc(arr.dtype)
    except ValueError:
        info = ifunc(arr.dtype)

    minv = minv if minv is not None else info.min
    maxv = maxv if maxv is not None else info.max

    if mask_nan is True:
        arr = np.ma.fix_invalid(arr)

    if isinstance(arr, np.ma.core.MaskedConstant):
        if arr is np.ma.masked or arr > maxv or arr < minv:
            return np.ma.masked
        return arr
    elif arr.mask.all():
        return arr
    else:
        # You can't use `masked_outside` with nan values or numpy will send a warning
        not_nan = ~np.isnan(arr)
        not_nan = not_nan.filled(True)
        arr[not_nan] = np.ma.masked_outside(
            arr[not_nan],
            minv,
            maxv
        )
        return arr


def pyscalar(val):
    return np.asscalar(val)


def get_fill_value(var):
    if hasattr(var, 'missing_value'):
        return var.missing_value
    elif hasattr(var, '_FillValue'):
        return var._FillValue
    return None


def get_masked_datetime_array(t, tvar):
    t_mask = []
    tfill = get_fill_value(tvar)
    if tfill is not None:
        t_mask = np.copy(np.ma.getmaskarray(t))
        # Temporarily set to 1 so num2date works
        t[t_mask] = 1

    dts = nc4.num2date(t, tvar.units, getattr(tvar, 'calendar', 'standard'))
    if isinstance(dts, datetime):
        dts = np.array([t.isoformat()], dtype='datetime64')

    # Patch the time variable back to its original mask, since num2date
    # breaks any missing/fill values
    nt = np.ma.MaskedArray(dts)
    nt[t_mask] = np.ma.masked
    return nt


def get_dtype(obj):
    if hasattr(obj, 'dtype'):
        if obj.dtype == object:
            return str
        return obj.dtype
    elif isinstance(obj, (tuple, list)):
        return getattr(obj[0], 'dtype', type(obj[0]))
    else:
        return type(obj)


def dict_update(d, u):
    # http://stackoverflow.com/a/3233356
    import collections
    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = dict_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k] }
    return d


class BasicNumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a list
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)
