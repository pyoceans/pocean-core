#!python
# coding=utf-8
import six
import base64
import operator
import itertools
import simplejson as json
from datetime import datetime
from collections import namedtuple, Mapping, Counter

import numpy as np
import netCDF4 as nc4

from . import logger
L = logger


def downcast_dataframe(df):
    for column in df:
        if np.issubdtype(df[column].dtype, np.int64):
            df[column] = df[column].astype(np.int32)
    return df


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def get_default_axes(axes=None):
    axes = axes or {}
    if isinstance(axes, tuple):
        axes = axes._asdict()

    # Make sure there are no duplicate values for axis names
    counts = Counter(axes.values())
    for v, c in counts.items():
        if c > 1:
            raise ValueError("Axis value '{}' appears twice.".format(v))

    default_axes = {
        'trajectory': 'trajectory',
        'station':    'station',
        'profile':    'profile',
        't':          't',
        'x':          'x',
        'y':          'y',
        'z':          'z',
    }

    return namedtuple_with_defaults(
        'AxisDefaults',
        'trajectory station profile t x y z',
        default_axes
    )(**axes)


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
    if np.issubdtype(var.dtype, np.str_) or np.issubdtype(var.dtype, np.bytes_):
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


def normalize_countable_array(cvar, count_if_none=None):
    try:
        p = normalize_array(cvar)
        if isinstance(p, six.string_types):
            p = np.asarray([p])
        elif hasattr(p, 'mask') and np.all(p.mask == True):  # noqa
            # All masked out so compute the index by counting!
            raise ValueError
    except BaseException:
        L.warning('Could not pull a countable array... using a calculated index')
        if cvar is None and count_if_none is not None:
            p = np.asarray(list(range(int(count_if_none))), dtype=np.integer)
        else:
            p = np.asarray(list(range(len(cvar))), dtype=np.integer)

    return p


def safe_attribute_typing(zdtype, value):
    try:
        return zdtype.type(value)
    except ValueError:
        L.warning("Could not convert {} to type {}".format(value, zdtype))
        return None


def generic_masked(arr, attrs=None, minv=None, maxv=None, mask_nan=True):
    """
    Returns a masked array with anything outside of values masked.
    The minv and maxv parameters take precendence over any dict values.
    The valid_range attribute takes precendence over the valid_min and
    valid_max attributes.
    """

    # Get the min/max of values that the hardware supports
    if np.issubdtype(arr.dtype, np.integer):
        ifunc = np.iinfo
    elif np.issubdtype(arr.dtype, np.floating):
        ifunc = np.finfo
    else:
        if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_):
            mask_nan = False

        if mask_nan is True:
            return np.ma.masked_array(np.ma.fix_invalid(arr))
        else:
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


def create_ncvar_from_series(ncd, var_name, dimensions, series):
    from pocean.cf import CFDataset

    if np.issubdtype(series.dtype, np.datetime64):
        # Datetimes always saved as float64
        fv = np.dtype('f8').type(CFDataset.default_fill_value)
        v = ncd.createVariable(var_name, 'f8', dimensions, fill_value=fv)
        v.units = CFDataset.default_time_unit
        v.calendar = 'standard'
    elif np.issubdtype(series.dtype, np.str_) or np.issubdtype(series.dtype, np.bytes_) or series.dtype == object:
        # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
        v = ncd.createVariable(var_name, get_dtype(series), dimensions)
    else:
        v = ncd.createVariable(var_name, get_dtype(series), dimensions, fill_value=series.dtype.type(CFDataset.default_fill_value))

    return v


def get_ncdata_from_series(series, ncvar):
    from pocean.cf import CFDataset

    if np.issubdtype(series.dtype, np.datetime64):
        units = getattr(ncvar, 'units', CFDataset.default_time_unit)
        calendar = getattr(ncvar, 'calendar', 'standard')
        nums = nc4.date2num(series.tolist(), units=units, calendar=calendar)
        return np.ma.masked_invalid(nums)
    else:
        fv = get_fill_value(ncvar) or np.nan
        return series.fillna(fv).values


def get_masked_datetime_array(t, tvar, mask_nan=True):
    # If we are passed in a scalar... return a scalar
    if isinstance(t, np.ma.core.MaskedConstant):
        return t
    elif np.isscalar(t):
        return nc4.num2date(t, tvar.units, getattr(tvar, 'calendar', 'standard'))

    if mask_nan is True:
        t = np.ma.masked_invalid(t)

    t_cal = getattr(tvar, 'calendar', 'standard')

    # Get the min value we can have and mask anything else
    # This is limied by **python** datetime objects and not
    # nc4 objects. The min nc4 datetime object is
    # min_date = nc4.netcdftime.datetime(-4713, 1, 1, 12, 0, 0, 40)
    # There is no max date for nc4.
    min_nums = nc4.date2num([datetime.min, datetime.max], tvar.units, t_cal)
    t = np.ma.masked_outside(t, *min_nums)
    # Avoid deprecation warnings between numpy 1.11 and 1.14
    # After 1.14 this is the default behavior
    t._sharedmask = False
    # Temporarily set to 1 so num2date works
    t_mask = np.copy(np.ma.getmaskarray(t))
    t[t_mask] = 1

    dts = nc4.num2date(t, tvar.units, t_cal)
    if isinstance(dts, datetime):
        dts = np.array([dts.isoformat()], dtype='datetime64')

    # Patch the time variable back to its original mask, since num2date
    # breaks any missing/fill values
    nt = np.ma.MaskedArray(dts)
    nt[t_mask] = np.ma.masked
    return nt


def get_mapped_axes_variables(ncd, axes=None, skip=None):
    axes = get_default_axes(axes or {})
    skip = skip or []

    ax = namedtuple('AxisVariables', 'trajectory station profile t x y z')

    # Z
    if axes.z in ncd.variables:
        zvar = ncd.variables[axes.z]
    else:
        zvar = ncd.z_axes()[0]

    # T
    if axes.t in ncd.variables:
        tvar = ncd.variables[axes.t]
    else:
        tvar = ncd.t_axes()[0]

    # X
    if axes.x in ncd.variables:
        xvar = ncd.variables[axes.x]
    else:
        xvar = ncd.x_axes()[0]

    # Y
    if axes.y in ncd.variables:
        yvar = ncd.variables[axes.y]
    else:
        yvar = ncd.y_axes()[0]

    # Trajectory
    if axes.trajectory in skip:
        rvar = None
    elif axes.trajectory in ncd.variables:
        rvar = ncd.variables[axes.trajectory]
    else:
        try:
            rvar = ncd.filter_by_attrs(cf_role='trajectory_id')[0]
        except IndexError:
            rvar = None

    # Profile
    if axes.profile in skip:
        pvar = None
    elif axes.profile in ncd.variables:
        pvar = ncd.variables[axes.profile]
    else:
        try:
            pvar = ncd.filter_by_attrs(cf_role='profile_id')[0]
        except IndexError:
            pvar = None

    # Station
    if axes.station in skip:
        svar = None
    elif axes.station in ncd.variables:
        svar = ncd.variables[axes.station]
    else:
        try:
            svar = ncd.filter_by_attrs(cf_role='timeseries_id')[0]
        except IndexError:
            svar = None

    return ax(
        rvar,
        svar,
        pvar,
        tvar,
        xvar,
        yvar,
        zvar
    )


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
