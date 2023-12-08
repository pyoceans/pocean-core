#!python
import decimal
import itertools
import operator
import uuid
from collections import Counter, namedtuple
from collections.abc import Mapping
from datetime import date, datetime, time

import simplejson as json

try:
    # PY2 support
    from urlparse import urlparse as uparse
except ImportError:
    from urllib.parse import urlparse as uparse

import netCDF4 as nc4
import numpy as np
import pandas as pd
from cftime import date2num, num2date, num2pydate
from cftime import datetime as cfdt

from . import logger

L = logger


def safe_issubdtype(source, goal):
    try:
        return np.issubdtype(source, goal)
    except TypeError as te:
        L.error(f"Error testing issubdtype - {te}: {source}")
        return False


def downcast_dataframe(df):
    for column in df:
        if safe_issubdtype(df[column].dtype, np.int64):
            df[column] = df[column].astype(np.int32)

    return df


def nativize_times(df):
    for column in df:
        try:
            # datetime64 columns will not raise here
            if df[column].dt.tz is not None:
                df[column] = df[column].dt.tz_convert('UTC').dt.tz_convert(None)
        except AttributeError:
            pass

    return df


def is_url(url):
    try:
        result = uparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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

    axes = axes.copy()
    # Sample is only a dimension to remove from duplicate calc
    sample_dim = axes.pop('sample', 'obs')

    # Make sure there are no duplicate values for axis names
    counts = Counter(axes.values())
    for v, c in counts.items():
        if c > 1:
            raise ValueError(f"Axis value '{v}' appears twice.")

    default_axes = {
        'trajectory': 'trajectory',
        'station':    'station',
        'profile':    'profile',
        'sample':     sample_dim,
        't':          't',
        'x':          'x',
        'y':          'y',
        'z':          'z',
    }

    return namedtuple_with_defaults(
        'AxisDefaults',
        'trajectory station profile sample t x y z',
        default_axes
    )(**axes)


def all_subclasses(cls, skips=None):
    """ Recursively generate of all the subclasses of class cls. """
    if skips is None:
        skips = []

    for subclass in list(set(cls.__subclasses__())):
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
    # This is for single-value variables. netCDF4 converts them to a single string
    if var.dtype in [str]:
        # Python 2 on netCDF4 'string' variables needs this.
        # Python 3 returns false for np.issubdtype(var.dtype, 'S1')
        return var[:]

    elif hasattr(var.dtype, 'kind') and var.dtype.kind in ['U', 'S']:

        if var.size == 1:
            return var[:]

        if var.dtype.kind == 'S':
            def decoder(x):
                if hasattr(x, 'decode'):
                    return str(x.decode('utf-8'))
                else:
                    return str(x)
            vfunc = np.vectorize(decoder)
            return vfunc(nc4.chartostring(var[:]))
        else:
            return nc4.chartostring(var[:])

    else:
        return var[:]


def normalize_countable_array(cvar, count_if_none=None):
    try:
        p = normalize_array(cvar)
        if isinstance(p, str):
            p = np.asarray([p], dtype=str)
        elif hasattr(p, 'mask') and np.all(p.mask == True):  # noqa
            raise ValueError('All countable values were masked!')
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
        L.warning(f"Could not convert {value} to type {zdtype}")
        return None


def generic_masked(arr, attrs=None, minv=None, maxv=None, mask_nan=True):
    """
    Returns a masked array with anything outside of values masked.
    The minv and maxv parameters take precedence over any dict values.
    The valid_range attribute takes precedence over the valid_min and
    valid_max attributes.
    """

    # Get the min/max of values that the hardware supports
    if safe_issubdtype(arr.dtype, np.integer):
        ifunc = np.iinfo
    elif safe_issubdtype(arr.dtype, np.floating):
        ifunc = np.finfo
    else:
        if arr.dtype.kind in ['U', 'S']:
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
    return val.item()


def get_fill_value(var):
    if hasattr(var, 'missing_value'):
        return var.missing_value
    elif hasattr(var, '_FillValue'):
        return var._FillValue
    return None


def create_ncvar_from_series(ncd, var_name, dimensions, series, **kwargs):
    from pocean.cf import CFDataset
    kwargs["zlib"] = kwargs.get("zlib", True)
    kwargs["complevel"] = kwargs.get("complevel", 1)

    if safe_issubdtype(series.dtype, np.datetime64):
        # Datetimes always saved as float64
        fv = np.dtype('f8').type(CFDataset.default_fill_value)
        v = ncd.createVariable(var_name, 'f8', dimensions, fill_value=fv, **kwargs)
        v.units = CFDataset.default_time_unit
        v.calendar = 'standard'
    elif series.dtype.kind in ['U', 'S'] or series.dtype in [str]:
        # AttributeError: cannot set _FillValue attribute for VLEN or compound variable
        v = ncd.createVariable(var_name, get_dtype(series), dimensions, **kwargs)
    elif series.dtype == object:
        # Try to downcast to an int and then just take the type of the result
        # If we can't convert to a numeric use a string
        try:
            filled_down = pd.to_numeric(series.fillna(0), downcast='integer')
            # Catch boolean values... to_numeric() results in boolean for True / False
            if safe_issubdtype(filled_down.dtype, np.bool_):
                raise ValueError('datatype error: boolean needs to be converted to string')
        except BaseException:
            # Fall back to a string type
            kwargs.update({"zlib": False})
            v = ncd.createVariable(var_name, get_dtype(series), dimensions, **kwargs)
        else:
            v = ncd.createVariable(
                var_name,
                get_dtype(filled_down),
                dimensions,
                fill_value=filled_down.dtype.type(CFDataset.default_fill_value),
                **kwargs
            )
    else:
        v = ncd.createVariable(
            var_name,
            get_dtype(series),
            dimensions,
            fill_value=series.dtype.type(CFDataset.default_fill_value),
            **kwargs
        )

    return v


def _safe_date2num(dtime, units, calendar):
    return np.nan if pd.isnull(dtime) else date2num(dtime, units, calendar)


def get_ncdata_from_series(series, ncvar, fillna=True):
    from pocean.cf import CFDataset

    if safe_issubdtype(series.dtype, np.datetime64):
        units = getattr(ncvar, 'units', CFDataset.default_time_unit)
        calendar = getattr(ncvar, 'calendar', 'standard')
        nums = np.array([_safe_date2num(dtime, units=units, calendar=calendar) for dtime in series])
        return np.ma.masked_invalid(nums)
    else:
        if fillna is True:
            fv = get_fill_value(ncvar) or np.nan
            return series.fillna(fv).values.astype(ncvar.dtype)
        else:
            return series.values.astype(ncvar.dtype)


def get_masked_datetime_array(t, tvar, mask_nan=True):
    # If we are passed in a scalar... return a scalar
    if isinstance(t, np.ma.core.MaskedConstant):
        return t
    elif np.isscalar(t):
        return num2date(t, tvar.units, getattr(tvar, 'calendar', 'standard'))

    if mask_nan is True:
        t = np.ma.masked_invalid(t)

    t_cal = getattr(tvar, 'calendar', 'standard')

    # Get the min value we can have and mask anything else
    # This is limited by **python** datetime objects and not
    # nc4 objects. The min nc4 datetime object is
    # min_date = nc4.netcdftime.datetime(-4713, 1, 1, 12, 0, 0, 40)
    # There is no max date for nc4.
    min_nums = date2num([datetime.min, datetime.max], tvar.units, t_cal)
    t = np.ma.masked_outside(t, *min_nums)

    dts = num2pydate(t, tvar.units, t_cal)
    if isinstance(dts, (datetime, cfdt)):
        dts = np.array([dts.isoformat()], dtype='datetime64')

    return dts


def get_mapped_axes_variables(ncd, axes=None, skip=None):
    axes = get_default_axes(axes or {})
    skip = skip or []

    ax = namedtuple('AxisVariables', 'trajectory station profile t x y z')

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

    # Z
    if axes.z in ncd.variables:
        zvar = ncd.variables[axes.z]
    else:
        try:
            zvar = ncd.z_axes()[0]
        except IndexError:
            zvar = None

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
    for k, v in u.items():
        if isinstance(d, Mapping):
            if isinstance(v, Mapping):
                r = dict_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k] }
    return d


class JSONEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a list
        """
        try:
            from pathlib import Path
        except ImportError:
            Path = str

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()
        elif isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        elif isinstance(obj, (decimal.Decimal, uuid.UUID)):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return json.JSONEncoder.default(self, obj)
