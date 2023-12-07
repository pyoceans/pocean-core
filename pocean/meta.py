#!python
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy

import numpy as np
import simplejson as json

from . import logger


class MetaInterface(Mapping):

    VALID_KEYS = ['dimensions', 'variables', 'attributes']

    @classmethod
    def from_jsonfile(cls, jsf):
        if not os.path.isfile(jsf):
            raise ValueError(f'{jsf} is not a file')

        with open(jsf) as jf:
            return cls.from_jsonstr(jf.read())

    @classmethod
    def from_jsonstr(cls, js):
        try:
            d = json.loads(js, object_pairs_hook=OrderedDict)
        except BaseException as e:
            raise ValueError(f'Could not parse JSON string: {e}')

        return cls(d)

    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return str(self._data)


def safe_attribute_typing(zdtype, value):
    try:
        return zdtype.type(value)
    except ValueError:
        logger.warning(f"Could not convert {value} to type {zdtype}")
        return None


def string_to_dtype(type_str):

    # int - we avoid int64
    if type_str in ['int', 'int32', 'int64', 'i', 'i4', 'i8', 'i32', 'i64', 'long']:
        return np.dtype('int32')

    elif type_str in ['uint', 'ui4', 'ui', 'uint32', 'uint64', 'ui64', 'u4', 'u8']:
        return np.dtype('uint32')

    elif type_str in ['float', 'float32', 'f', 'f4', 'f32']:
        return np.dtype('float32')

    elif type_str in ['double', 'float64', 'd', 'f8', 'f64']:
        return np.dtype('float64')

    elif type_str in ['byte', 'bytes8', 'i1', 'b', 'B', 'int8']:
        return np.dtype('int8')

    elif type_str in ['ubyte', 'ui1', 'ub' 'uB', 'uint8']:
        return np.dtype('uint8')

    elif type_str in ['char', 'c', 'string', 'S1', 'str', 'unicode', 'string8']:
        return np.dtype('U')

    elif type_str in ['short', 's', 'i2', 'h', 'int16']:
        return np.dtype('int16')

    elif type_str in ['ushort', 'us', 'u2', 'ui2', 'uh', 'uint16']:
        return np.dtype('uint16')

    raise ValueError(f"Could not find dtype for {type_str}")


def untype_attributes(vd):
    typed = OrderedDict()
    for k, v in vd.items():
        if isinstance(v, dict):
            dtype = string_to_dtype(v.get('type'))
            vval = v.get('data')
            if isinstance(vval, (list, tuple)):
                safe = ( safe_attribute_typing(dtype, x) for x in vval )
                typed[k] = [ x for x in safe if x is not None ]
            else:
                safe = safe_attribute_typing(dtype, vval)
                if safe is not None:
                    typed[k] = safe
        else:
            typed[k] = v
    return typed


def ncpyattributes(obj, verbose=True):
    """ Converts any attributes that are not native python types to those types """

    return_copy = deepcopy(obj)

    for k, v in obj.items():

        if isinstance(v, np.ndarray):
            newv = v.tolist()
        elif hasattr(v, 'dtype'):
            newv = v.item()
        else:
            newv = v

        if hasattr(v, 'dtype'):
            newt = v.dtype.name
        else:
            if isinstance(v, Iterable) and v:
                # Use the type of the first one
                v = v[0]
            else:
                # This is likely an empty value
                # so just default to an empty string
                v = ''
            newt = type(v).__name__

        if verbose is True:
            return_copy[k] = {
                'type': newt,
                'data': newv
            }
        else:
            return_copy[k] = newv

    return return_copy
