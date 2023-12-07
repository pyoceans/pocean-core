#!python

from datetime import datetime

import pandas as pd
from shapely.geometry import (
    box,
    LineString,
    Point,
    Polygon,
)
from shapely.validation import make_valid

from pocean import logger as L  # noqa
from pocean.utils import dict_update, get_default_axes, unique_justseen


def get_calculated_attributes(df, axes=None, history=None):
    """ Functions to automate netCDF attribute generation from the data itself
    This is a wrapper for the other four functions, which could be called separately.

    :param df: data (Pandas DataFrame)
    :param axes: keys (x,y,z,t) are associated with actual column names (dictionary)
    :param history: history: text initializing audit trail for modifications to the original data (optional, string)
    :return: dictionary of global attributes
    """

    axes = get_default_axes(axes)
    attrs = get_geographic_attributes(df, axes)
    attrs = dict_update(attrs, get_vertical_attributes(df, axes))
    attrs = dict_update(attrs, get_temporal_attributes(df, axes))
    attrs = dict_update(attrs, get_creation_attributes(history))

    return attrs


def get_geographic_attributes(df, axes=None):
    """ Use values in a dataframe to set geographic attributes for the eventual netCDF file
    Attribute names come from https://www.ncei.noaa.gov/data/oceans/ncei/formats/netcdf/v2.0/index.html
    The coordinate reference system (CRS) is assumed to be EPSG:4326, which is WGS84 and is used with
    GPS satellite navigation (http://spatialreference.org/ref/epsg/wgs-84/).  This is NCEI's default.
    Coordinate values are latitude (decimal degrees_north) and longitude (decimal degrees_east).
    Longitude values are limited to [-180, 180).

    :param df: data (Pandas DataFrame)
    :param axes: keys (x,y,z,t) are associated with actual column names (dictionary)
    :return: nested dictionary of variable and global attributes
    """
    axes = get_default_axes(axes)

    carry_miny = round(float(df[axes.y].min()), 6)
    carry_maxy = round(float(df[axes.y].max()), 6)
    carry_minx = round(float(df[axes.x].min()), 6)
    carry_maxx = round(float(df[axes.x].max()), 6)

    notnull = df[axes.x].notnull() & df[axes.y].notnull()
    coords = list(zip(df.loc[notnull, axes.x], df.loc[notnull, axes.y]))

    if len(set(coords)) == 1:
        geoclass = Point
        # The set is to workaround the fact tht pocean
        # relied in a shapely<2 bug to pass a vector here instead of a point.
        coords = set(coords)
    elif len(coords) > 2:
        geoclass = Polygon
    else:
        geoclass = LineString

    p = geoclass(coords)
    dateline = LineString([(180, 90), (-180, -90)])
    # If we cross the dateline normalize the coordinates before polygon
    if dateline.crosses(p):
        newx = (df.loc[notnull, axes.x] + 360) % 360
        p = geoclass(zip(newx, df.loc[notnull, axes.y]))
        p = make_valid(p)

    geometry_bbox = box(*p.bounds).wkt
    geometry_wkt = p.convex_hull.wkt

    return {
        'variables': {
            axes.y: {
                'attributes': {
                    'actual_min': carry_miny,
                    'actual_max': carry_maxy,
                }
            },
            axes.x: {
                'attributes': {
                    'actual_min': carry_minx,
                    'actual_max': carry_maxx,
                }
            },
        },
        'attributes': {
            'geospatial_lat_min': carry_miny,
            'geospatial_lat_max': carry_maxy,
            'geospatial_lon_min': carry_minx,
            'geospatial_lon_max': carry_maxx,
            'geospatial_bbox': geometry_bbox,
            'geospatial_bounds': geometry_wkt,
            'geospatial_bounds_crs': 'EPSG:4326',
        }
    }


def get_vertical_attributes(df, axes=None):
    """ Use values in a dataframe to set vertical attributes for the eventual netCDF file
    Attribute names come from https://www.ncei.noaa.gov/data/oceans/ncei/formats/netcdf/v2.0/index.html
    The CRS, geospatial_bounds_vertical_crs, cannot be assumed because NCEI suggests any of
    * 'EPSG:5829' (instantaneous height above sea level),
    * 'EPSG:5831' (instantaneous depth below sea level), or
    * 'EPSG:5703' (NAVD88 height).
    Likewise, geospatial_vertical_positive cannot be assumed to be either 'up' or 'down'.
    Set these attributes separately according to the dataset.
    Note: values are cast from numpy.int to float

    :param df: data (Pandas DataFrame)
    :param axes: keys (x,y,z,t) are associated with actual column names (dictionary). z in meters.
    :return: nested dictionary of variable and global attributes
    """
    axes = get_default_axes(axes)
    minz = round(float(df[axes.z].min()), 6)
    maxz = round(float(df[axes.z].max()), 6)

    return {
        'variables': {
            axes.z: {
                'attributes': {
                    'actual_min': minz,
                    'actual_max': maxz,
                }
            },
        },
        'attributes': {
            'geospatial_vertical_min': minz,
            'geospatial_vertical_max': maxz,
            'geospatial_vertical_units': 'm',
        }
    }


def get_temporal_attributes(df, axes=None):
    """ Use values in a dataframe to set temporal attributes for the eventual netCDF file
    Attribute names come from https://www.ncei.noaa.gov/data/oceans/ncei/formats/netcdf/v2.0/index.html

    :param df: data (Pandas DataFrame)
    :param axes: keys (x,y,z,t) are associated with actual column names (dictionary). z in meters.
    :return: nested dictionary of variable and global attributes
    """

    axes = get_default_axes(axes)
    mint = df[axes.t].min()
    maxt = df[axes.t].max()

    times = pd.DatetimeIndex(unique_justseen(df[axes.t]))
    dt_index_diff = times[1:] - times[:-1]
    dt_counts = dt_index_diff.value_counts(sort=True)

    if dt_counts.size > 0 and dt_counts.values[0] / (len(times) - 1) > 0.75:
        mode_value = dt_counts.index[0]
    else:
        # Calculate a static resolution
        mode_value = ((maxt - mint) / len(times))

    return {
        'variables': {
            axes.t: {
                'attributes': {
                    'actual_min': mint.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'actual_max': maxt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
            },
        },
        'attributes': {
            'time_coverage_start': mint.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time_coverage_end': maxt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time_coverage_duration': (maxt - mint).round('1S').isoformat(),
            'time_coverage_resolution': mode_value.round('1S').isoformat()
        }
    }


def get_creation_attributes(history=None):
    """ Query system for netCDF file creation times

    :param history: text initializing audit trail for modifications to the original data (optional, string)
    :return: dictionary of global attributes
    """
    nc_create_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    attrs = {
        'attributes': {
            'date_created': nc_create_ts,
            'date_issued': nc_create_ts,
            'date_modified': nc_create_ts,
        }
    }

    # Add in the passed in history
    if history is not None:
        attrs['attributes']['history'] = '{} - {}'.format(
            nc_create_ts,
            history
        )

    return attrs
