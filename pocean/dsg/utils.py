#!python
# coding=utf-8
from __future__ import division
from datetime import datetime

import pandas as pd
from shapely.geometry import Polygon, LineString

from pocean.utils import (
    get_default_axes,
    unique_justseen,
    dict_update
)

from pocean import logger as L  # noqa


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
    Attribute names come from https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/
    The coordinate reference system (CRS) is assumed to be EPSG:4326, which is WGS84 and is used with
    GPS satellite navigation (http://spatialreference.org/ref/epsg/wgs-84/).  This is NCEI's default.
    Coordinate values are latitude (decimal degrees_north) and longitude (decimal degrees_east).
    Longitude values are limited to [-180, 180).

    :param df: data (Pandas DataFrame)
    :param axes: keys (x,y,z,t) are associated with actual column names (dictionary)
    :return: nested dictionary of variable and global attributes
    """
    axes = get_default_axes(axes)
    miny = round(float(df[axes.y].min()), 6)
    maxy = round(float(df[axes.y].max()), 6)
    minx = round(float(df[axes.x].min()), 6)
    maxx = round(float(df[axes.x].max()), 6)
    if minx == maxx and miny == maxy:
        geometry_wkt = 'POINT (' \
            '{maxx:.6f} {maxy:.6f})'.format(
                maxx=maxx,
                maxy=maxy,
            )
        geometry_bbox = geometry_wkt
    else:
        p = Polygon(zip(df[axes.x], df[axes.y]))
        dateline = LineString([(180, 90), (-180, -90)])
        # If we cross the dateline normalize the coordinates before polygon
        if dateline.crosses(p):
            newx = (df[axes.x] + 360) % 360
            p = Polygon(zip(newx, df[axes.y]))
        geometry_bbox = p.envelope.wkt
        geometry_wkt = p.convex_hull.wkt

    return {
        'variables': {
            axes.y: {
                'attributes': {
                    'actual_min': miny,
                    'actual_max': maxy,
                }
            },
            axes.x: {
                'attributes': {
                    'actual_min': minx,
                    'actual_max': maxx,
                }
            },
        },
        'attributes': {
            'geospatial_lat_min': miny,
            'geospatial_lat_max': maxy,
            'geospatial_lon_min': minx,
            'geospatial_lon_max': maxx,
            'geospatial_bbox': geometry_bbox,
            'geospatial_bounds': geometry_wkt,
            'geospatial_bounds_crs': 'EPSG:4326',
        }
    }


def get_vertical_attributes(df, axes=None):
    """ Use values in a dataframe to set vertical attributes for the eventual netCDF file
    Attribute names come from https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/
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
    Attribute names come from https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/

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
