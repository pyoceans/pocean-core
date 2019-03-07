#!python
# coding=utf-8
import unittest
from datetime import datetime, timedelta

import pytz
import pandas as pd
from dateutil.parser import parse as dtparse

from pocean.dsg import utils
from pocean import logger as L  # noqa


class TestDsgUtils(unittest.TestCase):

    geo = pd.DataFrame({
        'x': [-1, -2, -3, -4],
        'y': [1, 2, 3, 4]
    })

    z = pd.DataFrame({
        'z': [1, 2, 3, 4],
    })

    times = pd.DataFrame({
        't': pd.to_datetime([
            '2018-08-19 00:00:00',
            '2018-08-20 00:00:00',
            '2018-08-21 00:00:00',
            '2018-08-22 00:00:00',
            '2018-08-23 00:00:00',
            '2018-08-23 00:00:05',
        ])
    })

    avgtimes = pd.DataFrame({
        't': pd.to_datetime([
            '2018-08-19 00:00:00',
            '2018-08-20 23:00:55',
            '2018-08-21 00:00:35',
        ])
    })

    def test_get_vertical_meta(self):
        meta = utils.get_vertical_attributes(self.z)

        assert meta == {
            'variables': {
                'z': {
                    'attributes': {
                        'actual_min': 1,
                        'actual_max': 4,
                    }
                },
            },
            'attributes': {
                'geospatial_vertical_min': 1,
                'geospatial_vertical_max': 4,
                'geospatial_vertical_units': 'm',
            }
        }

    def test_get_geospatial_meta(self):
        meta = utils.get_geographic_attributes(self.geo)

        assert meta == {
            'variables': {
                'y': {
                    'attributes': {
                        'actual_min': 1,
                        'actual_max': 4,
                    }
                },
                'x': {
                    'attributes': {
                        'actual_min': -4,
                        'actual_max': -1,
                    }
                },
            },
            'attributes': {
                'geospatial_lat_max': 4,
                'geospatial_lat_min': 1,
                'geospatial_lon_max': -1,
                'geospatial_lon_min': -4,
                'geospatial_bounds': (
                    'POLYGON (('
                    '4.000000 -4.000000, '
                    '4.000000 -1.000000, '
                    '1.000000 -1.000000, '
                    '1.000000 -4.000000, '
                    '4.000000 -4.000000'
                    '))'
                ),
                'geospatial_bounds_crs': 'EPSG:4326',
            }
        }

    def test_get_temporal_meta_from_times_average(self):
        meta = utils.get_temporal_attributes(self.avgtimes)

        assert meta == {
            'variables': {
                't': {
                    'attributes': {
                        'actual_min': '2018-08-19T00:00:00Z',
                        'actual_max': '2018-08-21T00:00:35Z',
                    }
                }
            },
            'attributes': {
                'time_coverage_start': '2018-08-19T00:00:00Z',
                'time_coverage_end': '2018-08-21T00:00:35Z',
                'time_coverage_duration': 'P2DT0H0M35S',
                'time_coverage_resolution': 'P0DT16H0M12S',
            }
        }

    def test_get_temporal_meta_from_times(self):
        meta = utils.get_temporal_attributes(self.times)

        assert meta == {
            'variables': {
                't': {
                    'attributes': {
                        'actual_min': '2018-08-19T00:00:00Z',
                        'actual_max': '2018-08-23T00:00:05Z',
                    }
                }
            },
            'attributes': {
                'time_coverage_start': '2018-08-19T00:00:00Z',
                'time_coverage_end': '2018-08-23T00:00:05Z',
                'time_coverage_duration': 'P4DT0H0M5S',
                'time_coverage_resolution': 'P1DT0H0M0S',
            }
        }

    def test_get_creation(self):
        meta = utils.get_creation_attributes(history='DID THINGS')

        now = datetime.utcnow().replace(tzinfo=pytz.utc)

        assert (now - dtparse(meta['attributes']['date_created'])) < timedelta(minutes=1)
        assert (now - dtparse(meta['attributes']['date_issued'])) < timedelta(minutes=1)
        assert (now - dtparse(meta['attributes']['date_modified'])) < timedelta(minutes=1)
        assert 'DID THINGS' in meta['attributes']['history']
