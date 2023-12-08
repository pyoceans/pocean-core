#!python
import os
import unittest
from datetime import datetime, timedelta

import pandas as pd
import pytz
from dateutil.parser import parse as dtparse

from pocean import logger as L  # noqa
from pocean.cf import CFDataset
from pocean.dsg import utils


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
                'geospatial_lat_min': 1.0,
                'geospatial_lat_max': 4.0,
                'geospatial_lon_min': -4.0,
                'geospatial_lon_max': -1.0,
                'geospatial_bbox': 'POLYGON ((-1 1, -1 4, -4 4, -4 1, -1 1))',
                'geospatial_bounds': 'LINESTRING (-1 1, -4 4)',
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

    def test_wrap_dateline(self):
        ncfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/wrapping_dateline.nc")

        with CFDataset.load(ncfile) as ncd:
            axes = {
                't': 'time',
                'z': 'z',
                'x': 'lon',
                'y': 'lat',
            }
            df = ncd.to_dataframe(axes=axes)

            meta = utils.get_geographic_attributes(df, axes=axes)

            assert meta == {
                "variables": {
                    "lat": {
                        "attributes": {
                            "actual_min": 61.777,
                            "actual_max": 67.068
                        }
                    },
                    "lon": {
                        "attributes": {
                            "actual_min": -179.966,
                            "actual_max": 179.858
                        }
                    }
                },
                "attributes": {
                    "geospatial_lat_min": 61.777,
                    "geospatial_lat_max": 67.068,
                    "geospatial_lon_min": -179.966,
                    "geospatial_lon_max": 179.858,
                    "geospatial_bbox": "POLYGON ((198.669 61.777, 198.669 67.068, 174.79200000000003 67.068, 174.79200000000003 61.777, 198.669 61.777))",
                    'geospatial_bounds': "POLYGON ((174.79200000000003 61.777, 174.92599999999993 62.206, 178.812 64.098, 192.86 67.029, 196.86 67.068, 197.094 67.044, 198.669 66.861, 187.784 64.188, 179.10799999999995 62.266, 176.16899999999998 61.862, 174.79200000000003 61.777))",
                    "geospatial_bounds_crs": "EPSG:4326"
                }
            }

    def test_wrap_small_coords(self):

        geo = pd.DataFrame({
            'x': [-1, -2],
            'y': [1, 2]
        })

        meta = utils.get_geographic_attributes(geo)

        assert meta == {
            'variables': {
                'y': {
                    'attributes': {
                        'actual_min': 1,
                        'actual_max': 2,
                    }
                },
                'x': {
                    'attributes': {
                        'actual_min': -2,
                        'actual_max': -1,
                    }
                },
            },
            'attributes': {
                'geospatial_lat_min': 1,
                'geospatial_lat_max': 2,
                'geospatial_lon_min': -2,
                'geospatial_lon_max': -1,
                'geospatial_bbox': 'POLYGON ((-1 1, -1 2, -2 2, -2 1, -1 1))',
                'geospatial_bounds': 'LINESTRING (-1 1, -2 2)',
                'geospatial_bounds_crs': 'EPSG:4326',
            }
        }

    def test_wrap_same_coords(self):

        geo = pd.DataFrame({
            'x': [-1, -1, -1],
            'y': [1, 1, 1]
        })

        meta = utils.get_geographic_attributes(geo)

        assert meta == {
            'variables': {
                'y': {
                    'attributes': {
                        'actual_min': 1,
                        'actual_max': 1,
                    }
                },
                'x': {
                    'attributes': {
                        'actual_min': -1,
                        'actual_max': -1,
                    }
                },
            },
            'attributes': {
                'geospatial_lat_min': 1,
                'geospatial_lat_max': 1,
                'geospatial_lon_min': -1,
                'geospatial_lon_max': -1,
                'geospatial_bbox': 'POLYGON ((-1 1, -1 1, -1 1, -1 1))',
                'geospatial_bounds': 'POINT (-1 1)',
                'geospatial_bounds_crs': 'EPSG:4326',
            }
        }
