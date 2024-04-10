#!python
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pocean import logger
from pocean.dsg import OrthogonalMultidimensionalTimeseriesProfile
from pocean.tests.dsg.test_new import test_is_mine

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalTimeseriesProfile(unittest.TestCase):

    def test_omtp_multi(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'om-multiple.nc')

        with OrthogonalMultidimensionalTimeseriesProfile(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as result_ncd:
                assert 'station' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unique_dims=True) as result_ncd:
                assert 'station_dim' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unique_dims=True, reduce_dims=True, unlimited=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station_dim' in result_ncd.dimensions
                assert result_ncd.dimensions['t_dim'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_omtp_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'om-single.nc')

        with OrthogonalMultidimensionalTimeseriesProfile(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as result_ncd:
                assert 'station' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_omtp_single_but_multi_file(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'om-multi-format-but-one-station.nc')

        with OrthogonalMultidimensionalTimeseriesProfile(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as result_ncd:
                assert 'station' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
                assert result_ncd.dimensions['t'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_omtp_change_axis_names(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'om-multiple.nc')

        new_axis = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'depth'
        }

        with OrthogonalMultidimensionalTimeseriesProfile(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False, axes=new_axis)

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, axes=new_axis) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert 'time' in result_ncd.variables
                assert 'lon' in result_ncd.variables
                assert 'lat' in result_ncd.variables
                assert 'depth' in result_ncd.variables
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_detach_z_with_station(self):

        df = pd.DataFrame({
            'station': 'awesome',
            'y': 100,
            'x': 100,
            't': pd.to_datetime(np.arange(10).repeat(10), unit='D'),
            'z': np.tile(np.arange(10), 10),
            'full': np.random.rand(10, 10).flatten() * 100,
            'top': np.tile(np.ma.fix_invalid( [1] + [np.nan] * 9), 10),
            'middle': np.tile(np.ma.fix_invalid( [np.nan] * 4 + [2] + [np.nan] * 5), 10),
            'bottom': np.tile(np.ma.fix_invalid( [np.nan] * 9 + [3]), 10),
        })

        # top, middle, bottom has a value of "1" in a different location and should all be
        # detached from the profile and end up with that single "1" value in the variable.

        fid, tmpfile = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(
            df,
            output=tmpfile,
            detach_z=['top', 'bottom', 'middle'],
            reduce_dims=False,
        ) as ncd:
            t = ncd.variables['top'][:]
            m = ncd.variables['middle'][:]
            b = ncd.variables['bottom'][:]

            assert t.size == 10
            assert m.size == 10
            assert b.size == 10
            assert np.all(t == 1)
            assert np.all(m == 2)
            assert np.all(b == 3)

        os.close(fid)
        os.remove(tmpfile)

    def test_detach_z_no_station(self):

        df = pd.DataFrame({
            'station': 'awesome',
            'y': 100,
            'x': 100,
            't': pd.to_datetime(np.arange(10).repeat(10), unit='D'),
            'z': np.tile(np.arange(10), 10),
            'full': np.random.rand(10, 10).flatten() * 100,
            'top': np.tile(np.ma.fix_invalid( [1] + [np.nan] * 9), 10),
            'middle': np.tile(np.ma.fix_invalid( [np.nan] * 4 + [2] + [np.nan] * 5), 10),
            'bottom': np.tile(np.ma.fix_invalid( [np.nan] * 9 + [3]), 10),
        })

        fid, tmpfile = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(
            df,
            output=tmpfile,
            detach_z=['top', 'bottom', 'middle'],
            reduce_dims=True,
        ) as ncd:
            t = ncd.variables['top'][:]
            m = ncd.variables['middle'][:]
            b = ncd.variables['bottom'][:]

            assert t.size == 10
            assert m.size == 10
            assert b.size == 10
            assert np.all(t == 1)
            assert np.all(m == 2)
            assert np.all(b == 3)
        os.close(fid)

        # Test the cycle
        with OrthogonalMultidimensionalTimeseriesProfile(tmpfile) as ncd:
            fid2, tmpfile2 = tempfile.mkstemp(suffix='.nc')
            df2 = ncd.to_dataframe()
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(
                df2,
                output=tmpfile2,
                detach_z=['top', 'bottom', 'middle'],
                reduce_dims=True,
            ) as ncd2:
                t2 = ncd2.variables['top'][:]
                m2 = ncd2.variables['middle'][:]
                b2 = ncd2.variables['bottom'][:]

                assert t2.size == 10
                assert m2.size == 10
                assert b2.size == 10
                assert np.all(t2 == 1)
                assert np.all(m2 == 2)
                assert np.all(b2 == 3)

            os.close(fid2)
            os.remove(tmpfile2)

        os.remove(tmpfile)
