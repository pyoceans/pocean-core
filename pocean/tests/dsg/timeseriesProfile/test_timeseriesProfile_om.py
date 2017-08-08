#!python
# coding=utf-8
import os
import tempfile
import unittest

from pocean.dsg import OrthogonalMultidimensionalTimeseriesProfile
from pocean.tests.dsg.test_new import test_is_mine

import logging
from pocean import logger
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

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' in result_ncd.dimensions
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['time'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' in result_ncd.dimensions
                assert result_ncd.dimensions['time'].isunlimited() is True
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
                assert result_ncd.dimensions['time'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
                assert result_ncd.dimensions['time'].isunlimited() is True
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
                assert result_ncd.dimensions['time'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in result_ncd.dimensions
                assert result_ncd.dimensions['time'].isunlimited() is True
            test_is_mine(OrthogonalMultidimensionalTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)
