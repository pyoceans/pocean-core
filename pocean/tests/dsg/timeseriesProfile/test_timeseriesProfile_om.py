#!python
# coding=utf-8
import os
import tempfile
import unittest

from pocean.dsg import OrthogonalMultidimensionalTimeseriesProfile

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalTimeseriesProfile(unittest.TestCase):

    def test_omtp_multi(self):
        multi = os.path.join(os.path.dirname(__file__), 'resources', 'om-multiple.nc')
        with OrthogonalMultidimensionalTimeseriesProfile(multi) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as ncd:
                assert 'station' in ncd.dimensions
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' in ncd.dimensions
            os.close(fid)
            os.remove(tmpfile)

    def test_omtp_single(self):
        single = os.path.join(os.path.dirname(__file__), 'resources', 'om-single.nc')
        with OrthogonalMultidimensionalTimeseriesProfile(single) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as ncd:
                assert 'station' in ncd.dimensions
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in ncd.dimensions
            os.close(fid)
            os.remove(tmpfile)

    def test_omtp_single_but_multi_file(self):
        single_multi = os.path.join(os.path.dirname(__file__), 'resources', 'om-multi-format-but-one-station.nc')
        with OrthogonalMultidimensionalTimeseriesProfile(single_multi) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile) as ncd:
                assert 'station' in ncd.dimensions
            with OrthogonalMultidimensionalTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as ncd:
                # Should remove the station dim since there is only one station
                assert 'station' not in ncd.dimensions
            os.close(fid)
            os.remove(tmpfile)
