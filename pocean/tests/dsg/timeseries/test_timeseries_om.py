#!python
# coding=utf-8
import os
import tempfile
import unittest

import numpy as np

from pocean.dsg import OrthogonalMultidimensionalTimeseries
from pocean.tests.dsg.test_new import test_is_mine

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalTimeseries(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'tt.nc')
        self.ph = np.ma.array([
            8.1080176, 8.11740265, 8.11924184, 8.11615471, 8.11445695, 8.11600021,
            8.11903291, 8.1187229, 8.105218, 8.10998784, 8.10715445, 8.10530323,
            8.11167052, 8.11142766, 8.10897461, 8.08827717, 8.11343609, 8.11746859,
            8.12326458, 8.11770947, 8.09127117, 8.10770576, 8.10252467, 8.10252874
        ])

    def test_omp_load(self):
        OrthogonalMultidimensionalTimeseries(self.single).close()

    def test_timeseries_omt_dataframe(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            with OrthogonalMultidimensionalTimeseries.from_dataframe(df, single_tmp) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert np.ma.allclose(
                    result_ncd.variables['pH'][:].flatten(),
                    self.ph
                )
        test_is_mine(OrthogonalMultidimensionalTimeseries, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    def test_timeseries_omt_reduce_dims(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            with OrthogonalMultidimensionalTimeseries.from_dataframe(
                df,
                single_tmp,
                reduce_dims=True
            ) as result_ncd:
                assert 'station' not in result_ncd.dimensions
                assert np.ma.allclose(
                    result_ncd.variables['pH'][:].flatten(),
                    self.ph
                )
        test_is_mine(OrthogonalMultidimensionalTimeseries, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    def test_timeseries_omt_no_z(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            axes = {
                'z': None
            }
            df.drop(columns=['z'], inplace=True)
            with OrthogonalMultidimensionalTimeseries.from_dataframe(
                df,
                single_tmp,
                axes=axes,
            ) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert 'z' not in result_ncd.variables
                assert np.ma.allclose(
                    result_ncd.variables['pH'][:].flatten(),
                    self.ph
                )
        test_is_mine(OrthogonalMultidimensionalTimeseries, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    def test_timeseries_omt_no_z_no_station(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            axes = {
                'z': None
            }
            df.drop(columns=['z'], inplace=True)
            with OrthogonalMultidimensionalTimeseries.from_dataframe(
                df,
                single_tmp,
                axes=axes,
                reduce_dims=True
            ) as result_ncd:
                assert 'station' not in result_ncd.dimensions
                assert 'z' not in result_ncd.variables
                assert np.ma.allclose(
                    result_ncd.variables['pH'][:].flatten(),
                    self.ph
                )
        test_is_mine(OrthogonalMultidimensionalTimeseries, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)
