#!python
# coding=utf-8
import os
import tempfile
import unittest

from pocean.dsg import OrthogonalMultidimensionalTimeseries
from pocean.tests.dsg.test_new import test_is_mine

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalTimeseries(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'tt.nc')

    def test_omp_load(self):
        OrthogonalMultidimensionalTimeseries(self.single).close()

    def test_timeseries_omp_dataframe(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            with OrthogonalMultidimensionalTimeseries.from_dataframe(df, single_tmp) as result_ncd:
                assert 'station' in result_ncd.dimensions
        test_is_mine(OrthogonalMultidimensionalTimeseries, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    # def test_omp_calculated_metadata(self):
    #     with OrthogonalMultidimensionalTimeseries(self.single) as ncd:
    #         s = ncd.calculated_metadata()
    #         assert s.min_t == dtparse('2005-07-09 01:48:00')
    #         assert s.max_t == dtparse('2005-07-09 01:48:00')
    #         assert np.isclose(s.profiles[1].min_z, 0.)
    #         assert np.isclose(s.profiles[1].max_z, 96.06)
    #         assert s.profiles[1].t == dtparse('2005-07-09 01:48:00')
    #         assert np.isclose(s.profiles[1].x, -149.3582)
    #         assert np.isclose(s.profiles[1].y, 60.0248)
