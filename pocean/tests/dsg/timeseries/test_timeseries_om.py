# -*- coding: utf-8 -*-
import os
import tempfile

import unittest
from dateutil.parser import parse as dtparse
import numpy as np

from pocean.dsg import OrthogonalMultidimensionalTimeseries

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalTimeseries(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'tt.nc')

    def test_omp_load(self):
        OrthogonalMultidimensionalTimeseries(self.single).close()

    def test_omp_dataframe(self):
        single_tmp = tempfile.mkstemp(suffix='.nc')[-1]
        with OrthogonalMultidimensionalTimeseries(self.single) as s:
            df = s.to_dataframe()
            nc = OrthogonalMultidimensionalTimeseries.from_dataframe(df, single_tmp)
            nc.close()
            logger.info(single_tmp)
        #os.remove(single_tmp)

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

    # def test_json_attributes(self):
    #     ds = os.path.join(os.path.dirname(__file__), 'resources', 'tt.nc')
    #     om = OrthogonalMultidimensionalTimeseries(ds)
    #     om.to_dataframe()
    #     om.json_attributes()
    #     om.close()
