# -*- coding: utf-8 -*-
import os

import unittest
from dateutil.parser import parse as dtparse
import numpy as np

from pocean.dsg import OrthogonalMultidimensionalProfile

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalProfile(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'om-single.nc')
        self.multi = os.path.join(os.path.dirname(__file__), 'resources', 'om-multiple.nc')

    def test_omp_load(self):
        OrthogonalMultidimensionalProfile(self.single).close()
        OrthogonalMultidimensionalProfile(self.multi).close()

    def test_omp_dataframe(self):
        with OrthogonalMultidimensionalProfile(self.single) as s:
            s.to_dataframe()

        with OrthogonalMultidimensionalProfile(self.multi) as m:
            m.to_dataframe()

    def test_omp_calculated_metadata(self):
        with OrthogonalMultidimensionalProfile(self.single) as ncd:
            s = ncd.calculated_metadata()
            assert s.min_t == dtparse('2005-07-09 01:48:00')
            assert s.max_t == dtparse('2005-07-09 01:48:00')
            assert np.isclose(s.profiles[1].min_z, 0.)
            assert np.isclose(s.profiles[1].max_z, 96.06)
            assert s.profiles[1].t == dtparse('2005-07-09 01:48:00')
            assert np.isclose(s.profiles[1].x, -149.3582)
            assert np.isclose(s.profiles[1].y, 60.0248)

        with OrthogonalMultidimensionalProfile(self.multi) as ncd:
            m = ncd.calculated_metadata()
            assert m.min_t == dtparse('2005-09-10 07:08:00')
            assert m.max_t == dtparse('2005-09-14 17:27:00')
            assert len(m.profiles.keys()) == 35
            assert np.isclose(m.profiles[2].min_z, 0.)
            assert np.isclose(m.profiles[2].max_z, 499.69)
            assert m.profiles[2].t == dtparse('2005-09-10 07:08:00')
            assert np.isclose(m.profiles[2].x, -148.2182)
            assert np.isclose(m.profiles[2].y, 58.5395)

            assert np.isclose(m.profiles[37].min_z, 0.)
            assert np.isclose(m.profiles[37].max_z, 292.01001)
            assert m.profiles[37].t == dtparse('2005-09-14 17:27:00')
            assert np.isclose(m.profiles[37].x, -149.468)
            assert np.isclose(m.profiles[37].y, 60.01)

    def test_json_attributes(self):
        ds = os.path.join(os.path.dirname(__file__), 'resources', 'om-1dy11.nc')
        om = OrthogonalMultidimensionalProfile(ds)
        om.to_dataframe()
        om.json_attributes()
        om.close()
