# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

from dateutil.parser import parse as dtparse
import numpy as np
from pocean.dsg import IncompleteMultidimensionalProfile
from pocean.tests.dsg.test_new import test_is_mine

import logging
from pocean import logger
logger.level = logging.DEBUG
logger.handlers = [logging.StreamHandler()]


class TestIncompleteMultidimensionalProfile(unittest.TestCase):

    def setUp(self):
        self.multi = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')

    def test_imp_load(self):
        IncompleteMultidimensionalProfile(self.multi).close()

    def test_imp_dataframe(self):
        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with IncompleteMultidimensionalProfile(self.multi) as ncd:
            df = ncd.to_dataframe()
            with IncompleteMultidimensionalProfile.from_dataframe(df, single_tmp) as result_ncd:
                assert 'profile' in result_ncd.dimensions
        test_is_mine(IncompleteMultidimensionalProfile, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    def test_imp_calculated_metadata(self):
        with IncompleteMultidimensionalProfile(self.multi) as ncd:
            m = ncd.calculated_metadata()
            assert m.min_t == dtparse('1990-01-01 00:00:00')
            assert m.max_t == dtparse('1990-01-06 21:00:00')
            assert len(m.profiles.keys()) == 137
            assert np.isclose(m.profiles[0].min_z, 0.05376, atol=1e-5)
            assert np.isclose(m.profiles[0].max_z, 9.62958, atol=1e-5)
            assert m.profiles[0].t == dtparse('1990-01-01 00:00:00')
            assert m.profiles[0].x == 119
            assert m.profiles[0].y == 171

            assert np.isclose(m.profiles[141].min_z, 0.04196, atol=1e-5)
            assert np.isclose(m.profiles[141].max_z, 9.85909, atol=1e-5)
            assert m.profiles[141].t == dtparse('1990-01-06 21:00:00')
            assert m.profiles[141].x == 34
            assert m.profiles[141].y == 80
