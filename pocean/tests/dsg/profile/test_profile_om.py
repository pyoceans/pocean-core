#!python
import logging
import os
import tempfile
import unittest

import numpy as np
from dateutil.parser import parse as dtparse

from pocean import logger
from pocean.cf import CFDataset
from pocean.dsg import OrthogonalMultidimensionalProfile
from pocean.tests.dsg.test_new import test_is_mine

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestOrthogonalMultidimensionalProfile(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'om-single.nc')
        self.multi = os.path.join(os.path.dirname(__file__), 'resources', 'om-multiple.nc')

    def test_omp_load(self):
        OrthogonalMultidimensionalProfile(self.single).close()
        OrthogonalMultidimensionalProfile(self.multi).close()

    def test_omp_dataframe_single(self):
        CFDataset.load(self.single)

        fid, single_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalProfile(self.single) as ncd:
            df = ncd.to_dataframe()
            with self.assertRaises(NotImplementedError):
                with OrthogonalMultidimensionalProfile.from_dataframe(df, single_tmp) as result_ncd:
                    assert 'profile' in result_ncd.dimensions
                test_is_mine(OrthogonalMultidimensionalProfile, single_tmp)  # Try to load it again
        os.close(fid)
        os.remove(single_tmp)

    def test_omp_dataframe_multi(self):
        CFDataset.load(self.multi)

        fid, multi_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalProfile(self.multi) as ncd:
            df = ncd.to_dataframe()
            with self.assertRaises(NotImplementedError):
                with OrthogonalMultidimensionalProfile.from_dataframe(df, multi_tmp) as result_ncd:
                    assert 'profile' in result_ncd.dimensions
                test_is_mine(OrthogonalMultidimensionalProfile, multi_tmp)  # Try to load it again
        os.close(fid)
        os.remove(multi_tmp)

    def test_omp_dataframe_multi_unique_dims(self):
        CFDataset.load(self.multi)

        fid, multi_tmp = tempfile.mkstemp(suffix='.nc')
        with OrthogonalMultidimensionalProfile(self.multi) as ncd:
            df = ncd.to_dataframe()
            with self.assertRaises(NotImplementedError):
                with OrthogonalMultidimensionalProfile.from_dataframe(df, multi_tmp, unique_dims=True) as result_ncd:
                    assert 'profile_dim' in result_ncd.dimensions
                test_is_mine(OrthogonalMultidimensionalProfile, multi_tmp)  # Try to load it again
        os.close(fid)
        os.remove(multi_tmp)

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
