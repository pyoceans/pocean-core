# -*- coding: utf-8 -*-
import os
import tempfile

import unittest
from dateutil.parser import parse as dtparse
import numpy as np

from pocean.dsg import IncompleteMultidimensionalTrajectory
from pocean.tests.dsg.test_new import test_is_mine

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestIncompleteMultidimensionalTrajectory(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'im-single.nc')
        self.multi = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')

    def test_imt_load(self):
        IncompleteMultidimensionalTrajectory(self.single).close()
        IncompleteMultidimensionalTrajectory(self.multi).close()

    def test_imt_dataframe(self):
        with IncompleteMultidimensionalTrajectory(self.single) as ncd:
            fid, single_tmp = tempfile.mkstemp(suffix='.nc')
            single_df = ncd.to_dataframe(clean_rows=False)
            with IncompleteMultidimensionalTrajectory.from_dataframe(single_df, single_tmp) as ncd:
                assert 'trajectory' in ncd.dimensions
            os.close(fid)
            os.remove(single_tmp)

        with IncompleteMultidimensionalTrajectory(self.multi) as ncd:
            fid, multip_tmp = tempfile.mkstemp(suffix='.nc')
            multip_df = ncd.to_dataframe(clean_rows=False)
            with IncompleteMultidimensionalTrajectory.from_dataframe(multip_df, multip_tmp) as ncd:
                assert 'trajectory' in ncd.dimensions
            os.close(fid)
            os.remove(multip_tmp)

    def test_imt_dataframe_reduce_dims(self):
        with IncompleteMultidimensionalTrajectory(self.single) as ncd:
            fid, single_tmp = tempfile.mkstemp(suffix='.nc')
            single_df = ncd.to_dataframe(clean_rows=False)
            with IncompleteMultidimensionalTrajectory.from_dataframe(single_df, single_tmp, reduce_dims=True) as ncd:
                # Reduced trajectory dimension
                assert 'trajectory' not in ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, single_tmp)  # Try to load it again
            os.close(fid)
            os.remove(single_tmp)

        with IncompleteMultidimensionalTrajectory(self.multi) as ncd:
            fid, multip_tmp = tempfile.mkstemp(suffix='.nc')
            multip_df = ncd.to_dataframe(clean_rows=False)
            with IncompleteMultidimensionalTrajectory.from_dataframe(multip_df, multip_tmp, reduce_dims=True) as ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory' in ncd.dimensions
            os.close(fid)
            os.remove(multip_tmp)

    def test_imt_calculated_metadata(self):
        with IncompleteMultidimensionalTrajectory(self.single) as ncd:
            s = ncd.calculated_metadata()
            assert s.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert s.max_t.round('S') == dtparse('1990-01-05 03:00:00')
            traj1 = s.trajectories["Trajectory1"]
            assert traj1.min_z == 0
            assert traj1.max_z == 99
            assert traj1.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj1.max_t.round('S') == dtparse('1990-01-05 03:00:00')
            assert np.isclose(traj1.first_loc.x, -7.9336)
            assert np.isclose(traj1.first_loc.y, 42.00339)

        with IncompleteMultidimensionalTrajectory(self.multi) as ncd:
            m = ncd.calculated_metadata()
            assert m.min_t == dtparse('1990-01-01 00:00:00')
            assert m.max_t == dtparse('1990-01-02 12:00:00')
            assert len(m.trajectories) == 4
            traj0 = m.trajectories["Trajectory0"]
            assert traj0.min_z == 0
            assert traj0.max_z == 35
            assert traj0.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj0.max_t.round('S') == dtparse('1990-01-02 11:00:00')
            assert np.isclose(traj0.first_loc.x, -35.07884)
            assert np.isclose(traj0.first_loc.y, 2.15286)

            traj3 = m.trajectories["Trajectory3"]
            assert traj3.min_z == 0
            assert traj3.max_z == 36
            assert traj3.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj3.max_t.round('S') == dtparse('1990-01-02 12:00:00')
            assert np.isclose(traj3.first_loc.x, -73.3026)
            assert np.isclose(traj3.first_loc.y, 1.95761)

    def test_json_attributes(self):
        with IncompleteMultidimensionalTrajectory(self.single) as s:
            s.json_attributes()

        with IncompleteMultidimensionalTrajectory(self.multi) as m:
            m.json_attributes()
