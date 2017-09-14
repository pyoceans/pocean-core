import os
import math

import unittest
from dateutil.parser import parse as dtparse
import numpy as np
from shapely.wkt import loads as wktloads

from pocean.dsg import ContiguousRaggedTrajectoryProfile

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestContinousRaggedTrajectoryProfile(unittest.TestCase):

    def setUp(self):
        self.single = os.path.join(os.path.dirname(__file__), 'resources', 'cr-single.nc')
        self.multi = os.path.join(os.path.dirname(__file__), 'resources', 'cr-multiple.nc')
        self.missing_time = os.path.join(os.path.dirname(__file__), 'resources', 'cr-missing-time.nc')
        self.nan_locations = os.path.join(os.path.dirname(__file__), 'resources', 'cr-nan-locations.nc')

    def test_crtp_load(self):
        ContiguousRaggedTrajectoryProfile(self.single).close()
        ContiguousRaggedTrajectoryProfile(self.multi).close()
        ContiguousRaggedTrajectoryProfile(self.missing_time).close()

    def test_crtp_dataframe(self):
        with ContiguousRaggedTrajectoryProfile(self.single) as s:
            s.to_dataframe()

        with ContiguousRaggedTrajectoryProfile(self.multi) as m:
            m.to_dataframe()

        with ContiguousRaggedTrajectoryProfile(self.missing_time) as t:
            t.to_dataframe()

    def test_crtp_calculated_metadata(self):
        with ContiguousRaggedTrajectoryProfile(self.single) as st:
            s = st.calculated_metadata()
            assert s.min_t.round('S') == dtparse('2014-11-25 18:57:30')
            assert s.max_t.round('S') == dtparse('2014-11-27 07:10:30')
            assert len(s.trajectories) == 1
            traj = s.trajectories["sp025-20141125T1730"]
            assert traj.min_z == 0
            assert np.isclose(traj.max_z, 504.37827)
            assert traj.min_t.round('S') == dtparse('2014-11-25 18:57:30')
            assert traj.max_t.round('S') == dtparse('2014-11-27 07:10:30')
            assert np.isclose(traj.first_loc.x, -119.79025)
            assert np.isclose(traj.first_loc.y, 34.30818)
            assert len(traj.profiles) == 17

        with ContiguousRaggedTrajectoryProfile(self.multi) as mt:
            m = mt.calculated_metadata()
            assert m.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert m.max_t.round('S') == dtparse('1990-01-03 02:00:00')
            assert len(m.trajectories) == 5
            # First trajectory
            traj0 = m.trajectories[0]
            assert traj0.min_z == 0
            assert traj0.max_z == 43
            assert traj0.min_t.round('S') == dtparse('1990-01-02 05:00:00')
            assert traj0.max_t.round('S') == dtparse('1990-01-03 01:00:00')
            assert traj0.first_loc.x == -60
            assert traj0.first_loc.y == 53
            assert len(traj0.profiles) == 4
            assert traj0.profiles[0].t.round('S') == dtparse('1990-01-03 01:00:00')
            assert traj0.profiles[0].x == -60
            assert traj0.profiles[0].y == 49
            # Last trajectory
            traj4 = m.trajectories[4]
            assert traj4.min_z == 0
            assert traj4.max_z == 38
            assert traj4.min_t.round('S') == dtparse('1990-01-02 14:00:00')
            assert traj4.max_t.round('S') == dtparse('1990-01-02 15:00:00')
            assert traj4.first_loc.x == -67
            assert traj4.first_loc.y == 47
            assert len(traj4.profiles) == 4
            assert traj4.profiles[19].t.round('S') == dtparse('1990-01-02 14:00:00')
            assert traj4.profiles[19].x == -44
            assert traj4.profiles[19].y == 47

        with ContiguousRaggedTrajectoryProfile(self.missing_time) as mmt:
            t = mmt.calculated_metadata()
            assert t.min_t == dtparse('2014-11-16 21:32:29.952482')
            assert t.max_t == dtparse('2014-11-17 07:59:08.398475')
            assert len(t.trajectories) == 1

            traj = t.trajectories["UW157-20141116T211809"]
            assert np.isclose(traj.min_z, 0.47928014)
            assert np.isclose(traj.max_z, 529.68005)
            assert traj.min_t == dtparse('2014-11-16 21:32:29.952482')
            assert traj.max_t == dtparse('2014-11-17 07:59:08.398475')
            assert np.isclose(traj.first_loc.x, -124.681526638573)
            assert np.isclose(traj.first_loc.y,  43.5022166666667)
            assert len(traj.profiles) == 13

    def test_just_missing_time(self):
        with ContiguousRaggedTrajectoryProfile(self.missing_time) as mmt:
            t = mmt.calculated_metadata()
            assert t.min_t == dtparse('2014-11-16 21:32:29.952482')
            assert t.max_t == dtparse('2014-11-17 07:59:08.398475')
            assert len(t.trajectories) == 1

            traj = t.trajectories["UW157-20141116T211809"]
            assert np.isclose(traj.min_z, 0.47928014)
            assert np.isclose(traj.max_z, 529.68005)
            assert traj.min_t == dtparse('2014-11-16 21:32:29.952482')
            assert traj.max_t == dtparse('2014-11-17 07:59:08.398475')
            assert np.isclose(traj.first_loc.x, -124.681526638573)
            assert np.isclose(traj.first_loc.y,  43.5022166666667)
            assert len(traj.profiles) == 13

    def test_just_missing_locations(self):
        with ContiguousRaggedTrajectoryProfile(self.nan_locations) as ml:
            t = ml.calculated_metadata()
            assert len(t.trajectories) == 1

            traj = t.trajectories["clark-20150709T1803"]
            coords = list(wktloads(traj.geometry.wkt).coords)
            assert True not in [ math.isnan(x) for x, y in coords ]
            assert True not in [ math.isnan(y) for x, y in coords ]
