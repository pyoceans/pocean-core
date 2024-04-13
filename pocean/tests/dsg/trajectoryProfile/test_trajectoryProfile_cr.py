import logging
import math
import os
import tempfile
import unittest

import numpy as np
from dateutil.parser import parse as dtparse
from shapely.wkt import loads as wktloads

from pocean import logger as L
from pocean.dsg import ContiguousRaggedTrajectoryProfile
from pocean.tests.dsg.test_new import test_is_mine

L.level = logging.INFO
L.handlers = [logging.StreamHandler()]


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

    def test_crtp_dataframe_single(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectoryProfile(self.single) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectoryProfile.from_dataframe(df, tmpnc, axes=axes) as result_ncd:
                assert 'profile' in result_ncd.dimensions
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectoryProfile, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crtp_dataframe_single_unique_dims(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectoryProfile(self.single) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectoryProfile.from_dataframe(df, tmpnc, axes=axes, unique_dims=True) as result_ncd:
                assert 'profile_dim' in result_ncd.dimensions
                assert 'trajectory_dim' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectoryProfile, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crtp_dataframe_multi(self):
        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectoryProfile(self.multi) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectoryProfile.from_dataframe(df, tmpnc, axes=axes) as result_ncd:
                assert 'profile' in result_ncd.dimensions
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectoryProfile, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crtp_dataframe_missing_time(self):
        axes = {
            't': 'precise_time',
            'x': 'precise_lon',
            'y': 'precise_lat',
            'z': 'depth',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectoryProfile(self.missing_time) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectoryProfile.from_dataframe(df, tmpnc, axes=axes) as result_ncd:
                assert 'profile' in result_ncd.dimensions
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectoryProfile, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crtp_calculated_metadata_single(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }

        with ContiguousRaggedTrajectoryProfile(self.single) as st:
            s = st.calculated_metadata(axes=axes)
            assert s.min_t.round('s') == dtparse('2014-11-25 18:57:30')
            assert s.max_t.round('s') == dtparse('2014-11-27 07:10:30')
            assert len(s.trajectories) == 1
            traj = s.trajectories["sp025-20141125T1730"]
            assert traj.min_z == 0
            assert np.isclose(traj.max_z, 504.37827)
            assert traj.min_t.round('s') == dtparse('2014-11-25 18:57:30')
            assert traj.max_t.round('s') == dtparse('2014-11-27 07:10:30')

            first_loc = traj.geometry.coords[0]
            assert np.isclose(first_loc[0], -119.79025)
            assert np.isclose(first_loc[1], 34.30818)
            assert len(traj.profiles) == 17

    def test_crtp_calculated_metadata_multi(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }

        with ContiguousRaggedTrajectoryProfile(self.multi) as mt:
            m = mt.calculated_metadata(axes=axes)
            assert m.min_t.round('s') == dtparse('1990-01-01 00:00:00')
            assert m.max_t.round('s') == dtparse('1990-01-03 02:00:00')
            assert len(m.trajectories) == 5
            # First trajectory
            traj0 = m.trajectories[0]
            assert traj0.min_z == 0
            assert traj0.max_z == 43
            assert traj0.min_t.round('s') == dtparse('1990-01-02 05:00:00')
            assert traj0.max_t.round('s') == dtparse('1990-01-03 01:00:00')
            first_loc = traj0.geometry.coords[0]
            assert first_loc[0] == -60
            assert first_loc[1] == 53
            assert len(traj0.profiles) == 4
            assert traj0.profiles[0].t.round('s') == dtparse('1990-01-03 01:00:00')
            assert traj0.profiles[0].x == -60
            assert traj0.profiles[0].y == 49

            # Last trajectory
            traj4 = m.trajectories[4]
            assert traj4.min_z == 0
            assert traj4.max_z == 38
            assert traj4.min_t.round('s') == dtparse('1990-01-02 14:00:00')
            assert traj4.max_t.round('s') == dtparse('1990-01-02 15:00:00')
            first_loc = traj4.geometry.coords[0]
            assert first_loc[0] == -67
            assert first_loc[1] == 47
            assert len(traj4.profiles) == 4
            assert traj4.profiles[19].t.round('s') == dtparse('1990-01-02 14:00:00')
            assert traj4.profiles[19].x == -44
            assert traj4.profiles[19].y == 47

    def test_crtp_calculated_metadata_missing_time(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }

        with ContiguousRaggedTrajectoryProfile(self.missing_time) as mmt:
            t = mmt.calculated_metadata(axes=axes)
            assert t.min_t == dtparse('2014-11-16 21:32:29.952500')
            assert t.max_t == dtparse('2014-11-17 07:59:08.398500')
            assert len(t.trajectories) == 1

            traj = t.trajectories["UW157-20141116T211809"]
            assert np.isclose(traj.min_z, 0.47928014)
            assert np.isclose(traj.max_z, 529.68005)
            assert traj.min_t == dtparse('2014-11-16 21:32:29.952500')
            assert traj.max_t == dtparse('2014-11-17 07:59:08.398500')

            first_loc = traj.geometry.coords[0]

            assert np.isclose(first_loc[0], -124.681526638573)
            assert np.isclose(first_loc[1],  43.5022166666667)
            assert len(traj.profiles) == 13

    def test_crtp_just_missing_time(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }

        with ContiguousRaggedTrajectoryProfile(self.missing_time) as mmt:
            t = mmt.calculated_metadata(axes=axes)
            assert t.min_t == dtparse('2014-11-16 21:32:29.952500')
            assert t.max_t == dtparse('2014-11-17 07:59:08.398500')
            assert len(t.trajectories) == 1

            traj = t.trajectories["UW157-20141116T211809"]
            assert np.isclose(traj.min_z, 0.47928014)
            assert np.isclose(traj.max_z, 529.68005)
            assert traj.min_t == dtparse('2014-11-16 21:32:29.952500')
            assert traj.max_t == dtparse('2014-11-17 07:59:08.398500')

            first_loc = traj.geometry.coords[0]
            assert np.isclose(first_loc[0], -124.681526638573)
            assert np.isclose(first_loc[1],  43.5022166666667)
            assert len(traj.profiles) == 13

    def test_crtp_just_missing_locations(self):
        axes = {
            't': 'time',
            'x': 'longitude',
            'y': 'latitude',
            'z': 'depth',
        }

        with ContiguousRaggedTrajectoryProfile(self.nan_locations) as ml:
            t = ml.calculated_metadata(axes=axes)
            assert len(t.trajectories) == 1

            traj = t.trajectories["clark-20150709T1803"]
            coords = list(wktloads(traj.geometry.wkt).coords)
            assert True not in [ math.isnan(x) for x, y in coords ]
            assert True not in [ math.isnan(y) for x, y in coords ]
