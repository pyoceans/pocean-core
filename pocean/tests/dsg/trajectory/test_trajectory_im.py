#!python
import logging
import os
import tempfile
import unittest

import numpy as np
from dateutil.parser import parse as dtparse

from pocean import logger
from pocean.cf import CFDataset
from pocean.dsg import IncompleteMultidimensionalTrajectory
from pocean.tests.dsg.test_new import test_is_mine

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestIncompleteMultidimensionalTrajectory(unittest.TestCase):

    def test_im_single_row(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-singlerow.nc')

        with IncompleteMultidimensionalTrajectory(filepath) as s:
            df = s.to_dataframe(clean_rows=True)
            assert len(df) == 1

    def test_imt_multi(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')

        CFDataset.load(filepath).close()

        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile) as result_ncd:
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, unique_dims=True) as result_ncd:
                assert 'trajectory_dim' in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory' in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, unique_dims=True, reduce_dims=True, unlimited=True) as result_ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory_dim' in result_ncd.dimensions
                assert result_ncd.dimensions['obs_dim'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_imt_multi_not_string(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple-nonstring.nc')

        CFDataset.load(filepath).close()

        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile) as result_ncd:
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory' not in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Could not reduce dims since there was more than one trajectory
                assert 'trajectory' not in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_imt_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-single.nc')

        CFDataset.load(filepath).close()

        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile) as result_ncd:
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Reduced trajectory dimension
                assert 'trajectory' not in result_ncd.dimensions
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                # Reduced trajectory dimension
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                # Reduced trajectory dimension
                assert 'trajectory' not in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_imt_change_axis_names(self):
        new_axis = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'depth'
        }

        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')
        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False, axes=new_axis)

            with IncompleteMultidimensionalTrajectory.from_dataframe(df, tmpfile, axes=new_axis) as result_ncd:
                assert 'trajectory' in result_ncd.dimensions
                assert 'time' in result_ncd.variables
                assert 'lon' in result_ncd.variables
                assert 'lat' in result_ncd.variables
                assert 'depth' in result_ncd.variables
            test_is_mine(IncompleteMultidimensionalTrajectory, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)

    def test_imt_calculated_metadata_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-single.nc')

        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            s = ncd.calculated_metadata()
            assert s.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert s.max_t.round('S') == dtparse('1990-01-05 03:00:00')
            traj1 = s.trajectories["Trajectory1"]
            assert traj1.min_z == 0
            assert traj1.max_z == 99
            assert traj1.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj1.max_t.round('S') == dtparse('1990-01-05 03:00:00')
            first_loc = traj1.geometry.coords[0]
            assert np.isclose(first_loc[0], -7.9336)
            assert np.isclose(first_loc[1], 42.00339)

    def test_imt_calculated_metadata_multi(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')

        with IncompleteMultidimensionalTrajectory(filepath) as ncd:
            m = ncd.calculated_metadata()
            assert m.min_t == dtparse('1990-01-01 00:00:00')
            assert m.max_t == dtparse('1990-01-02 12:00:00')
            assert len(m.trajectories) == 4
            traj0 = m.trajectories["Trajectory0"]
            assert traj0.min_z == 0
            assert traj0.max_z == 35
            assert traj0.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj0.max_t.round('S') == dtparse('1990-01-02 11:00:00')
            first_loc = traj0.geometry.coords[0]
            assert np.isclose(first_loc[0], -35.07884)
            assert np.isclose(first_loc[1], 2.15286)

            traj3 = m.trajectories["Trajectory3"]
            assert traj3.min_z == 0
            assert traj3.max_z == 36
            assert traj3.min_t.round('S') == dtparse('1990-01-01 00:00:00')
            assert traj3.max_t.round('S') == dtparse('1990-01-02 12:00:00')
            first_loc = traj3.geometry.coords[0]
            assert np.isclose(first_loc[0], -73.3026)
            assert np.isclose(first_loc[1], 1.95761)

    def test_json_attributes_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-single.nc')

        with IncompleteMultidimensionalTrajectory(filepath) as s:
            s.json_attributes()

    def test_json_attributes_multi(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'im-multiple.nc')

        with IncompleteMultidimensionalTrajectory(filepath) as s:
            s.json_attributes()
