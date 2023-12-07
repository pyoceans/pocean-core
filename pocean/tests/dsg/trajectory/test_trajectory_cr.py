#!python
import logging
import os
import tempfile
import unittest
from os.path import dirname as dn
from os.path import join as jn

import pytest

from pocean import logger
from pocean.dsg import ContiguousRaggedTrajectory, get_calculated_attributes
from pocean.tests.dsg.test_new import test_is_mine

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


@pytest.mark.parametrize("fp", [
    #jn(dn(__file__), 'resources', 'cr-single.nc'),
    jn(dn(__file__), 'resources', 'cr-multiple.nc'),
    jn(dn(__file__), 'resources', 'cr-oot-A.nc'),
    jn(dn(__file__), 'resources', 'cr-oot-B.nc'),
])
def test_crt_load(fp):
    test_is_mine(ContiguousRaggedTrajectory, fp)


class TestContiguousRaggedTrajectory(unittest.TestCase):

    def setUp(self):
        self.multi = jn(dn(__file__), 'resources', 'cr-multiple.nc')
        self.oot_A = jn(dn(__file__), 'resources', 'cr-oot-A.nc')
        self.oot_B = jn(dn(__file__), 'resources', 'cr-oot-B.nc')

    def test_crt_dataframe_multiple(self):
        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectory(self.multi) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectory.from_dataframe(df, tmpnc, axes=axes) as result_ncd:
                assert 'trajectory' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectory, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crt_dataframe_multiple_unique_dims(self):
        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectory(self.multi) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectory.from_dataframe(df, tmpnc, axes=axes, unique_dims=True) as result_ncd:
                assert 'trajectory_dim' in result_ncd.dimensions
            test_is_mine(ContiguousRaggedTrajectory, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crt_dataframe_unlimited_dim(self):
        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectory(self.multi) as ncd:
            df = ncd.to_dataframe(axes=axes)
            with ContiguousRaggedTrajectory.from_dataframe(df, tmpnc, axes=axes, unlimited=True, unique_dims=True) as result_ncd:
                assert 'trajectory_dim' in result_ncd.dimensions
                assert 'obs_dim' in result_ncd.dimensions
                assert result_ncd.dimensions['obs_dim'].isunlimited() is True
            test_is_mine(ContiguousRaggedTrajectory, tmpnc)  # Try to load it again
        os.close(fid)
        os.remove(tmpnc)

    def test_crt_dataframe_oot_A(self):
        axes = {
            't':      'time',
            'x':      'lon',
            'y':      'lat',
            'z':      'depth',
            'sample': 'sample'
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectory(self.oot_A) as ncd:
            df = ncd.to_dataframe(axes=axes)
            df = df.sort_values(['trajectory', 'time'])
            attrs = get_calculated_attributes(df, axes=axes)

            with ContiguousRaggedTrajectory.from_dataframe(df, tmpnc, axes=axes, mode='a') as result_ncd:
                assert 'sample' in result_ncd.dimensions
                assert result_ncd.dimensions['sample'].size == 6610
                assert 'trajectory' in result_ncd.dimensions
                # This is removing null trajectories that have no data. Not much to do about this
                # because there is no way to store this empty trajectory in a dataframe.
                assert result_ncd.dimensions['trajectory'].size == 507
                result_ncd.apply_meta(attrs)

            test_is_mine(ContiguousRaggedTrajectory, tmpnc)  # Try to load it again

        os.close(fid)
        os.remove(tmpnc)

    def test_crt_dataframe_oot_B(self):
        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'depth',
        }
        fid, tmpnc = tempfile.mkstemp(suffix='.nc')
        with ContiguousRaggedTrajectory(self.oot_B) as ncd:
            df = ncd.to_dataframe(axes=axes)
            df = df.sort_values(['trajectory', 'time'])
            attrs = get_calculated_attributes(df, axes=axes)

            with ContiguousRaggedTrajectory.from_dataframe(df, tmpnc, axes=axes, mode='a') as result_ncd:
                assert 'obs' in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].size == 64116
                assert 'trajectory' in result_ncd.dimensions
                # This is removing null trajectories that have no data. Not much to do about this
                # because there is no way to store this empty trajectory in a dataframe.
                assert result_ncd.dimensions['trajectory'].size == 1000
                result_ncd.apply_meta(attrs)

            test_is_mine(ContiguousRaggedTrajectory, tmpnc)  # Try to load it again

        os.close(fid)
        os.remove(tmpnc)
