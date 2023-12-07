#!python
import logging
import os
import tempfile
import unittest
from datetime import datetime

import netCDF4 as nc4
import pandas as pd
from numpy.testing import assert_array_equal as npeq

from pocean import logger
from pocean.cf import CFDataset
from pocean.dsg import RaggedTimeseriesProfile
from pocean.tests.dsg.test_new import test_is_mine

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestRaggedTimeseriesProfile(unittest.TestCase):

    def test_csv_to_nc_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'r-single.csv')

        df = pd.read_csv(filepath)
        fid, tmpfile = tempfile.mkstemp(suffix='.nc')

        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z'
        }

        df.time = pd.to_datetime(df.time)

        CFDataset.default_time_unit = 'hours since 2003-01-01 00:00:00Z'

        with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, axes=axes) as result_ncd:
            assert 'station' in result_ncd.dimensions
            assert result_ncd.dimensions['station'].size == 1
            assert 'profile' in result_ncd.dimensions
            assert result_ncd.dimensions['profile'].size == 1

            check_vars = ['z', 't090C', 'SP', 'SA', 'SR', 'CT', 'sigma0_CT']
            for v in check_vars:
                npeq(
                    result_ncd.variables[v][:],
                    df[v].values
                )

            assert result_ncd.variables['station'][0] == df.station.iloc[0] == 'CH2'
            assert result_ncd.variables['profile'][0] == df.profile.iloc[0] == '030617B'
            assert result_ncd.variables['lat'].size == 1
            assert result_ncd.variables['lat'].ndim == 1  # Not reduced
            assert result_ncd.variables['lat'][0] == df.lat.iloc[0] == 33.558
            assert result_ncd.variables['lon'].size == 1
            assert result_ncd.variables['lon'].ndim == 1  # Not reduced
            assert result_ncd.variables['lon'][0] == df.lon.iloc[0] == -118.405

            assert result_ncd.variables['time'].units == 'hours since 2003-01-01 00:00:00Z'
            assert result_ncd.variables['time'][0] == nc4.date2num(
                datetime(2003, 6, 17, 10, 32, 0),
                units=result_ncd.variables['time'].units
            )

            assert RaggedTimeseriesProfile.is_mine(result_ncd, strict=True)

        os.close(fid)
        os.remove(tmpfile)

    def test_csv_to_nc_multi(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'r-multi.csv')

        df = pd.read_csv(filepath)
        fid, tmpfile = tempfile.mkstemp(suffix='.nc')

        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z'
        }

        df.time = pd.to_datetime(df.time)

        with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, axes=axes) as result_ncd:
            assert 'station' in result_ncd.dimensions
            assert result_ncd.dimensions['station'].size == 2
            assert 'profile' in result_ncd.dimensions
            assert result_ncd.dimensions['profile'].size == 5

            check_vars = ['z', 'salinity', 'sigma0']
            for v in check_vars:
                npeq(
                    result_ncd.variables[v][:],
                    df[v].values
                )

            npeq(
                result_ncd.variables['station'][:],
                ['CN1', 'CN2']
            )
            npeq(
                result_ncd.variables['profile'][:],
                [
                    '030312B',
                    '030617B',
                    '030702B',
                    '030814B',
                    '031216C'
                ]
            )
            assert result_ncd.variables['profile'][0] == df.profile.iloc[0] == '030312B'
            assert result_ncd.variables['lat'].size == 2
            assert result_ncd.variables['lat'].ndim == 1  # Not reduced
            assert result_ncd.variables['lat'][0] == df.lat.iloc[0] == 33.5
            assert result_ncd.variables['lon'].size == 2
            assert result_ncd.variables['lon'].ndim == 1  # Not reduced
            assert result_ncd.variables['lon'][0] == df.lon.iloc[0] == -118.4

            npeq(
                result_ncd.variables['stationIndex'][:],
                [0, 0, 1, 0, 1]
            )

            npeq(
                result_ncd.variables['rowSize'][:],
                [844, 892, 893, 893, 891]
            )

            assert result_ncd.variables['time'][0] == nc4.date2num(
                datetime(2013, 3, 12, 10, 19, 6),
                units=result_ncd.variables['time'].units
            )
            assert RaggedTimeseriesProfile.is_mine(result_ncd, strict=True)

        os.close(fid)
        os.remove(tmpfile)

    def test_csv_to_nc_single_timezones(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'r-single.csv')

        df = pd.read_csv(filepath)
        fid, tmpfile = tempfile.mkstemp(suffix='.nc')

        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z'
        }

        df.time = pd.to_datetime(df.time)
        df.time = df.time.dt.tz_localize('UTC')

        with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, axes=axes) as result_ncd:
            assert 'station' in result_ncd.dimensions
            assert result_ncd.dimensions['station'].size == 1
            assert 'profile' in result_ncd.dimensions
            assert result_ncd.dimensions['profile'].size == 1

            check_vars = ['z', 't090C', 'SP', 'SA', 'SR', 'CT', 'sigma0_CT']
            for v in check_vars:
                npeq(
                    result_ncd.variables[v][:],
                    df[v].values
                )

            assert result_ncd.variables['station'][0] == df.station.iloc[0] == 'CH2'
            assert result_ncd.variables['profile'][0] == df.profile.iloc[0] == '030617B'
            assert result_ncd.variables['lat'].size == 1
            assert result_ncd.variables['lat'].ndim == 1  # Not reduced
            assert result_ncd.variables['lat'][0] == df.lat.iloc[0] == 33.558
            assert result_ncd.variables['lon'].size == 1
            assert result_ncd.variables['lon'].ndim == 1  # Not reduced
            assert result_ncd.variables['lon'][0] == df.lon.iloc[0] == -118.405

            assert result_ncd.variables['time'][0] == nc4.date2num(
                datetime(2003, 6, 17, 10, 32, 0),
                units=result_ncd.variables['time'].units
            )

            assert RaggedTimeseriesProfile.is_mine(result_ncd, strict=True)

        os.close(fid)
        os.remove(tmpfile)

    def test_csv_to_nc_single_reduce(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'r-single.csv')

        df = pd.read_csv(filepath)
        fid, tmpfile = tempfile.mkstemp(suffix='.nc')

        axes = {
            't': 'time',
            'x': 'lon',
            'y': 'lat',
            'z': 'z'
        }

        df.time = pd.to_datetime(df.time)

        with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, axes=axes, reduce_dims=True) as result_ncd:
            assert 'station' not in result_ncd.dimensions
            assert 'profile' in result_ncd.dimensions
            assert result_ncd.dimensions['profile'].size == 1

            check_vars = ['z', 't090C', 'SP', 'SA', 'SR', 'CT', 'sigma0_CT']
            for v in check_vars:
                npeq(
                    result_ncd.variables[v][:],
                    df[v].values
                )

            assert result_ncd.variables['station'][0] == df.station.iloc[0] == 'CH2'
            assert result_ncd.variables['profile'][0] == df.profile.iloc[0] == '030617B'
            assert result_ncd.variables['lat'].size == 1
            assert result_ncd.variables['lat'].ndim == 0  # Reduced to 0
            assert result_ncd.variables['lat'][0] == df.lat.iloc[0] == 33.558
            assert result_ncd.variables['lon'].size == 1
            assert result_ncd.variables['lon'].ndim == 0  # Reduced to 0
            assert result_ncd.variables['lon'][0] == df.lon.iloc[0] == -118.405

            assert RaggedTimeseriesProfile.is_mine(result_ncd, strict=True)

        os.close(fid)
        os.remove(tmpfile)

    def test_rtp_single(self):
        filepath = os.path.join(os.path.dirname(__file__), 'resources', 'r-ctd-single.nc')

        with RaggedTimeseriesProfile(filepath) as ncd:
            fid, tmpfile = tempfile.mkstemp(suffix='.nc')
            df = ncd.to_dataframe(clean_rows=False)

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile) as result_ncd:
                assert 'station' in result_ncd.dimensions
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, unique_dims=True) as result_ncd:
                assert 'station_dim' in result_ncd.dimensions
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True) as result_ncd:
                # Even though we pass reduce_dims, there are two stations so it is not reduced
                assert 'station' not in result_ncd.dimensions
                assert 'profile' in result_ncd.dimensions
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, unlimited=True) as result_ncd:
                assert 'station' in result_ncd.dimensions
                assert 'profile' in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, reduce_dims=True, unlimited=True) as result_ncd:
                assert 'station' not in result_ncd.dimensions
                assert 'profile' in result_ncd.dimensions
                assert result_ncd.dimensions['obs'].isunlimited() is True
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            with RaggedTimeseriesProfile.from_dataframe(df, tmpfile, unique_dims=True, reduce_dims=False, unlimited=True) as result_ncd:
                assert 'station_dim' in result_ncd.dimensions
                assert 'profile_dim' in result_ncd.dimensions
                assert result_ncd.dimensions['obs_dim'].isunlimited() is True
            test_is_mine(RaggedTimeseriesProfile, tmpfile)  # Try to load it again

            os.close(fid)
            os.remove(tmpfile)
