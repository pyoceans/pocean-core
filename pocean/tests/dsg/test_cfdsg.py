#!python
# coding=utf-8
from os.path import join as jn
from os.path import dirname as dn

import pytest
import xarray as xr

from pocean.cf import CFDataset
from pocean.utils import all_subclasses
from pocean.dsg import *


import logging
from pocean import logger
logger.level = logging.DEBUG
logger.handlers = [logging.StreamHandler()]


@pytest.mark.parametrize("klass,fp,axes", [
    # (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-single.nc'), {}),
    # (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-multiple.nc'), {}),
    # (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-1dy11.nc'), {}),
    # (IncompleteMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'im-multiple.nc'), {}),
    # (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'im-single.nc'), {}),
    # (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'im-multiple.nc'), {}),
    # (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-single.nc'), {}),
    # (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-multiple.nc'), {}),
    # (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-missing-time.nc'), {}),
    # (IncompleteMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'im-multiple.nc'), {}),
    # (OrthogonalMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'om-single.nc'), {}),
    # (OrthogonalMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'om-multiple.nc'), {}),
    # (IndexedRaggedTimeseries,                     jn(dn(__file__), 'timeseries', 'resources', 'cr-multiple.nc'), {}),
    # (ContiguousRaggedTimeseries,                  jn(dn(__file__), 'timeseries', 'resources', 'cr-multiple.nc'), {}),
    # (OrthogonalMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'om-multiple.nc'), {}),
    # (IncompleteMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'im-single.nc'), {}),
    # (IncompleteMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'im-multiple.nc'), {}),
    #(RaggedTimeseriesProfile,                     jn(dn(__file__), 'timeseriesProfile', 'resources', 'r-single.nc'), dict(t='time', z='height', x='lon', y='lat')),
    (RaggedTimeseriesProfile,                     jn(dn(__file__), 'timeseriesProfile', 'resources', 'r-multiple.nc'), dict(t='time', z='height', x='lon', y='lat')),
])
def test_loading_and_converting(klass, fp, axes):
    xds = xr.open_dataset(fp)
    with open_cfdsg_dataframe(xds.to_dataframe(), klass=klass, axis_mapping=axes) as ncd:
        ncd.cfdsg.update_attrs()
        ncd.cfdsg.reduce()
