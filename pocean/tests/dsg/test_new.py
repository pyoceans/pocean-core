import logging
from os.path import dirname as dn
from os.path import join as jn

import pytest

from pocean import logger
from pocean.cf import CFDataset
from pocean.dsg import *
from pocean.utils import all_subclasses

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


@pytest.mark.parametrize("klass,fp", [
    (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-single.nc')),
    (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-multiple.nc')),
    (OrthogonalMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'om-1dy11.nc')),
    (IncompleteMultidimensionalProfile,           jn(dn(__file__), 'profile', 'resources', 'im-multiple.nc')),
    (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'im-single.nc')),
    (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'im-multiple.nc')),
    (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'im-multiple-nonstring.nc')),
    (IncompleteMultidimensionalTrajectory,        jn(dn(__file__), 'trajectory', 'resources', 'wave-glider-int-attrs.nc')),
    (ContiguousRaggedTrajectory,                  jn(dn(__file__), 'trajectory', 'resources', 'cr-multiple.nc')),
    (ContiguousRaggedTrajectory,                  jn(dn(__file__), 'trajectory', 'resources', 'cr-oot-A.nc')),
    (ContiguousRaggedTrajectory,                  jn(dn(__file__), 'trajectory', 'resources', 'cr-oot-B.nc')),
    (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-single.nc')),
    (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-multiple.nc')),
    (ContiguousRaggedTrajectoryProfile,           jn(dn(__file__), 'trajectoryProfile', 'resources', 'cr-missing-time.nc')),
    (IncompleteMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'im-multiple.nc')),
    (OrthogonalMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'om-single.nc')),
    (OrthogonalMultidimensionalTimeseries,        jn(dn(__file__), 'timeseries', 'resources', 'om-multiple.nc')),
    #(IndexedRaggedTimeseries,                     jn(dn(__file__), 'timeseries', 'resources', 'cr-multiple.nc')),
    #(ContiguousRaggedTimeseries,                  jn(dn(__file__), 'timeseries', 'resources', 'cr-multiple.nc')),
    (OrthogonalMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'om-multiple.nc')),
    (IncompleteMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'im-single.nc')),
    (IncompleteMultidimensionalTimeseriesProfile, jn(dn(__file__), 'timeseriesProfile', 'resources', 'im-multiple.nc')),
    (RaggedTimeseriesProfile,                     jn(dn(__file__), 'timeseriesProfile', 'resources', 'r-single.nc')),
    (RaggedTimeseriesProfile,                     jn(dn(__file__), 'timeseriesProfile', 'resources', 'r-multiple.nc')),
])
def test_is_mine(klass, fp):
    with CFDataset.load(fp) as dsg:
        assert dsg.__class__ == klass

    allsubs = list(all_subclasses(CFDataset))
    subs = [ s for s in allsubs if s != klass ]
    with CFDataset(fp) as dsg:
        logger.debug(f'\nTesting {klass.__name__}')
        assert klass.is_mine(dsg, strict=True) is True
        for s in subs:
            if hasattr(s, 'is_mine'):
                logger.debug(f'  * Trying {s.__name__}...')
                assert s.is_mine(dsg) is False
