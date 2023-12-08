#!python
import logging
import os
import unittest

from pocean import logger as L
from pocean.cf import CFDataset
from pocean.dsg import OrthogonalMultidimensionalTimeseries as omt

L.level = logging.INFO
L.handlers = [logging.StreamHandler()]


class TestCFDatasetLoad(unittest.TestCase):

    def test_load_url(self):
        ncd = CFDataset.load('https://geoport.whoi.edu/thredds/dodsC/usgs/data2/emontgomery/stellwagen/CF-1.6/ARGO_MERCHANT/1211-AA.cdf')
        assert omt.is_mine(ncd) is True
        ncd.close()

    def test_load_strict(self):
        ncfile = os.path.join(os.path.dirname(__file__), 'dsg', 'profile', 'resources', 'om-single.nc')

        ncd = CFDataset.load(ncfile)
        assert omt.is_mine(ncd) is False
        with self.assertRaises(BaseException):
            omt.is_mine(ncd, strict=True)
        ncd.close()
