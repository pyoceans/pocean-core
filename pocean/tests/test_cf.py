#!python
# coding=utf-8
import unittest

from pocean.cf import CFDataset
from pocean.dsg import OrthogonalMultidimensionalTimeseries as omt

import logging
from pocean import logger as L
L.level = logging.INFO
L.handlers = [logging.StreamHandler()]


class TestCFDatasetLoad(unittest.TestCase):

    def test_load_url(self):
        ncd = CFDataset.load('http://geoport.whoi.edu/thredds/dodsC/usgs/data2/emontgomery/stellwagen/CF-1.6/ARGO_MERCHANT/1211-AA.cdf')
        assert omt.is_mine(ncd) is True
        ncd.close()
