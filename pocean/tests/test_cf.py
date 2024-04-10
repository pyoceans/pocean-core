#!python
# coding=utf-8
import logging
import os
import unittest

import pooch

from pocean import logger as L
from pocean.cf import CFDataset
from pocean.dsg import OrthogonalMultidimensionalTimeseries as omt

L.level = logging.INFO
L.handlers = [logging.StreamHandler()]

url = "https://github.com/pyoceans/pocean-core/releases/download"
version = "3.1.0"

fname = pooch.retrieve(
    url=f"{url}/{version}/1211-AA.cdf",
    known_hash="sha256:2b535b768d5edf6bf326dbe926eb164f4d4a18c685bc56211b5ef43e40e6a55e",
)

class TestCFDatasetLoad(unittest.TestCase):

    def test_load_url(self):
        # File downloaded from https://geoport.usgs.esipfed.org/thredds/dodsC/silt/usgs/Projects/stellwagen/CF-1.6/ARGO_MERCHANT/1211-AA.cdf.html
        ncd = CFDataset.load(fname)
        assert omt.is_mine(ncd) is True
        ncd.close()

    def test_load_strict(self):
        ncfile = os.path.join(os.path.dirname(__file__), 'dsg', 'profile', 'resources', 'om-single.nc')

        ncd = CFDataset.load(ncfile)
        assert omt.is_mine(ncd) is False
        with self.assertRaises(BaseException):
            omt.is_mine(ncd, strict=True)
        ncd.close()
