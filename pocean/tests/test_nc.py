#!python
# coding=utf-8
import os
import unittest
import tempfile

from pocean.dataset import EnhancedDataset
from pocean.meta import MetaInterface, ncpyattributes

import logging
from pocean import logger as L
L.level = logging.INFO
L.handlers = [logging.StreamHandler()]


class TestJsonDataset(unittest.TestCase):

    def setUp(self):
        self.maxDiff = 9999
        self.hdl, self.ncdf = tempfile.mkstemp(prefix='pocean_test_')

    def tearDown(self):
        os.close(self.hdl)
        os.remove(self.ncdf)

    def test_lvl0_apply(self):
        jsf = os.path.join(os.path.dirname(__file__), "resources/coamps_lvl0.json")
        mi = MetaInterface.from_jsonfile(jsf)

        with EnhancedDataset(self.ncdf, 'w') as ncd:
            ncd.__apply_meta_interface__(mi)

            assert { k: v.size for k, v in ncd.dimensions.items() } == mi['dimensions']

            fileglobatts = mi['attributes']
            newglobatts = {}
            for nk in ncd.ncattrs():
                newglobatts[nk] = ncd.getncattr(nk)

            self.assertDictEqual(
                fileglobatts,
                newglobatts
            )

            for k, v in ncd.variables.items():

                filevaratts = mi['variables'][k]['attributes']
                newvaratts = ncpyattributes(dict(v.__dict__), verbose=False)

                # _FillValue gets added even if it wasn't in the original attributes
                if '_FillValue' in newvaratts:
                    del newvaratts['_FillValue']

                self.assertDictEqual(
                    filevaratts,
                    newvaratts
                )

    def test_lvl2_apply(self):
        jsf = os.path.join(os.path.dirname(__file__), "resources/coamps_lvl2.json")
        mi = MetaInterface.from_jsonfile(jsf)

        with EnhancedDataset(self.ncdf, 'w') as ncd:
            ncd.__apply_meta_interface__(mi)

            assert { k: v.size for k, v in ncd.dimensions.items() } == mi['dimensions']

            fileglobatts = { k: v['data'] for k, v in mi['attributes'].items() }
            newglobatts = {}
            for nk in ncd.ncattrs():
                newglobatts[nk] = ncd.getncattr(nk)

            self.assertDictEqual(
                fileglobatts,
                newglobatts
            )

            for k, v in ncd.variables.items():

                filevaratts = { k: v['data'] for k, v in mi['variables'][k]['attributes'].items() }
                newvaratts = ncpyattributes(dict(v.__dict__), verbose=False)

                # _FillValue gets added even if it wasn't in the original attributes
                if '_FillValue' in newvaratts:
                    del newvaratts['_FillValue']

                self.assertDictEqual(
                    filevaratts,
                    newvaratts
                )

    def test_input_output(self):
        ncfile = os.path.join(os.path.dirname(__file__), "resources/coamps.nc")

        with EnhancedDataset(ncfile, 'r') as original_ncd:
            mi = original_ncd.__meta_interface__

            with EnhancedDataset(self.ncdf, 'w') as ncd:
                ncd.__apply_meta_interface__(mi)

                self.assertDictEqual(
                    ncpyattributes(dict(original_ncd.__dict__)),
                    ncpyattributes(dict(ncd.__dict__))
                )

                for k, v in original_ncd.variables.items():

                    oldatts = ncpyattributes(dict(v.__dict__))
                    newatts = ncpyattributes(dict(ncd.variables[k].__dict__))

                    # _FillValue gets added even if it wasn't in the original attributes
                    if '_FillValue' in newatts:
                        del newatts['_FillValue']

                    self.assertDictEqual(
                        oldatts,
                        newatts
                    )
