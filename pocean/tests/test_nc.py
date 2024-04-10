#!python
import logging
import os
import tempfile
import unittest

from numpy import testing as npt

from pocean import logger as L
from pocean.cf import CFDataset
from pocean.dataset import EnhancedDataset
from pocean.meta import MetaInterface, ncpyattributes

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
            ncd.apply_meta(mi)

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

                if 'missing_value' in filevaratts:
                    del filevaratts['missing_value']

                self.assertDictEqual(
                    filevaratts,
                    newvaratts
                )

    def test_lvl2_apply(self):
        jsf = os.path.join(os.path.dirname(__file__), "resources/coamps_lvl2.json")
        mi = MetaInterface.from_jsonfile(jsf)

        with EnhancedDataset(self.ncdf, 'w') as ncd:
            ncd.apply_meta(mi)

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

                if 'missing_value' in filevaratts:
                    del filevaratts['missing_value']

                self.assertDictEqual(
                    filevaratts,
                    newvaratts
                )

    def test_input_output(self):
        ncfile = os.path.join(os.path.dirname(__file__), "resources/coamps.nc")

        with EnhancedDataset(ncfile, 'r') as original_ncd:
            mi = original_ncd.meta()

            with EnhancedDataset(self.ncdf, 'w') as ncd:
                ncd.apply_meta(mi)

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

                    if 'missing_value' in oldatts:
                        del oldatts['missing_value']

                    self.assertDictEqual(
                        oldatts,
                        newatts
                    )

    def test_serialize_and_reload_data(self):
        ncfile = os.path.join(os.path.dirname(__file__), "resources/qc-month.nc")

        with CFDataset(ncfile) as cfncd:

            # Data from netCDF variable
            ncdata = cfncd.variables['data1'][:]

            # Not filled
            meta = cfncd.json(return_data=True, fill_data=False)
            jsdata = meta['variables']['data1']['data']
            npt.assert_array_equal(ncdata, jsdata)
            fhandle1, fname1 = tempfile.mkstemp()
            with CFDataset(fname1, 'w') as newcf:
                newcf.apply_json(meta)
            with CFDataset(fname1, 'r') as rcf:
                newncdata = rcf.variables['data1'][:]
                npt.assert_array_equal(ncdata, newncdata)
            os.close(fhandle1)
            os.remove(fname1)

            # Filled
            meta = cfncd.json(return_data=True, fill_data=True)
            jsdata = meta['variables']['data1']['data']
            npt.assert_array_equal(ncdata, jsdata)
            fhandle2, fname2 = tempfile.mkstemp()
            with CFDataset(fname2, 'w') as newcf:
                newcf.apply_json(meta)

            with CFDataset(fname2, 'r') as rcf:
                newncdata = rcf.variables['data1'][:]
                npt.assert_array_equal(ncdata, newncdata)

            os.close(fhandle2)
            os.remove(fname2)
