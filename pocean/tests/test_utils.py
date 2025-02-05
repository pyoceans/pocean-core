#!python
import logging
import os
import shutil
import tempfile
import unittest

import netCDF4 as nc4
import numpy as np
import pytest

from pocean import logger
from pocean.dataset import EnhancedDataset
from pocean.utils import generic_masked, get_default_axes, normalize_array

logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.input_file = os.path.join(os.path.dirname(__file__), "resources/coamps.nc")

    def test_single_attr_filter(self):
        with EnhancedDataset(self.input_file) as nc:
            grid_spacing_vars = nc.filter_by_attrs(grid_spacing="4.0 km")

            x = nc.variables.get("x")
            y = nc.variables.get("y")

            self.assertEqual(len(grid_spacing_vars), 2)
            assert x in grid_spacing_vars
            assert y in grid_spacing_vars

    def test_multiple_attr_filter(self):
        with EnhancedDataset(self.input_file) as nc:
            grid_spacing_vars = nc.filter_by_attrs(
                grid_spacing="4.0 km", standard_name="projection_y_coordinate"
            )

            y = nc.variables.get("y")

            self.assertEqual(len(grid_spacing_vars), 1)
            assert y in grid_spacing_vars

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_generic_masked_bad_min_max_value(self):
        fid, tpath = tempfile.mkstemp(suffix=".nc", prefix="pocean-test")
        shutil.copy2(self.input_file, tpath)

        with EnhancedDataset(tpath, "a") as ncd:
            v = ncd.variables["v_component_wind_true_direction_all_geometries"]
            v.valid_min = np.float32(0.1)
            v.valid_max = np.float32(0.1)
            r = generic_masked(v[:], attrs=ncd.vatts(v.name))
            rflat = r.flatten()
            assert rflat[~rflat.mask].size == 0

            # Create a byte variable with a float valid_min and valid_max
            # to make sure it doesn't error
            b = ncd.createVariable("imabyte", "b")
            b.valid_min = 0
            b.valid_max = np.int16(600)  # this is over a byte and thus invalid
            b[:] = 3
            r = generic_masked(b[:], attrs=ncd.vatts(b.name))
            assert np.all(r.mask == False)  # noqa

            b.valid_min = 0
            b.valid_max = 2
            r = generic_masked(b[:], attrs=ncd.vatts(b.name))
            assert np.all(r.mask == True)  # noqa

            c = ncd.createVariable("imanotherbyte", "f4")
            c.setncattr("valid_min", b"0")
            c.setncattr("valid_max", b"9")
            c[:] = 3
            r = generic_masked(c[:], attrs=ncd.vatts(c.name))
            assert np.all(r.mask == False)  # noqa

            c = ncd.createVariable("imarange", "f4")
            c.valid_range = [0.0, 2.0]
            c[:] = 3.0
            r = generic_masked(c[:], attrs=ncd.vatts(c.name))
            assert np.all(r.mask == True)  # noqa

            c.valid_range = [0.0, 2.0]
            c[:] = 1.0
            r = generic_masked(c[:], attrs=ncd.vatts(c.name))
            assert np.all(r.mask == False)  # noqa

        os.close(fid)
        if os.path.exists(tpath):
            os.remove(tpath)
