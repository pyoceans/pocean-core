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


input_file = os.path.join(os.path.dirname(__file__), "resources/coamps.nc")

@pytest.fixture
def load_dataset():
    nc = EnhancedDataset(input_file)
    yield nc
    nc.close()


@pytest.fixture
def copy_dataset():
    fid, tpath = tempfile.mkstemp(suffix=".nc", prefix="pocean-test")
    shutil.copy(input_file, tpath)
    ncd =  EnhancedDataset(tpath, "a")
    yield ncd
    ncd.close()
    os.close(fid)
    if os.path.exists(tpath):
        os.remove(tpath)

def test_single_attr_filter(load_dataset):
    grid_spacing_vars = load_dataset.filter_by_attrs(grid_spacing="4.0 km")

    x = load_dataset.variables.get("x")
    y = load_dataset.variables.get("y")

    assert len(grid_spacing_vars) == 2
    assert x in grid_spacing_vars
    assert y in grid_spacing_vars

def test_multiple_attr_filter(load_dataset):
    grid_spacing_vars = load_dataset.filter_by_attrs(
        grid_spacing="4.0 km", standard_name="projection_y_coordinate"
    )

    y = load_dataset.variables.get("y")

    assert len(grid_spacing_vars) == 1
    assert y in grid_spacing_vars

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generic_masked_bad_min_max_value(copy_dataset):

    v = copy_dataset.variables["v_component_wind_true_direction_all_geometries"]
    v.valid_min = np.float32(0.1)
    v.valid_max = np.float32(0.1)
    r = generic_masked(v[:], attrs=copy_dataset.vatts(v.name))
    rflat = r.flatten()
    assert rflat[~rflat.mask].size == 0

    # Create a byte variable with a float valid_min and valid_max
    # to make sure it doesn't error
    b = copy_dataset.createVariable("imabyte", "b")
    b.valid_min = 0
    b.valid_max = np.int16(600)  # this is over a byte and thus invalid
    b[:] = 3
    r = generic_masked(b[:], attrs=copy_dataset.vatts(b.name))
    assert np.all(r.mask == False)  # noqa

    b.valid_min = 0
    b.valid_max = 2
    r = generic_masked(b[:], attrs=copy_dataset.vatts(b.name))
    assert np.all(r.mask == True)  # noqa

    c = copy_dataset.createVariable("imanotherbyte", "f4")
    c.setncattr("valid_min", b"0")
    c.setncattr("valid_max", b"9")
    c[:] = 3
    r = generic_masked(c[:], attrs=copy_dataset.vatts(c.name))
    assert np.all(r.mask == False)  # noqa

    c = copy_dataset.createVariable("imarange", "f4")
    c.valid_range = [0.0, 2.0]
    c[:] = 3.0
    r = generic_masked(c[:], attrs=copy_dataset.vatts(c.name))
    assert np.all(r.mask == True)  # noqa

    c.valid_range = [0.0, 2.0]
    c[:] = 1.0
    r = generic_masked(c[:], attrs=copy_dataset.vatts(c.name))
    assert np.all(r.mask == False)  # noqa
