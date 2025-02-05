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
