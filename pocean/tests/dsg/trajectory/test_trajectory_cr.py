# -*- coding: utf-8 -*-
import os

import unittest
from dateutil.parser import parse as dtparse
import numpy as np

from pocean.dsg import ContiguousRaggedTrajectory

import logging
from pocean import logger
logger.level = logging.INFO
logger.handlers = [logging.StreamHandler()]


class TestContiguousRaggedTrajectory(unittest.TestCase):

    def setUp(self):
        pass

    def test_crt_load(self):
        pass

    def test_crt_dataframe(self):
        pass

    def test_crt_calculated_metadata(self):
        pass
