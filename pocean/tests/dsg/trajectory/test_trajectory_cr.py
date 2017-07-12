#!python
# coding=utf-8
import unittest

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
