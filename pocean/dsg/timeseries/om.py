#!python
# coding=utf-8
from pocean.cf import CFDataset

from pocean import logger


class OrthogonalMultidimensionalTimeseries(CFDataset):

    @classmethod
    def is_mine(cls, dsg):
        try:
            rvars = dsg.filter_by_attrs(cf_role='timeseries_id')
            assert len(rvars) == 1
            assert dsg.featureType.lower() == 'timeseries'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1

            # Not a CR
            assert not dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )

            # Not an IR
            assert not dsg.filter_by_attrs(
                instance_dimension=lambda x: x is not None
            )

            # OM files will always have a time variable with one dimension.
            assert len(dsg.t_axes()[0].dimensions) == 1

            # Allow for string variables
            rvar = rvars[0]
            # 0 = single
            # 1 = array of strings/ints/bytes/etc
            # 2 = array of character arrays
            assert 0 <= len(rvar.dimensions) <= 2

        except AssertionError:
            return False

        return True

    def from_dataframe(self, df, variable_attributes=None, global_attributes=None):
        variable_attributes = variable_attributes or {}
        global_attributes = global_attributes or {}
        raise NotImplementedError

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError
