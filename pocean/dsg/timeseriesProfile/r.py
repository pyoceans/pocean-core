#!python
# coding=utf-8
from pocean.cf import CFDataset
from pocean.utils import normalize_array
from pocean.utils import logger  # noqa


class RaggedTimeseriesProfile(CFDataset):

    @classmethod
    def required_columns(cls):
        return ['profile']

    @classmethod
    def reduce_variables(cls):
        return {}

    @classmethod
    def apply_indexes(cls, df):
        return df.set_index(['profile', 'z'])

    @classmethod
    def attrs(cls, xds):
        # Add
        return {
            'attributes': {
                'featureType': 'profile',
                'cdm_data_type': 'Profile',
                'Conventions': 'CF-1.6'
            },
            'variables': {
                'profile': {
                    'attributes': {
                        'cf_role': 'profile_id',
                        'long_name' : 'profile identifier'
                    }
                },
                'time': {
                    'axis': 'T',
                },
                'latitude': {
                    'attributes': {
                        'axis': 'Y',
                        'standard_name': 'latitude',
                        'units': 'degrees_north'
                    }
                },
                'longitude': {
                    'attributes': {
                        'axis': 'Y',
                        'standard_name': 'longitude',
                        'units': 'degrees_east'
                    }
                },
                'z': {
                    'attributes': {
                        'axis': 'Z',
                        'standard_name': 'depth',
                        'positive': 'down'
                    }
                }
            }
        }

    @classmethod
    def is_mine(cls, dsg):
        try:
            assert dsg.featureType.lower() == 'timeseriesprofile'
            assert len(dsg.t_axes()) >= 1
            assert len(dsg.x_axes()) >= 1
            assert len(dsg.y_axes()) >= 1
            assert len(dsg.z_axes()) >= 1

            o_index_vars = dsg.filter_by_attrs(
                sample_dimension=lambda x: x is not None
            )
            assert len(o_index_vars) == 1
            assert o_index_vars[0].sample_dimension in dsg.dimensions  # Sample dimension

            svar = dsg.filter_by_attrs(
                cf_role='timeseries_id'
            )[0]
            sdata = normalize_array(svar)
            if len(sdata.shape) > 0:
                r_index_vars = dsg.filter_by_attrs(
                    instance_dimension=lambda x: x is not None
                )
                assert len(r_index_vars) == 1
                assert r_index_vars[0].instance_dimension in dsg.dimensions  # Station dimension

        except AssertionError:
            return False

        return True

    def from_dataframe(cls, df, output, **kwargs):
        raise NotImplementedError

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError
