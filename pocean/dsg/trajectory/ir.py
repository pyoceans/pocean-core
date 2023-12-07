#!python
from pocean.cf import CFDataset


class IndexedRaggedTrajectory(CFDataset):

    def from_dataframe(cls, df, output, **kwargs):
        raise NotImplementedError

    def calculated_metadata(self, df=None, geometries=True, clean_cols=True, clean_rows=True):
        # if df is None:
        #     df = self.to_dataframe(clean_cols=clean_cols, clean_rows=clean_rows)
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError
