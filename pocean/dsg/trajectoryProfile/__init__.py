#!python
# coding=utf-8
from pocean.dsg.profile import profile_calculated_metadata
from pocean.dsg.trajectory import trajectories_meta


def trajectory_profile_calculated_metadata(df, geometries=True):

    trajectories = {}
    for tid, tgroup in df.groupby('trajectory'):
        tgroup = tgroup.sort_values('t')
        trajectories[tid] = profile_calculated_metadata(tgroup, geometries)

    return trajectories_meta(
        min_z=df.z.min(),
        max_z=df.z.max(),
        min_t=df.t.min(),
        max_t=df.t.max(),
        trajectories=trajectories
    )
