#!python
from pocean.dsg.profile import profile_calculated_metadata
from pocean.dsg.trajectory import trajectories_meta


def trajectory_profile_calculated_metadata(df, axes, geometries=True):

    trajectories = {}
    for tid, tgroup in df.groupby(axes.trajectory):
        tgroup = tgroup.sort_values(axes.t)
        trajectories[tid] = profile_calculated_metadata(tgroup, axes, geometries)

    return trajectories_meta(
        min_z=df[axes.z].min(),
        max_z=df[axes.z].max(),
        min_t=df[axes.t].min(),
        max_t=df[axes.t].max(),
        trajectories=trajectories
    )
