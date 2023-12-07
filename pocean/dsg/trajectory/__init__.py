#!python
from collections import namedtuple

from shapely.geometry import LineString, Point

from pocean.utils import (
    unique_justseen,
)

trajectory_meta = namedtuple('Trajectory', [
    'min_z',
    'max_z',
    'min_t',
    'max_t',
    'geometry'
])

trajectories_meta = namedtuple('TrajectoryCollection', [
    'min_z',
    'max_z',
    'min_t',
    'max_t',
    'trajectories'
])


def trajectory_calculated_metadata(df, axes, geometries=True):
    trajectories = {}
    for tid, tgroup in df.groupby(axes.trajectory):
        tgroup = tgroup.sort_values(axes.t)

        if geometries:
            null_coordinates = tgroup[axes.x].isnull() | tgroup[axes.y].isnull()
            coords = list(unique_justseen(zip(
                tgroup.loc[~null_coordinates, axes.x].tolist(),
                tgroup.loc[~null_coordinates, axes.y].tolist()
            )))
        else:
            # Calculate the geometry as the linestring between all of the profile points
            first_row = tgroup.iloc[0]
            coords = [(first_row[axes.x], first_row[axes.y])]

        geometry = None
        if len(coords) > 1:
            geometry = LineString(coords)
        elif len(coords) == 1:
            geometry = Point(coords[0])

        trajectories[tid] = trajectory_meta(
            min_z=tgroup[axes.z].min(),
            max_z=tgroup[axes.z].max(),
            min_t=tgroup[axes.t].min(),
            max_t=tgroup[axes.t].max(),
            geometry=geometry
        )

    return trajectories_meta(
        min_z=df[axes.z].min(),
        max_z=df[axes.z].max(),
        min_t=df[axes.t].min(),
        max_t=df[axes.t].max(),
        trajectories=trajectories
    )
