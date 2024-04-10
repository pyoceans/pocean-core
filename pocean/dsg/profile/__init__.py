#!python
from collections import namedtuple

from shapely.geometry import LineString, Point

from pocean.utils import logger as L  # noqa
from pocean.utils import (
    unique_justseen,
)

profile_meta = namedtuple('Profile', [
    'min_z',
    'max_z',
    't',
    'x',
    'y',
    'id',
    'geometry'
])
profiles_meta = namedtuple('ProfileCollection', [
    'min_z',
    'max_z',
    'min_t',
    'max_t',
    'profiles',
    'geometry'
])


def profile_calculated_metadata(df, axes, geometries=True):
    profiles = {}
    for pid, pgroup in df.groupby(axes.profile):
        pgroup = pgroup.sort_values(axes.t)
        first_row = pgroup.iloc[0]
        profiles[pid] = profile_meta(
            min_z=pgroup[axes.z].min(),
            max_z=pgroup[axes.z].max(),
            t=first_row[axes.t],
            x=first_row[axes.x],
            y=first_row[axes.y],
            id=pid,
            geometry=Point(first_row[axes.x], first_row[axes.y])
        )

    if geometries:
        null_coordinates = df[axes.x].isnull() | df[axes.y].isnull()
        coords = list(unique_justseen(zip(
            df.loc[~null_coordinates, axes.x].tolist(),
            df.loc[~null_coordinates, axes.y].tolist()
        )))
    else:
        # Calculate the geometry as the linestring between all of the profile points
        coords = [ p.geometry for _, p in profiles.items() ]

    geometry = None
    if len(coords) > 1:
        geometry = LineString(coords)
    elif len(coords) == 1:
        geometry = Point(coords[0])

    return profiles_meta(
        min_z=df[axes.z].min(),
        max_z=df[axes.z].max(),
        min_t=df[axes.t].min(),
        max_t=df[axes.t].max(),
        profiles=profiles,
        geometry=geometry
    )
