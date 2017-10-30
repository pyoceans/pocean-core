#!python
# coding=utf-8
from collections import namedtuple

from shapely.geometry import Point, LineString

from pocean.utils import (
    unique_justseen,
)

profile_meta = namedtuple('Profile', [
    'min_z',
    'max_z',
    't',
    'x',
    'y',
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


def profile_calculated_metadata(df, geometries=True):
    profiles = {}
    for pid, pgroup in df.groupby('profile'):
        pgroup = pgroup.sort_values('t')
        first_row = pgroup.iloc[0]
        profiles[pid] = profile_meta(
            min_z=pgroup.z.min(),
            max_z=pgroup.z.max(),
            t=first_row.t,
            x=first_row.x,
            y=first_row.y,
            geometry=Point(first_row.x, first_row.y)
        )

    if geometries:
        null_coordinates = df.x.isnull() | df.y.isnull()
        coords = list(unique_justseen(zip(
            df.loc[~null_coordinates, 'x'].tolist(),
            df.loc[~null_coordinates, 'y'].tolist()
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
        min_z=df.z.min(),
        max_z=df.z.max(),
        min_t=df.t.min(),
        max_t=df.t.max(),
        profiles=profiles,
        geometry=geometry
    )
