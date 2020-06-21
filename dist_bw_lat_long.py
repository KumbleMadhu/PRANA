from haversine import haversine, Unit
from math import pi, sqrt, sin, cos, atan2


def lat_long():
    loc1 = (12.979700, 77.591200)  # (lat, long)
    loc2 = (12.979100, 77.590700)
    dist = haversine(loc1, loc2, unit=Unit.METERS)
    # print(dist)
    if dist <= 10:
        print("WARNING: {} meter(s) could cause possible potential infection!".format(dist))
    else:
        print(dist)
    return ()


if __name__ == '__main__':
    lat_long()
