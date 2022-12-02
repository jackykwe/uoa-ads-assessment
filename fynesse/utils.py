# General util functions are placed here
import math
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
import shapely


def counts_to_probability(counts_list):
    # Duck typing problems...
    return np.array(counts_list) / sum(counts_list)


def merge_categories(dict_of_categories_to_merge):
    """
    Prepares a dictionary ready to be passed as tags argument to osmnx.geometries.geometries_from_bbox()
    A category is a dictionaries where the keys are str, and the values are either list or True.
    """
    result = defaultdict(lambda: set())
    for _, category in dict_of_categories_to_merge.items():
        for k, ls_or_True in category.items():
            # this block abuses duck-typing in Python
            if result[k] is True:
                continue
            if ls_or_True is True:
                result[k] = True
                continue
            result[k].update(ls_or_True)  # ls_or_True is a list here

    # At this point, set_or_True is either a set or True
    return {k: (set_or_True if set_or_True is True else list(set_or_True)) for k, set_or_True in result.items()}


def distance_km_to_degrees(distance_in_km):
    # Circumference of the Earth is arround 40_000 km
    return distance_in_km / 40_000 * 360


def get_prices_near_interested_point(gdf_pppodata, latitude, longitude, box_width_km, box_height_km, maximum_num_prices_to_return=10):
    degree_distance_threshold = distance_km_to_degrees(
        max(box_width_km, box_height_km)
    )
    result = gdf_pppodata[
        (
            abs(gdf_pppodata["latitude"] - Decimal(latitude))
            < degree_distance_threshold
        ) &
        (
            abs(gdf_pppodata["longitude"] - Decimal(longitude))
            < degree_distance_threshold
        )
    ]["price"]
    if len(result) == 0:
        # should not see this exception, an earlier exception should've been thrown
        raise Exception("Unable to get any nearby property prices")
    while True:
        degree_distance_threshold /= 10
        new_result = gdf_pppodata[
            (
                abs(gdf_pppodata["latitude"] - Decimal(latitude))
                < degree_distance_threshold
            ) & (
                abs(gdf_pppodata["longitude"] - Decimal(longitude))
                < degree_distance_threshold
            )
        ]["price"]
        if len(new_result) <= maximum_num_prices_to_return:
            return result.sort_values()
        result = new_result


def filter_pois(pois_gdf, category):
    """
    Filter, keeping only POIs (rows) that match at least one of the provided category dictionary
    :param pois_df: the dataframe returned by osmnx.geometries_from_bbox()
    :param category: a category (dictionary that maps string keys to True/a list)
    """
    filter = pd.Series(False, index=pois_gdf.index)
    for tag, ls_or_True in category.items():
        if tag not in pois_gdf.columns:
            # sometimes, no POIs that match a particular tag is found in the bounding box.
            # # In this case, there won't be a column for it.
            continue
        # abusing duck-typing
        if ls_or_True is True:
            filter |= pois_gdf[tag].notna()
        else:
            filter |= pois_gdf[tag].isin(ls_or_True)
    return pois_gdf[filter]


def find_number_within_bounding_box_of_point(point_geometry, gdf_pois_category, box_width_km, box_height_km):
    box_width = distance_km_to_degrees(box_width_km)
    box_height = distance_km_to_degrees(box_height_km)
    north = point_geometry.y + box_height / 2
    south = point_geometry.y - box_height / 2
    west = point_geometry.x - box_width / 2
    east = point_geometry.x + box_width / 2
    bounding_box_of_point = shapely.geometry.box(west, south, east, north)
    filter = gdf_pois_category["geometry"].apply(
        lambda geo: geo.intersects(bounding_box_of_point)
    )
    gdf_pois_category_within_bounding_box_of_point = gdf_pois_category.loc[filter]

    return len(gdf_pois_category_within_bounding_box_of_point)


def find_degree_distance_to_closest_within_bounding_box_of_point(point_geometry, gdf_pois_category, box_width_km, box_height_km):
    box_width = distance_km_to_degrees(box_width_km)
    box_height = distance_km_to_degrees(box_height_km)
    north = point_geometry.y + box_height / 2
    south = point_geometry.y - box_height / 2
    west = point_geometry.x - box_width / 2
    east = point_geometry.x + box_width / 2
    bounding_box_of_point = shapely.geometry.box(west, south, east, north)
    filter = gdf_pois_category["geometry"].apply(
        lambda geo: geo.intersects(bounding_box_of_point))
    gdf_pois_category_within_bounding_box_of_point = gdf_pois_category.loc[filter]

    if len(gdf_pois_category_within_bounding_box_of_point) == 0:
        # assume that we can find the next closest one if our bounding box dimensions is scaled by x2 (see diagram)
        return math.sqrt(2) * max(box_width, box_height)
    return (gdf_pois_category_within_bounding_box_of_point["geometry"].apply(lambda geo: geo.distance(point_geometry))).min()
