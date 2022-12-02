# This file contains code for suporting addressing questions in the data

import os
import uuid
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx
import pandas as pd
import shapely
import statsmodels.api as sm
from tqdm.auto import tqdm

from . import assess, aws_utils, constants, utils

"""Address a particular question that arises from the data"""

# dictionary of (name of category to the dictionary representing the category)
DEFAULT_DICT_OF_CATEGORIES = {
    "food_groceries_retail": {
        "amenity": ["cafe", "fast_food", "food_court", "restaurant", "marketplace"],
        "building": ["apartments", "retail", "supermarket"],
        "shop": True  # True means include all kinds of shops
    },
    "security_social": {
        "amenity": ["fire_station", "police", "post_box", "post_office", "place_of_worship"],
        "building": ["cathedral", "chapel", "church", "kingdom_hall", "monastery", "mosque", "religious", "shrine", "synagogue", "temple", "fire_station", "government", "garage", "garages", "parking"]
    },
    "health": {
        "amenity": ["clinic", "dentist", "doctors", "hospital", "pharmacy", "social_facility"],
        "building": ["hospital"],
        "healthcare": True
    },
    "school": {
        "amenity": ["college", "kindergarten", "language_school", "library", "training", "school", "university"],
        "building": ["college", "kindergarten", "school", "university"],
        "landuse": ["education"],
        "leisure": ["playground"]
    },
    "entertainment": {
        "amenity": ["cinema", "community_centre", "events_venue", "fountain", "theatre", "bbq"],
        "building": ["sports_hall", "stadium"],
        "leisure": ["fitness_station", "sports_centre", "stadium", "swimming_pool", "track"],
        "sport": True,
        "tourism": True,
    },
    # assumption: land transportation only (boat etc. not included)
    "transportation": {
        "amenity": ["bicycle_parking", "bicycle_repair_station", "bicycle_rental", "bus_station", "car_rental", "car_sharing", "car_wash", "charging_station", "fuel", "mortorcycle_parking", "parking", "taxi"],
        "building": ["train_station", "transportation"],
        "public_transport": ["stop_position", "platform", "station", "stop_area"],
        "railway": ["halt", "platform"],
        "public_transport": ["platform", "station", "subway_entrance", "tram_stop"]
    },
    "nature": {
        "boundary": ["forest", "national_park"],
        "landuse": ["farmland", "forest", "meadow"],
        "leisure": ["garden", "nature_reserve", "park"]
    }
}


def predict_price(
    latitude, longitude, date_isostr, property_type,
    dict_of_categories=DEFAULT_DICT_OF_CATEGORIES,
    box_width_km=2,
    box_height_km=2,
    num_years_history=3,
    svg_output_dir="prediction_svgs/"
):
    """
    Price prediction for UK housing.

    :param latitude: a float
    :param longitude: a float
    :param date_isostr: a string in ISO format (e.g. "2022-11-30 00:00" for midnight on 30 November 2022)
    :param property_type: a string, one of "D/S/T/F/O"
    :param dict_of_categories: dictionary of (name of category to the dictionary representing the category).
                               Each entry here will result in 2 features being created:
                               - nearby POIs of that category
                               - shortest distance to any nearby POI of that category
                                 (upper-clamped by sqrt(2) * max(box_width_km, box_height_km).)
    :param box_width_km: an int or float. Distance of bounding box, in km.
                         Only pppodata entries within this bounding box will be fetched.
    :param box_height_km: an int or float. Distance of bounding box, in km.
                          Only pppodata entries within this bounding box will be fetched.
    :param num_years_history: an int. The number of years worth of pppodata entries to pull.
    """
    if property_type not in ("D", "S", "T", "F", "O"):
        raise Exception(
            f"""
            Invalid property_type: expected one of ["D", "S", "T", "F", "O"], got {property_type}
            """
        )

    os.makedirs(svg_output_dir, exist_ok=True)

    prediction_uuid = uuid.uuid4()
    print(
        f"The UUID for this prediction is: {prediction_uuid}. All saved figures will be labelled with this UUID."
    )

    # Select a bounding box around the housing location in latitude and longitude.
    # get all POIs we'd ever need in our generation of features, hence (* 2) in the following lines
    box_width = utils.distance_km_to_degrees(box_width_km * 2)
    box_height = utils.distance_km_to_degrees(box_height_km * 2)
    north = latitude + box_height / 2
    south = latitude - box_height / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2

    # Select a data range around the prediction date.
    date = datetime.fromisoformat(date_isostr)
    sql_cutoff_earliest_date = (
        date - timedelta(days=365 * num_years_history)
    ).isoformat()
    sql_cutoff_latest_date = date.isoformat()

    tags = utils.merge_categories(dict_of_categories)

    # Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    print()
    print("#################")
    print("# FETCHING DATA #")
    print("#################")
    print("Querying rows from AWS...")
    for _ in tqdm((1, ), leave=False):
        df_pppodata = aws_utils.get_pppodata_conditioned(
            property_type,
            sql_cutoff_earliest_date,
            sql_cutoff_latest_date,
            north,
            south,
            east,
            west
        )
    if len(df_pppodata) == 0:
        raise Exception(
            f"Fetched 0 rows from AWS. No model can be trained to generate predictions. Try to increasing box_width_km (currently {box_width_km}), box_height_km (currently {box_height_km}), or num_years_history (currently ({num_years_history}))."
        )
    print(f"Fetched {len(df_pppodata)} rows from AWS.")

    geometry = gpd.points_from_xy(
        df_pppodata["longitude"], df_pppodata["latitude"]
    )
    gdf_pppodata = gpd.GeoDataFrame(df_pppodata, geometry=geometry)
    gdf_pppodata.crs = "EPSG:4326"

    print("Getting POIs from OSM...")
    for _ in tqdm((1, ), leave=False):
        gdf_pois = osmnx.geometries_from_bbox(
            north, south, east, west, tags=tags)
        if "relation" in gdf_pois.index:
            gdf_pois.drop("relation", inplace=True)
    print(
        f"Fetched {len(gdf_pois)} POIs in the bounding box around latitude={latitude}, longitude={longitude}."
    )

    list_of_gdf_pois_categories = [
        utils.filter_pois(gdf_pois, category)
        for _, category in dict_of_categories.items()
    ]
    # Sanity check: spliting the gdf_pois into fragments (gdf_pois_category-s) then unioning them shouldn't result in loss of rows
    assert sum(
        [len(gdf) for gdf in list_of_gdf_pois_categories]
    ) >= len(gdf_pois)

    print("Getting edges from OSM for plotting...")
    for _ in tqdm((1,), leave=False):
        graph = osmnx.graph_from_bbox(north, south, east, west)
    # Retrieve nodes and edges
    nodes, edges = osmnx.graph_to_gdfs(graph)
    print("Got edges from OSM for plotting.")

    print()
    print("################################")
    print("# CHARACTERISATION OF AWS ROWS #")
    print("################################")
    assess.characterise_aws_pppodata(
        prediction_uuid, df_pppodata, print_banners=False
    )

    print()
    print("#######################################################################")
    print("# MAP AROUND PREDICTED HOUSE (at the provided latitude and longitude) #")
    print("#######################################################################")
    fig, ax = plt.subplots(
        figsize=(constants.PLT_MAP_WIDTH, constants.PLT_MAP_HEIGHT))
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Plot all POIs
    for i, gdf_pois_category in enumerate(list_of_gdf_pois_categories):
        if "node" in gdf_pois_category.index:
            gdf_pois_category.loc["node"].plot(
                ax=ax, color="blue", alpha=0.7, marker=f"${i}$", markersize=150)
        if "way" in gdf_pois_category.index:
            gdf_pois_category.loc["way"].plot(ax=ax, color="red", alpha=0.25)
    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_map_around_{latitude}-{longitude}.svg"
        )
    )
    plt.tight_layout()

    print("########################")
    print("# MODEL AND PREDICTION #")
    print("########################")
    # Feature generation
    print("Generating features...")

    # Train a linear model on the data set you have created.
    for gdf_pois_category, category_name in tqdm(zip(list_of_gdf_pois_categories, dict_of_categories)):
        gdf_pppodata[f"{category_name}_count"] = gdf_pppodata["geometry"].apply(
            lambda geo: utils.find_number_within_bounding_box_of_point(geo, gdf_pois_category))
        gdf_pppodata[f"{category_name}_closest_degree_distance"] = gdf_pppodata["geometry"].apply(
            lambda geo: utils.find_degree_distance_to_closest_within_bounding_box_of_point(geo, gdf_pois_category))
    feature_columns_names = [f"{category_name}_count" for category_name in dict_of_categories] + [
        f"{category_name}_closest_degree_distance" for category_name in dict_of_categories
    ]

    prediction_point_geometry = shapely.geometry.Point(longitude, latitude)
    prediction_point_features_dict = {}
    for gdf_pois_category, category_name in tqdm(zip(list_of_gdf_pois_categories, dict_of_categories)):
        prediction_point_features_dict[f"{category_name}_count"] = utils.find_number_within_bounding_box_of_point(
            prediction_point_geometry, gdf_pois_category
        )
        prediction_point_features_dict[f"{category_name}_closest_degree_distance"] = utils.find_degree_distance_to_closest_within_bounding_box_of_point(
            prediction_point_geometry, gdf_pois_category
        )
    # Transform each value into a list, as expected by pd.DataFrame.from_dict()
    prediction_point_features_dict = {
        k: [v]
        for k, v in prediction_point_features_dict.items()
    }
    prediction_point_features = pd.DataFrame.from_dict(
        prediction_point_features_dict
    )[feature_columns_names]
    prediction_point_features.insert(0, "const", 1.0)  # Insert a ones column

    # OLS
    print("Training model...")
    train_Y = gdf_pppodata.sort_values("price", ascending=True)["price"]
    train_X = gdf_pppodata.sort_values("price", ascending=True)[
        feature_columns_names]
    train_X = sm.add_constant(train_X)
    model = sm.OLS(train_Y, train_X)

    results = model.fit()
    print(results.summary())

    # alpha is the significance level. Confidence interval is 1 - alpha.
    Y_pred_of_train_X = results.get_prediction(
        train_X
    ).summary_frame(alpha=0.05)

    print("Plotting model results... (NB: x axis has no meaning)")
    fig, ax = plt.subplots(figsize=(constants.PLT_WIDTH, constants.PLT_HEIGHT))
    ax.scatter(range(len(train_X)), train_Y, zorder=2)
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['mean'], color='red', linestyle='--', zorder=1)
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['obs_ci_lower'], color='red', linestyle='-', zorder=1)
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['obs_ci_upper'], color='red', linestyle='-', zorder=1)
    ax.fill_between(range(len(train_X)), Y_pred_of_train_X['obs_ci_lower'],
                    Y_pred_of_train_X['obs_ci_upper'], color='red', alpha=0.3, zorder=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()

    Y_pred_of_prediction_point = results.get_prediction(
        prediction_point_features).summary_frame(alpha=0.05)
    print(f"\nPrediction:\n{Y_pred_of_prediction_point}\n")

    final_prediction = Y_pred_of_prediction_point["mean"]
    fig, ax = plt.subplots(figsize=(constants.PLT_WIDTH, constants.PLT_HEIGHT))
    hist_values, _, _ = ax.hist(
        gdf_pppodata["price"],
        # between 10 to 1000 bins
        bins=np.clip(len(gdf_pppodata["date_of_transfer_dt"]) // 10, 10, 1000)
    )
    ax.plot(
        [final_prediction, final_prediction],
        [0, hist_values.max()],
        color="black",
        label="predicted price"
    )
    ax.legend()
    ax.set_xlabel("date_of_transfer")
    ax.set_ylabel("count")
    plt.show()
