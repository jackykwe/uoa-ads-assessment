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

    dict_of_gdf_pois_categories = {
        category_name: utils.filter_pois(gdf_pois, category)
        for category_name, category in dict_of_categories.items()
    }
    # Sanity check: spliting the gdf_pois into fragments (gdf_pois_category-s) then unioning them shouldn't result in loss of rows
    assert sum(
        [len(gdf) for gdf in dict_of_gdf_pois_categories.values()]
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
    # a modified version of assess.characterise_raw_ppdata_and_raw_podata(), which we've looked at before in Question 2
    assess.characterise_aws_pppodata(
        prediction_uuid, df_pppodata, print_banners=False
    )

    print()
    print("##########################################################################")
    print("# MAP AROUND PREDICTED PROPERTY (at the provided latitude and longitude) #")
    print("##########################################################################")
    fig, ax = plt.subplots(
        figsize=(constants.PLT_MAP_WIDTH, constants.PLT_MAP_HEIGHT))
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray", alpha=0.25)
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Plot all POIs
    for i, (category_name, gdf_pois_category) in enumerate(dict_of_gdf_pois_categories.items()):
        if "node" in gdf_pois_category.index:
            gdf_pois_category.loc["node"].plot(
                ax=ax,
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][
                    i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                ],
                alpha=0.7,
                marker=f"${i}$",
                markersize=100,
                label=category_name
            )
        if "way" in gdf_pois_category.index:
            gdf_pois_category.loc["way"].plot(
                ax=ax,
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][
                    i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                ],
                alpha=0.5
            )
    ax.plot(
        longitude, latitude,
        marker="*",
        markersize=25,
        color="black",
        alpha=0.5
    )
    for i in range(len(gdf_pppodata)):
        ax.plot(
            gdf_pppodata["longitude"][i],
            gdf_pppodata["latitude"][i],
            marker="o",
            markersize=10,
            color="black",
            alpha=0.5
        )
    ax.legend()
    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_map_around_{latitude}-{longitude}.svg"
        )
    )
    plt.show()

    print()
    print("######################")
    print("# FEATURE GENERATION #")
    print("######################")
    # Feature generation
    print("Generating features...")

    # Train a linear model on the data set you have created.
    for category_name, gdf_pois_category in tqdm(dict_of_gdf_pois_categories.items()):
        gdf_pppodata[f"{category_name}_count"] = gdf_pppodata["geometry"].apply(
            lambda geo: utils.find_number_within_bounding_box_of_point(
                geo,
                gdf_pois_category,
                box_width_km,
                box_height_km
            ))
        gdf_pppodata[f"{category_name}_closest_degree_distance"] = gdf_pppodata["geometry"].apply(
            lambda geo: utils.find_degree_distance_to_closest_within_bounding_box_of_point(
                geo,
                gdf_pois_category,
                box_width_km,
                box_height_km
            ))
    feature_columns_names = [f"{category_name}_count" for category_name in dict_of_categories] + [
        f"{category_name}_closest_degree_distance" for category_name in dict_of_categories
    ]

    prediction_point_geometry = shapely.geometry.Point(longitude, latitude)
    prediction_point_features_dict = {}
    for category_name, gdf_pois_category in tqdm(dict_of_gdf_pois_categories.items()):
        prediction_point_features_dict[f"{category_name}_count"] = utils.find_number_within_bounding_box_of_point(
            prediction_point_geometry,
            gdf_pois_category,
            box_width_km,
            box_height_km
        )
        prediction_point_features_dict[f"{category_name}_closest_degree_distance"] = utils.find_degree_distance_to_closest_within_bounding_box_of_point(
            prediction_point_geometry,
            gdf_pois_category,
            box_width_km,
            box_height_km
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

    print()
    print("########################")
    print("# MODEL AND PREDICTION #")
    print("########################")
    # OLS
    train_Y = gdf_pppodata.sort_values("price", ascending=True)["price"]
    train_X = gdf_pppodata.sort_values("price", ascending=True)[
        feature_columns_names]
    train_X.insert(0, "const", 1.0)  # add ones column

    print("\nTraining model (non-regularised)...")
    model = sm.OLS(train_Y, train_X)
    results = model.fit()
    print()
    print(results.summary())
    Y_pred_of_train_X = results.get_prediction(
        train_X
    ).summary_frame(alpha=0.05)
    Y_pred_of_train_X_mse = utils.mse(Y_pred_of_train_X["mean"], train_Y)

    print("\nTraining model (regularised)...")
    model_regularised = sm.OLS(train_Y, train_X)
    results_regularised = model.fit_regularized(alpha=0.10, L1_wt=1.0)
    # alpha is the significance level. Confidence interval is 1 - alpha.
    Y_pred_of_train_X_regularised = results_regularised.predict(train_X)
    Y_pred_of_train_X_regularised_mse = utils.mse(
        Y_pred_of_train_X_regularised, train_Y)
    print()
    print("Regularised model coefs:")
    for i, (feature_name, beta) in enumerate(zip(train_X.columns, results_regularised.params)):
        print(f"β_{feature_name}={beta}")

    print("\nPlotting model results... (NB: x axis has no meaning)")
    fig, ax = plt.subplots(figsize=(constants.PLT_WIDTH, constants.PLT_HEIGHT))
    ax.scatter(range(len(train_X)), train_Y, zorder=2)
    # Non-regularised plot
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['mean'], color='red', linestyle='--', zorder=1)
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['obs_ci_lower'], color='red', linestyle='-', zorder=1)
    ax.plot(range(len(train_X)),
            Y_pred_of_train_X['obs_ci_upper'], color='red', linestyle='-', zorder=1)
    ax.fill_between(range(len(train_X)), Y_pred_of_train_X['obs_ci_lower'],
                    Y_pred_of_train_X['obs_ci_upper'], color='red', alpha=0.3, zorder=1)
    # Regularised plot
    ax.plot(range(len(train_X)), Y_pred_of_train_X_regularised,
            color="black", linestyle='--', zorder=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    Y_pred_of_prediction_point = results.get_prediction(
        prediction_point_features).summary_frame(alpha=0.05)
    print()
    print(
        f"Prediction (non-regularised model; MSE={Y_pred_of_train_X_mse}):\n{Y_pred_of_prediction_point}")
    print()
    Y_pred_of_prediction_point_regularised = results_regularised.predict(
        prediction_point_features)
    print(
        f"Prediction (regularised model; MSE={Y_pred_of_train_X_regularised_mse}):\n{Y_pred_of_prediction_point_regularised}")
    print()

    final_prediction = Y_pred_of_prediction_point["mean"][0]
    final_prediction_obs_ci_lower = Y_pred_of_prediction_point["obs_ci_lower"][0]
    final_prediction_obs_ci_upper = Y_pred_of_prediction_point["obs_ci_upper"][0]
    final_prediction_regularised = Y_pred_of_prediction_point_regularised[0]
    fig, ax = plt.subplots(
        figsize=(constants.PLT_WIDTH_LARGE, constants.PLT_HEIGHT))
    hist_values, _, _ = ax.hist(
        gdf_pppodata["price"],
        # between 10 to 1000 bins
        bins=np.clip(len(gdf_pppodata["date_of_transfer_dt"]) // 10, 10, 1000)
    )
    ax.plot(
        [final_prediction, final_prediction],
        [0, hist_values.max()],
        color="red",
        linestyle="--",
        label="predicted price (non-regularised model)"
    )
    ax.plot(
        [final_prediction_obs_ci_lower, final_prediction_obs_ci_lower],
        [0, hist_values.max()],
        color="red",
        linestyle="-",
    )
    ax.plot(
        [final_prediction_obs_ci_upper, final_prediction_obs_ci_upper],
        [0, hist_values.max()],
        color="red",
        linestyle="-",
    )
    ax.fill_betweenx([0, hist_values.max()], final_prediction_obs_ci_lower,
                     final_prediction_obs_ci_upper, color='red', alpha=0.3)
    ax.plot(
        [final_prediction_regularised, final_prediction_regularised],
        [0, hist_values.max()],
        color="black",
        linestyle="--",
        label="predicted price (regularised model)"
    )
    ax.legend()
    ax.set_xlabel("price")
    ax.set_ylabel("count")
    plt.show()

    if final_prediction_obs_ci_lower < 0:
        print("WARNING: Lower end of confidence interval lies below 0 pounds!")
    # prefer prediction from regularised model if MSE of regularised model is smaller than the non-regularised model
    if Y_pred_of_train_X_regularised_mse < Y_pred_of_train_X_mse:
        print("Outputting prediction of regularised model (black dotted line)")
        output = final_prediction_regularised
    else:
        print("Outputting prediction of non-regularised model (red dotted line)")
        output = final_prediction
    if output < gdf_pppodata["price"].min() or output > gdf_pppodata["price"].max():
        print("WARNING: The predicted value lies outside of the training dataset's range!")
    return output
