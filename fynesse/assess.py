import os
from collections import defaultdict
from datetime import datetime

# import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

from . import access, config, constants, utils

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


#################################################################
# Characterising ppdata and podata (before any form of cleanup) #
#################################################################
def characterise_raw_ppdata_and_raw_podata(
    first_year_inclusive=constants.FIRST_YEAR_INCLUSIVE,
    final_year_inclusive=constants.FINAL_YEAR_INCLUSIVE,
    svg_output_dir="pppodata_characterisation/",
):
    os.makedirs(svg_output_dir, exist_ok=True)

    print("podata: Gathering statistics...")
    df_podata = access.read_podata_to_df(fixed_version=False)

    count_podata_record = len(df_podata)
    # Are there null postcodes in podata?
    count_podata_postcode_null = df_podata["postcode"].isna().sum()
    # NA values don't have value, can't find latlong
    print(
        f"Dropped {count_podata_postcode_null} null postcode rows from df_podata out of {count_podata_record}"
    )
    df_podata.dropna(subset=["postcode"], inplace=True)

    # ================================================================================= #
    # Aggregate statistics for ppdata from first_year_inclusive to final_year_inclusive #
    # ================================================================================= #
    count_ppdata_record = 0
    count_ppdata_postcode_null = 0
    # if postcode is null, that row won't count to wards the following statistics, as those rows
    # will not even be uploaded onto AWS after cleanup. We want to find out the statistics for the
    # rows that will be on AWS.
    count_ppdata_record_status_A = 0
    list_rows_record_status_not_A = []
    min_price = 1_000_000_000
    max_price = -1_000_000_000

    # We'll plot these following distributions later
    # ==================== #
    # ppdata distributions #
    # ==================== #
    count_property_type = defaultdict(lambda: 0)
    count_new_build_flag = defaultdict(lambda: 0)
    count_tenure_type = defaultdict(lambda: 0)
    count_ppd_category_type = defaultdict(lambda: 0)
    count_record_status = defaultdict(lambda: 0)
    # ==================== #
    # podata distributions #
    # ==================== #
    count_status = defaultdict(lambda: 0)
    count_usertype = defaultdict(lambda: 0)
    count_positional_quality_indicator = defaultdict(lambda: 0)
    # ============================================================== #
    # pppodata distributions (after table join of ppdata and podata) #
    # ============================================================== #
    list_of_df_price_date_latlongs_property_type = []

    for year in tqdm(range(first_year_inclusive, final_year_inclusive + 1)):
        for part in tqdm((1, 2), leave=False):
            # There may not be enough memory to load ALL csvs as dataframes in RAM at once
            # So we delete (del) each csv's dataframe once we're done extracting what we need
            tempdf_ppdata = access.read_ppdata_to_df(
                year, part, fixed_version=False
            )

            # --------------------------------------------------------------------------------- #
            # Aggregate statistics for ppdata from first_year_inclusive to final_year_inclusive #
            # --------------------------------------------------------------------------------- #
            count_ppdata_record += len(tempdf_ppdata)
            count_ppdata_postcode_null += tempdf_ppdata["postcode"].isna(
            ).sum()
            # NA postcodes are dropped so that table join can happen correctly
            print(
                f"[ppdata {year}part{part}] Dropped {tempdf_ppdata['postcode'].isna().sum()} null postcode rows out of {len(tempdf_ppdata)}"
            )
            tempdf_ppdata.dropna(subset=["postcode"], inplace=True)
            # inner join because we want all postcodes to have latlongs
            tempdf_pppodata = tempdf_ppdata.merge(
                df_podata, how="inner", on="postcode"
            )
            del tempdf_ppdata

            list_rows_record_status_not_A.append(
                tempdf_pppodata[tempdf_pppodata["record_status"] != "A"].copy()
            )
            count_ppdata_record_status_A += len(
                tempdf_pppodata[tempdf_pppodata["record_status"] == "A"]
            )
            min_price = min(min_price, tempdf_pppodata["price"].min())
            max_price = max(max_price, tempdf_pppodata["price"].max())

            # -------------------- #
            # ppdata distributions #
            # -------------------- #
            for _, property_type in tempdf_pppodata["property_type"].items():
                count_property_type[property_type] += 1
            for _, new_build_flag in tempdf_pppodata["new_build_flag"].items():
                count_new_build_flag[new_build_flag] += 1
            for _, tenure_type in tempdf_pppodata["tenure_type"].items():
                count_tenure_type[tenure_type] += 1
            for _, ppd_category_type in tempdf_pppodata["ppd_category_type"].items():
                count_ppd_category_type[ppd_category_type] += 1
            for _, record_status in tempdf_pppodata["record_status"].items():
                count_record_status[record_status] += 1

            # -------------------- #
            # podata distributions #
            # -------------------- #
            for _, status in tempdf_pppodata["status"].items():
                count_status[status] += 1
            for _, usertype in tempdf_pppodata["usertype"].items():
                count_usertype[usertype] += 1
            for _, positional_quality_indicator in tempdf_pppodata["positional_quality_indicator"].items():
                count_positional_quality_indicator[positional_quality_indicator] += 1

            # -------------------------------------------------------------- #
            # pppodata distributions (after table join of ppdata and podata) #
            # -------------------------------------------------------------- #
            list_of_df_price_date_latlongs_property_type.append(
                tempdf_pppodata.loc[
                    :,
                    [
                        "price",
                        "date_of_transfer",
                        "latitude",
                        "longitude",
                        "property_type"
                    ]
                ].copy()
            )
            del tempdf_pppodata

    print("\nStatistics collection complete.\n")
    print("###########")
    print("# RESULTS #")
    print("###########")
    print()
    # ================================================================================= #
    # Aggregate statistics for ppdata from first_year_inclusive to final_year_inclusive #
    # ================================================================================= #
    print(
        f"Number of rows in podata: {count_podata_record}")
    print(
        f"Number of rows in podata whose postcode is null: {count_podata_postcode_null}"
    )
    print(
        f"Number of rows in ppdata: {count_ppdata_record}")
    print(
        f"Number of rows in ppdata whose postcode is null: {count_ppdata_postcode_null}"
    )
    # if postcode is null, that row won't count to wards the following statistics, as those rows
    # will not even be uploaded onto AWS after cleanup. We want to find out the statistics for the
    # rows that will be on AWS.
    rows_record_status_not_A = pd.concat(
        list_rows_record_status_not_A, copy=False
    )
    print(
        f"Number of rows in ppdata whose record_status isn't A: {len(rows_record_status_not_A)}"
    )
    print(
        f"Number of rows in ppdata whose record_status is A: {count_ppdata_record_status_A}"
    )
    print(
        f"Minimum price in ppdata over all rows: {min_price}")
    print(
        f"Maximum price in ppdata over all rows: {max_price}")

    # We plot these following distributions now
    print("\nppdata_and_podata_distributions plot:\n")
    fig, axs = plt.subplots(
        4, 4,
        figsize=(4 * constants.PLT_WIDTH, 4 * constants.PLT_HEIGHT)
    )
    # ==================== #
    # ppdata distributions #
    # ==================== #
    axs[0][0].bar(
        count_property_type.keys(),
        utils.counts_to_probability(list(count_property_type.values()))
    )
    axs[0][0].set_xlabel("property_type")
    axs[0][0].set_ylabel("probability")

    axs[0][1].bar(
        count_property_type.keys(),
        utils.counts_to_probability(list(count_property_type.values())),
        color="red"
    )
    axs[0][1].set_xlabel("property_type")
    axs[0][1].set_ylabel("log probability")
    axs[0][1].set_yscale("log")

    axs[0][2].bar(
        count_new_build_flag.keys(),
        utils.counts_to_probability(list(count_new_build_flag.values()))
    )
    axs[0][2].set_xlabel("new_build_flag")
    axs[0][2].set_ylabel("probability")

    axs[0][3].bar(
        count_new_build_flag.keys(),
        utils.counts_to_probability(list(count_new_build_flag.values())),
        color="red"
    )
    axs[0][3].set_xlabel("new_build_flag")
    axs[0][3].set_ylabel("log probability")
    axs[0][3].set_yscale("log")

    axs[1][0].bar(
        count_tenure_type.keys(),
        utils.counts_to_probability(list(count_tenure_type.values()))
    )
    axs[1][0].set_xlabel("tenure_type")
    axs[1][0].set_ylabel("probability")

    axs[1][1].bar(
        count_tenure_type.keys(),
        utils.counts_to_probability(list(count_tenure_type.values())),
        color="red"
    )
    axs[1][1].set_xlabel("tenure_type")
    axs[1][1].set_ylabel("log probability")
    axs[1][1].set_yscale("log")

    axs[1][2].bar(
        count_ppd_category_type.keys(),
        utils.counts_to_probability(list(count_ppd_category_type.values()))
    )
    axs[1][2].set_xlabel("ppd_category_type")
    axs[1][2].set_ylabel("probability")

    axs[1][3].bar(
        count_ppd_category_type.keys(),
        utils.counts_to_probability(list(count_ppd_category_type.values())),
        color="red"
    )
    axs[1][3].set_xlabel("ppd_category_type")
    axs[1][3].set_ylabel("log probability")
    axs[1][3].set_yscale("log")

    axs[2][0].bar(
        count_record_status.keys(),
        utils.counts_to_probability(list(count_record_status.values()))
    )
    axs[2][0].set_xlabel("record_status")
    axs[2][0].set_ylabel("probability")

    axs[2][1].bar(
        count_record_status.keys(),
        utils.counts_to_probability(list(count_record_status.values())),
        color="red"
    )
    axs[2][1].set_xlabel("record_status")
    axs[2][1].set_ylabel("log probability")
    axs[2][1].set_yscale("log")
    # ==================== #
    # podata distributions #
    # ==================== #
    axs[2][2].bar(
        count_status.keys(),
        utils.counts_to_probability(list(count_status.values()))
    )
    axs[2][2].set_xlabel("status")
    axs[2][2].set_ylabel("probability")

    axs[2][3].bar(
        count_status.keys(),
        utils.counts_to_probability(list(count_status.values())),
        color="red"
    )
    axs[2][3].set_xlabel("status")
    axs[2][3].set_ylabel("log probability")
    axs[2][3].set_yscale("log")

    axs[3][0].bar(
        count_usertype.keys(),
        utils.counts_to_probability(list(count_usertype.values()))
    )
    axs[3][0].set_xlabel("usertype")
    axs[3][0].set_ylabel("probability")

    axs[3][1].bar(
        count_usertype.keys(),
        utils.counts_to_probability(list(count_usertype.values())),
        color="red"
    )
    axs[3][1].set_xlabel("usertype")
    axs[3][1].set_ylabel("log probability")
    axs[3][1].set_yscale("log")

    axs[3][2].bar(
        count_positional_quality_indicator.keys(),
        utils.counts_to_probability(
            list(count_positional_quality_indicator.values()))
    )
    axs[3][2].set_xlabel("positional_quality_indicator")
    axs[3][2].set_ylabel("probability")

    axs[3][3].bar(
        count_positional_quality_indicator.keys(),
        utils.counts_to_probability(
            list(count_positional_quality_indicator.values())),
        color="red"
    )
    axs[3][3].set_xlabel("positional_quality_indicator")
    axs[3][3].set_ylabel("log probability")
    axs[3][3].set_yscale("log")

    plt.savefig(
        os.path.join(svg_output_dir, "ppdata_and_podata_distributions.svg")
    )
    plt.show()
    print()
    # ============================================================== #
    # pppodata distributions (after table join of ppdata and podata) #
    # ============================================================== #
    df_price_date_latlongs_property_type = pd.concat(
        list_of_df_price_date_latlongs_property_type)
    # allow GC to collect the fragments
    del list_of_df_price_date_latlongs_property_type
    df_price_date_latlongs_property_type["date_of_transfer_dt"] = df_price_date_latlongs_property_type["date_of_transfer"].apply(
        datetime.fromisoformat
    )

    print("\npppodata_distributions plot:\n")
    fig, axs = plt.subplots(
        2, 1,
        figsize=(1 * constants.PLT_WIDTH_LARGE, 2 * constants.PLT_HEIGHT)
    )

    axs[0].hist(
        df_price_date_latlongs_property_type["price"],
        # between 10 to 1000 bins
        bins=np.clip(
            len(df_price_date_latlongs_property_type["price"]) // 10, 10, 1000
        )
    )
    axs[0].set_xlabel("price")
    axs[0].set_ylabel("log count")
    axs[0].set_yscale("log")

    axs[1].hist(df_price_date_latlongs_property_type["date_of_transfer_dt"],
                bins=336)  # 336 months from 1995 to 2022
    axs[1].set_xlabel("date_of_transfer")
    axs[1].set_ylabel("count")

    plt.savefig(os.path.join(svg_output_dir, "pppodata_distributions.svg"))
    plt.show()
    print()

    # ================================================ #
    # ppdata price histograms (for each property type) #
    # ================================================ #
    print("\nprice plots for given property_type:\n")
    fig, axs = plt.subplots(
        2, len(count_property_type),
        figsize=(
            len(count_property_type) *
            constants.PLT_WIDTH, 2 * constants.PLT_HEIGHT
        )
    )

    for i, property_type in enumerate(count_property_type):
        prices = df_price_date_latlongs_property_type[
            df_price_date_latlongs_property_type["property_type"] == property_type
        ]["price"]
        # between 10 to 100 bins
        axs[0][i].hist(
            prices,
            bins=np.clip(len(prices) // 10, 10, 100),
            density=True
        )
        axs[0][i].set_xlabel(f"price (where property={property_type})")
        axs[0][i].set_xscale("log")
        axs[0][i].set_ylabel("probability")

        axs[1][i].hist(
            prices,
            bins=np.clip(len(prices) // 10, 10, 100),
            density=True,
            color="red"
        )
        axs[1][i].set_xlabel(f"price (where property={property_type})")
        axs[1][i].set_xscale("log")
        axs[1][i].set_ylabel("log probability")
        axs[1][i].set_yscale("log")
    plt.savefig(
        os.path.join(
            svg_output_dir,
            "ppdata_price_plots_for_given_property_type.svg"
        )
    )
    plt.show()
    print()

    # Getting outline of UK (not needed; we're not doing a GDF plot because that may cause OOM)
    # gdf_world_outline = gpd.read_file(
    #     gpd.datasets.get_path('naturalearth_lowres'))
    # gdf_world_outline.crs = "EPSG:4326"
    # gdf_UK_outline = gdf_world_outline[(
    #     gdf_world_outline['name'] == 'United Kingdom')]
    # del gdf_world_outline

    # fig, ax = plt.subplots(figsize=(10, 10))
    # gdf_UK_outline.plot(ax=ax, color='white', edgecolor='black')
    # ax.set_xlabel('longitude')
    # ax.set_ylabel('latitude')
    # plt.show()

    print("\npppodata_latlong_distribution:\n")
    fig, ax = plt.subplots(
        figsize=(constants.PLT_MAP_WIDTH, constants.PLT_MAP_HEIGHT)
    )

    _, _, _, im = ax.hist2d(
        df_price_date_latlongs_property_type["longitude"].astype(float),
        df_price_date_latlongs_property_type["latitude"].astype(float),
        bins=np.clip(
            len(df_price_date_latlongs_property_type) // 10,
            100,
            500
        ),
        norm=LogNorm()
    )
    fig.colorbar(im, ax=ax)
    plt.savefig(
        os.path.join(svg_output_dir, "pppodata_latlong_distribution.svg")
    )
    plt.show()
    print()


############################################################
# Characterising ppdata and podata (result from AWS query) #
############################################################
def characterise_aws_pppodata(
    prediction_uuid,
    df_pppodata_from_aws,  # date column is a datatime already
    svg_output_dir="prediction_svgs/",
    print_banners=True
):
    """
    This function is coupled to the schema of the pp_data and postcode_data tables in the database.
    No way to get around this.
    """

    # =============================== #
    # Aggregate statistics for ppdata #
    # =============================== #
    count_record = len(df_pppodata_from_aws)
    min_price = df_pppodata_from_aws["price"].min()
    max_price = df_pppodata_from_aws["price"].max()
    # We'll plot these following distributions later
    # ==================== #
    # ppdata distributions #
    # ==================== #
    count_property_type = dict(
        df_pppodata_from_aws["property_type"].value_counts()
    )
    count_new_build_flag = dict(
        df_pppodata_from_aws["new_build_flag"].value_counts()
    )
    count_tenure_type = dict(
        df_pppodata_from_aws["tenure_type"].value_counts()
    )
    count_ppd_category_type = dict(
        df_pppodata_from_aws["ppd_category_type"].value_counts()
    )
    count_record_status = dict(
        df_pppodata_from_aws["record_status"].value_counts()
    )
    # ==================== #
    # podata distributions #
    # ==================== #
    count_status = dict(df_pppodata_from_aws["status"].value_counts())
    count_usertype = dict(df_pppodata_from_aws["usertype"].value_counts())
    count_positional_quality_indicator = dict(
        df_pppodata_from_aws["positional_quality_indicator"].value_counts()
    )
    # ============================================================== #
    # pppodata distributions (after table join of ppdata and podata) #
    # ============================================================== #
    # Nothing to do here

    if print_banners:
        print("################################")
        print("# CHARACTERISATION OF AWS ROWS #")
        print("################################")
        print()
    # ================================================================================= #
    # Aggregate statistics for ppdata from first_year_inclusive to final_year_inclusive #
    # ================================================================================= #
    print(f"Number of rows: {count_record}")
    print(f"Minimum price over all rows: {min_price}")
    print(f"Maximum price over all rows: {max_price}")

    # We plot these following distributions now
    print("\nppdata_and_podata_distributions plot:\n")
    fig, axs = plt.subplots(
        4, 4,
        figsize=(4 * constants.PLT_WIDTH, 4 * constants.PLT_HEIGHT)
    )
    # ==================== #
    # ppdata distributions #
    # ==================== #
    axs[0][0].bar(
        count_property_type.keys(),
        utils.counts_to_probability(list(count_property_type.values()))
    )
    axs[0][0].set_xlabel("property_type")
    axs[0][0].set_ylabel("probability")

    axs[0][1].bar(
        count_property_type.keys(),
        utils.counts_to_probability(list(count_property_type.values())),
        color="red"
    )
    axs[0][1].set_xlabel("property_type")
    axs[0][1].set_ylabel("log probability")
    axs[0][1].set_yscale("log")

    axs[0][2].bar(
        count_new_build_flag.keys(),
        utils.counts_to_probability(list(count_new_build_flag.values()))
    )
    axs[0][2].set_xlabel("new_build_flag")
    axs[0][2].set_ylabel("probability")

    axs[0][3].bar(
        count_new_build_flag.keys(),
        utils.counts_to_probability(list(count_new_build_flag.values())),
        color="red"
    )
    axs[0][3].set_xlabel("new_build_flag")
    axs[0][3].set_ylabel("log probability")
    axs[0][3].set_yscale("log")

    axs[1][0].bar(
        count_tenure_type.keys(),
        utils.counts_to_probability(list(count_tenure_type.values()))
    )
    axs[1][0].set_xlabel("tenure_type")
    axs[1][0].set_ylabel("probability")

    axs[1][1].bar(
        count_tenure_type.keys(),
        utils.counts_to_probability(list(count_tenure_type.values())),
        color="red"
    )
    axs[1][1].set_xlabel("tenure_type")
    axs[1][1].set_ylabel("log probability")
    axs[1][1].set_yscale("log")

    axs[1][2].bar(
        count_ppd_category_type.keys(),
        utils.counts_to_probability(list(count_ppd_category_type.values()))
    )
    axs[1][2].set_xlabel("ppd_category_type")
    axs[1][2].set_ylabel("probability")

    axs[1][3].bar(
        count_ppd_category_type.keys(),
        utils.counts_to_probability(list(count_ppd_category_type.values())),
        color="red"
    )
    axs[1][3].set_xlabel("ppd_category_type")
    axs[1][3].set_ylabel("log probability")
    axs[1][3].set_yscale("log")

    axs[2][0].bar(
        count_record_status.keys(),
        utils.counts_to_probability(list(count_record_status.values()))
    )
    axs[2][0].set_xlabel("record_status")
    axs[2][0].set_ylabel("probability")

    axs[2][1].bar(
        count_record_status.keys(),
        utils.counts_to_probability(list(count_record_status.values())),
        color="red"
    )
    axs[2][1].set_xlabel("record_status")
    axs[2][1].set_ylabel("log probability")
    axs[2][1].set_yscale("log")
    # ==================== #
    # podata distributions #
    # ==================== #
    axs[2][2].bar(
        count_status.keys(),
        utils.counts_to_probability(list(count_status.values()))
    )
    axs[2][2].set_xlabel("status")
    axs[2][2].set_ylabel("probability")

    axs[2][3].bar(
        count_status.keys(),
        utils.counts_to_probability(list(count_status.values())),
        color="red"
    )
    axs[2][3].set_xlabel("status")
    axs[2][3].set_ylabel("log probability")
    axs[2][3].set_yscale("log")

    axs[3][0].bar(
        count_usertype.keys(),
        utils.counts_to_probability(list(count_usertype.values()))
    )
    axs[3][0].set_xlabel("usertype")
    axs[3][0].set_ylabel("probability")

    axs[3][1].bar(
        count_usertype.keys(),
        utils.counts_to_probability(list(count_usertype.values())),
        color="red"
    )
    axs[3][1].set_xlabel("usertype")
    axs[3][1].set_ylabel("log probability")
    axs[3][1].set_yscale("log")

    axs[3][2].bar(
        count_positional_quality_indicator.keys(),
        utils.counts_to_probability(
            list(count_positional_quality_indicator.values()))
    )
    axs[3][2].set_xlabel("positional_quality_indicator")
    axs[3][2].set_ylabel("probability")

    axs[3][3].bar(
        count_positional_quality_indicator.keys(),
        utils.counts_to_probability(
            list(count_positional_quality_indicator.values())),
        color="red"
    )
    axs[3][3].set_xlabel("positional_quality_indicator")
    axs[3][3].set_ylabel("log probability")
    axs[3][3].set_yscale("log")

    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_ppdata_and_podata_distributions.svg"
        )
    )
    plt.show()
    print()
    # ============================================================== #
    # pppodata distributions (after table join of ppdata and podata) #
    # ============================================================== #
    df_pppodata_from_aws["date_of_transfer_dt"] = df_pppodata_from_aws["date_of_transfer"]
    # .apply(datetime.fromisoformat)  # no need, because date_of_transfer is already a column of datetimes

    print("\npppodata_distributions plot:\n")
    fig, axs = plt.subplots(
        2, 1,
        figsize=(1 * constants.PLT_WIDTH_LARGE, 2 * constants.PLT_HEIGHT)
    )

    axs[0].hist(
        df_pppodata_from_aws["price"],
        # between 10 to 1000 bins
        bins=np.clip(len(df_pppodata_from_aws["price"]) // 10, 10, 1000)
    )
    axs[0].set_xlabel("price")
    axs[0].set_ylabel("log count")
    axs[0].set_yscale("log")

    axs[1].hist(
        df_pppodata_from_aws["date_of_transfer_dt"],
        # between 10 to 1000 bins
        bins=np.clip(
            len(df_pppodata_from_aws["date_of_transfer_dt"]) // 10,
            10,
            1000
        )
    )
    axs[1].set_xlabel("date_of_transfer")
    axs[1].set_ylabel("count")

    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_pppodata_distributions.svg"
        )
    )
    plt.show()
    print()

    # ================================================ #
    # ppdata price histograms (for each property type) #
    # ================================================ #
    print("\nprice plots for given property_type:\n")
    fig, axs = plt.subplots(
        2, len(count_property_type),
        figsize=(
            len(count_property_type) *
            constants.PLT_WIDTH, 2 * constants.PLT_HEIGHT
        )
    )

    for i, property_type in enumerate(count_property_type):
        prices = df_pppodata_from_aws[
            df_pppodata_from_aws["property_type"] == property_type
        ]["price"]
        # between 10 to 100 bins
        axs[0][i].hist(
            prices,
            bins=np.clip(len(prices) // 10, 10, 100),
            density=True
        )
        axs[0][i].set_xlabel(f"price (where property={property_type})")
        axs[0][i].set_xscale("log")
        axs[0][i].set_ylabel("probability")

        axs[1][i].hist(
            prices,
            bins=np.clip(len(prices) // 10, 10, 100),
            density=True,
            color="red"
        )
        axs[1][i].set_xlabel(f"price (where property={property_type})")
        axs[1][i].set_xscale("log")
        axs[1][i].set_ylabel("log probability")
        axs[1][i].set_yscale("log")
    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_ppdata_price_plots_for_given_property_type.svg"
        )
    )
    plt.show()
    print()

    # Getting outline of UK (not needed; we're not doing a GDF plot because that may cause OOM)
    # gdf_world_outline = gpd.read_file(
    #     gpd.datasets.get_path('naturalearth_lowres'))
    # gdf_world_outline.crs = "EPSG:4326"
    # gdf_UK_outline = gdf_world_outline[(
    #     gdf_world_outline['name'] == 'United Kingdom')]
    # del gdf_world_outline

    # fig, ax = plt.subplots(figsize=(10, 10))
    # gdf_UK_outline.plot(ax=ax, color='white', edgecolor='black')
    # ax.set_xlabel('longitude')
    # ax.set_ylabel('latitude')
    # plt.show()

    print("\npppodata_latlong_distribution:\n")
    fig, ax = plt.subplots(
        figsize=(constants.PLT_MAP_WIDTH, constants.PLT_MAP_HEIGHT)
    )

    _, _, _, im = ax.hist2d(
        df_pppodata_from_aws["longitude"].astype(float),
        df_pppodata_from_aws["latitude"].astype(float),
        bins=100,
        norm=LogNorm()
    )
    fig.colorbar(im, ax=ax)
    plt.savefig(
        os.path.join(
            svg_output_dir,
            f"{prediction_uuid}_pppodata_latlong_distribution.svg"
        )
    )
    plt.show()
    print()
