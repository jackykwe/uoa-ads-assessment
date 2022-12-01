import csv
import os
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm.auto import tqdm

from . import aws_utils, config, constants

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

PPDATA_COLUMN_NAMES = [
    "transaction_unique_identifier",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "new_build_flag",
    "tenure_type",
    "primary_addressable_object_name",
    "secondary_addressable_object_name",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status"
]

PODATA_COLUMN_NAMES = [
    "postcode",
    "status",
    "usertype",
    "easting",
    "northing",
    "positional_quality_indicator",
    "country",
    "latitude",
    "longitude",
    "postcode_no_space",
    "postcode_fixed_width_seven",
    "postcode_fixed_width_eight",
    "postcode_area",
    "postcode_district",
    "postcode_sector",
    "outcode",
    "incode"
]


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


########################################################
# Downloading, cleanup of, and uploading data into AWS #
########################################################
def download_ppdata_from_gov(
    dir_name="data/ppdata",
    first_year_inclusive=constants.FIRST_YEAR_INCLUSIVE,
    final_year_inclusive=constants.FINAL_YEAR_INCLUSIVE,
    verbose=False
):
    os.makedirs(dir_name, exist_ok=True)
    for year in tqdm(range(first_year_inclusive, final_year_inclusive + 1)):
        for part in (1, 2):
            file_name = f"pp-{year}-part{part}.csv"
            file_name_with_dir = os.path.join(dir_name, file_name)
            if os.path.exists(file_name_with_dir):
                if verbose:
                    print(f"{file_name}: Already downloaded.")
            else:
                with open(file_name_with_dir, "wb") as f:
                    response = requests.get(
                        f"http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/{file_name}")
                    if response.status_code == 200:
                        f.write(response.content)
                    else:
                        raise Exception(
                            f"Request for file {file_name} failed with HTTP {response.status_code}")
                if verbose:
                    print(f"{file_name}: Downloaded.")


def fix_ppdata(
    dir_name="data/ppdata",
    first_year_inclusive=constants.FIRST_YEAR_INCLUSIVE,
    final_year_inclusive=constants.FINAL_YEAR_INCLUSIVE
):
    for year in tqdm(range(first_year_inclusive, final_year_inclusive + 1)):
        for part in tqdm((1, 2), leave=False):
            file_name = f"pp-{year}-part{part}.csv"
            file_name_with_dir = os.path.join(dir_name, file_name)
            df_tofix = pd.read_csv(
                file_name_with_dir,
                names=PPDATA_COLUMN_NAMES
            )

            # Dropping useless rows
            df_tofix = df_tofix.loc[
                df_tofix["postcode"].notna() &
                df_tofix["tenure_type"].isin(["F", "L"])
            ]

            fixed_file_name = f"pp-{year}-part{part}_fixed.csv"
            fixed_file_name_with_dir = os.path.join(dir_name, fixed_file_name)
            df_tofix.to_csv(fixed_file_name_with_dir, header=False,
                            index=False, quoting=csv.QUOTE_ALL)

            del df_tofix  # to prevent OOM; allow GC to collect the object


def read_ppdata_to_df(year, part, dir_name="data/ppdata", fixed_version=True):
    file_name = f"pp-{year}-part{part}{'_fixed' if fixed_version else ''}.csv"
    file_name_with_dir = os.path.join(dir_name, file_name)
    return pd.read_csv(
        file_name_with_dir,
        names=PPDATA_COLUMN_NAMES
    )


def download_podata_from_gtd(dir_name="data/podata", verbose=False):
    os.makedirs(dir_name, exist_ok=True)
    file_name = "open_postcode_geo.csv"
    file_name_with_dir = os.path.join(dir_name, file_name)

    zip_file_name = f"{file_name}.zip"
    zip_file_name_with_dir = os.path.join(dir_name, zip_file_name)

    if os.path.exists(file_name_with_dir):
        if verbose:
            print(f"{file_name}: Already downloaded.")
    else:
        with open(zip_file_name_with_dir, "wb") as f:
            response = requests.get(
                f"https://www.getthedata.com/downloads/{zip_file_name}")
            if response.status_code == 200:
                f.write(response.content)
            else:
                raise Exception(
                    f"Request for file {zip_file_name} failed with HTTP {response.status_code}")
        with ZipFile(zip_file_name_with_dir, "r") as f:
            f.extractall(dir_name)
        os.remove(zip_file_name_with_dir)
        if verbose:
            print(f"{file_name}: Downloaded and unzipped.")


def fix_podata(dir_name="data/podata"):
    file_name = "open_postcode_geo.csv"
    file_name_with_dir = os.path.join(dir_name, file_name)
    df_tofix = pd.read_csv(
        file_name_with_dir,
        names=PODATA_COLUMN_NAMES
    )

    # Dropping useless rows
    df_tofix = df_tofix.loc[
        df_tofix["postcode"].notna()
    ]

    fixed_file_name = "open_postcode_geo_fixed.csv"
    fixed_file_name_with_dir = os.path.join(
        dir_name,
        fixed_file_name
    )
    df_tofix.to_csv(fixed_file_name_with_dir, header=False,
                    index=False, quoting=csv.QUOTE_ALL)

    del df_tofix  # to prevent OOM; allow GC to collect the object


def read_podata_to_df(dir_name="data/podata", fixed_version=True):
    file_name = f"open_postcode_geo{'_fixed' if fixed_version else ''}.csv"
    file_name_with_dir = os.path.join(dir_name, file_name)
    return pd.read_csv(
        file_name_with_dir,
        names=PODATA_COLUMN_NAMES
    )


#####################
# Querying from AWS #
#####################
def aws_get_number_of_rows_in_ppdata():
    with aws_utils.create_connection(database=constants.DATABASE_NAME) as conn:
        return aws_utils.get_number_of_rows_from_table(conn, constants.PPDATA_TABLE_NAME)


def aws_get_number_of_rows_in_podata():
    with aws_utils.create_connection(database=constants.DATABASE_NAME) as conn:
        return aws_utils.get_number_of_rows_from_table(conn, constants.PPDATA_TABLE_NAME)
