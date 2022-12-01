import os

import pandas as pd
from tqdm.auto import tqdm

from . import access, aws_utils, constants

# These functions are coupled with the assessment task, as they provide sensible defaults for
# the underlying aws_utils functions. These functions thus serve as convenience functions for
# the assessment task. These can be easily adapted for other tasks apart from housing prediction.


def init_create_database_if_not_exists(drop_before_create=False):
    with aws_utils.create_connection(database=constants.DATABASE_NAME) as conn:
        created = aws_utils.create_database(
            conn,
            constants.DATABASE_NAME,
            drop_before_create=drop_before_create
        )
    if created:
        print(
            f"Database {constants.DATABASE_NAME}{' dropped and' if drop_before_create else ''} created.")
    else:
        print(
            f"Database {constants.DATABASE_NAME} already exists, not re-created.")


def init_setup_aws_ppdata(
    dir_name="data/ppdata",
    first_year_inclusive=constants.FIRST_YEAR_INCLUSIVE,
    final_year_inclusive=constants.FINAL_YEAR_INCLUSIVE
):
    """
    WARNING: Do NOT run more than once. This operation is NOT idempotent.
    """
    print("Downloading ppdata from gov...")
    access.download_ppdata_from_gov(
        dir_name=dir_name,
        first_year_inclusive=first_year_inclusive,
        final_year_inclusive=first_year_inclusive
    )

    # Data cleanup
    print("Fixing ppdata (removing invalid rows)...")
    access.fix_ppdata(
        dir_name=dir_name,
        first_year_inclusive=first_year_inclusive,
        final_year_inclusive=first_year_inclusive
    )

    total_rows_to_upload = 0
    with aws_utils.create_connection(database=constants.DATABASE_NAME) as conn:
        aws_utils.create_new_table_if_not_exists(
            conn,
            constants.PPDATA_TABLE_NAME,
            [
                ("transaction_unique_identifier",
                 "tinytext COLLATE utf8_bin NOT NULL"),
                ("price", "int(10) unsigned NOT NULL"),
                ("date_of_transfer", "date NOT NULL"),
                ("postcode", "varchar(8) COLLATE utf8_bin NOT NULL"),
                ("property_type", "varchar(1) COLLATE utf8_bin NOT NULL"),
                # NB this is modelled as text
                ("new_build_flag", "varchar(1) COLLATE utf8_bin NOT NULL"),
                ("tenure_type", "varchar(1) COLLATE utf8_bin NOT NULL"),
                ("primary_addressable_object_name",
                 "tinytext COLLATE utf8_bin NOT NULL"),
                ("secondary_addressable_object_name",
                 "tinytext COLLATE utf8_bin NOT NULL"),
                ("street", "tinytext COLLATE utf8_bin NOT NULL"),
                ("locality", "tinytext COLLATE utf8_bin NOT NULL"),
                ("town_city", "tinytext COLLATE utf8_bin NOT NULL"),
                ("district", "tinytext COLLATE utf8_bin NOT NULL"),
                ("county", "tinytext COLLATE utf8_bin NOT NULL"),
                ("ppd_category_type", "varchar(2) COLLATE utf8_bin NOT NULL"),
                ("record_status", "varchar(2) COLLATE utf8_bin NOT NULL"),
                # must define primary key here, not after uploading data
                ("db_id", "bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY"),
            ]
        )

        print("Uploading ppdata to AWS...")
        for year in tqdm(range(constants.FIRST_YEAR_INCLUSIVE, constants.FINAL_YEAR_INCLUSIVE + 1)):
            for part in tqdm((1, 2), leave=False):
                file_name_with_dir = os.path.join(
                    dir_name, f"pp-{year}-part{part}fixed.csv"
                )

                tempdf = access.read_ppdata_to_df(
                    year,
                    part,
                    dir_name=dir_name,
                    fixed_version=True
                )
                total_rows_to_upload += len(tempdf)
                del tempdf

                aws_utils.upload_data_into_table(
                    constants.PPDATA_TABLE_NAME,
                    file_name_with_dir
                )

        print("Adding ppdata indexes to AWS...")
        aws_utils.add_table_hash_index(
            conn, "pp.postcode", "pp_data", "postcode"
        )
        aws_utils.add_table_hash_index(
            conn, "pp.property_type", "pp_data", "property_type"
        )
        aws_utils.add_table_btree_index(
            conn, "pp.date", "pp_data", "date_of_transfer"
        )

        print("Successfully uploaded ppdata to AWS.")

        # Sanity checking
        rows_uploaded = aws_utils.aws_get_number_of_rows_in_ppdata()
        print(f"Successfully uploaded ppdata ({rows_uploaded} rows) to AWS.")
        assert rows_uploaded == total_rows_to_upload, f"Expected to upload {total_rows_to_upload} rows, uploaded only {rows_uploaded}"


def init_setup_aws_podata(dir_name="data/podata"):
    """
    WARNING: Do NOT run more than once. This operation is NOT idempotent.
    """
    print("Downloading podata from gtd...")
    access.download_podata_from_gtd(dir_name=dir_name)

    # Data cleanup
    print("Fixing podata (removing invalid rows)...")
    access.fix_podata(dir_name=dir_name)

    total_rows_to_upload = 0
    with aws_utils.create_connection(database=constants.DATABASE_NAME) as conn:
        aws_utils.create_new_table_if_not_exists(
            conn,
            constants.PODATA_TABLE_NAME,
            [
                ("postcode", "varchar(8) COLLATE utf8_bin NOT NULL"),
                ("status", "enum('live','terminated') NOT NULL"),
                ("usertype", "enum('small', 'large') NOT NULL"),
                ("easting", "int unsigned"),
                ("northing", "int unsigned"),
                ("positional_quality_indicator", "int NOT NULL"),
                ("country", "enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL"),
                ("latitude", "decimal(11,8) NOT NULL"),
                ("longitude", "decimal(10,8) NOT NULL"),
                ("postcode_no_space", "tinytext COLLATE utf8_bin NOT NULL"),
                ("postcode_fixed_width_seven",
                 "varchar(7) COLLATE utf8_bin NOT NULL"),
                ("postcode_fixed_width_eight",
                 "varchar(8) COLLATE utf8_bin NOT NULL"),
                ("postcode_area", "varchar(2) COLLATE utf8_bin NOT NULL"),
                ("postcode_district", "varchar(4) COLLATE utf8_bin NOT NULL"),
                ("postcode_sector", "varchar(6) COLLATE utf8_bin NOT NULL"),
                ("outcode", "varchar(4) COLLATE utf8_bin NOT NULL"),
                ("incode", "varchar(3)  COLLATE utf8_bin NOT NULL"),
                # must define primary key here, not after uploading data
                ("db_id", "bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY"),
            ]
        )

        print("Uploading podata to AWS...")
        for _ in tqdm((1, )):
            file_name_with_dir = os.path.join(
                dir_name, "open_postcode_geo.csv"
            )

            tempdf = access.read_podata_to_df(
                dir_name=dir_name,
                fixed_version=True
            )
            total_rows_to_upload += len(tempdf)
            del tempdf

        aws_utils.upload_data_into_table(
            constants.PODATA_TABLE_NAME,
            file_name_with_dir,
            fields_enclosed_by_char=""
        )

        print("Adding podata indexes to AWS...")
        aws_utils.add_table_hash_index(
            conn, "po.postcode", "postcode_data", "postcode"
        )
        aws_utils.add_table_btree_index(
            conn, "po.date", "postcode_data", "date_of_transfer"
        )
        aws_utils.add_table_btree_index(
            conn, "po.latitude", "postcode_data", "latitude"
        )
        aws_utils.add_table_btree_index(
            conn, "po.longitude", "postcode_data", "longitude"
        )

        # Sanity checking
        rows_uploaded = aws_utils.aws_get_number_of_rows_in_podata()
        print(f"Successfully uploaded ppdata ({rows_uploaded} rows) to AWS.")
        assert rows_uploaded == total_rows_to_upload, f"Expected to upload {total_rows_to_upload} rows, uploaded only {rows_uploaded}"
