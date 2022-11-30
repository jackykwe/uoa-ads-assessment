# Low level methods, should not be called directly by users

import pandas as pd
import pymysql
import yaml
from tqdm.auto import tqdm

from . import constants

database_details = {"url": "database-jwek2.cgrre17yxw11.eu-west-2.rds.amazonaws.com",
                    "port": 3306}
with open("credentials.yaml") as file:
    credentials = yaml.safe_load(file)
USERNAME = credentials["username"]
PASSWORD = credentials["password"]
URL = database_details["url"]


def create_connection(
    user=USERNAME,
    password=PASSWORD,
    host=URL,
    database=None,
    port=3306
):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def create_database(
    conn,
    new_database_name,
    drop_before_create=False
):
    """
    Returns True if database is created, False if database already exists
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
            """
        )
        cur.execute(
            """
            SET time_zone = "+00:00";
            """
        )
        if drop_before_create:
            cur.execute(
                f"""
                DROP DATABASE IF EXISTS `{new_database_name}`;
                """
            )
        database_created = cur.execute(
            f"""
            CREATE DATABASE IF NOT EXISTS `{new_database_name}` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
            """
        )  # returns 1 if created, 0 if already exists
        conn.commit()

    return bool(database_created)


def create_new_table_if_not_exists(
    conn,
    new_table_name,
    col_name_defn_list,
    new_table_default_charset="utf8",
    new_table_collate="utf8_bin",
    new_table_auto_increment=1,
    drop_before_create=False,
):
    create_table_body = ", ".join(
        [f"`{col_name}` {defn}" for (col_name, defn) in col_name_defn_list])

    with conn.cursor() as cur:
        if drop_before_create:
            cur.execute(
                f"""
                DROP TABLE IF EXISTS `{new_table_name}`;
                """
            )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{new_table_name}` (
                {create_table_body}
            )
            DEFAULT CHARSET={new_table_default_charset}
            COLLATE={new_table_collate}
            AUTO_INCREMENT={new_table_auto_increment};
            """
        )
        conn.commit()


def add_table_btree_index(
    conn,
    new_index_name,
    table_name,
    column_name
):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX `{new_index_name}` USING BTREE ON `{table_name}` ({column_name});
            """
        )
        conn.commit()


def add_table_hash_index(
    conn,
    new_index_name,
    table_name,
    column_name
):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX `{new_index_name}` USING HASH ON `{table_name}` ({column_name});
            """
        )
        conn.commit()


def upload_data_into_table(
    conn,
    table_name,
    csv_pathname_with_ext,
    fields_enclosed_by_char='"',  # single character or empty string
):
    """
    WARNING: This operation is not idempotent.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
                LOAD DATA
                LOCAL INFILE '{csv_pathname_with_ext}'
                INTO TABLE `{table_name}`
                FIELDS TERMINATED BY ',' ENCLOSED BY '{fields_enclosed_by_char}'
                LINES STARTING BY '' TERMINATED BY '\\n';
            """
        )
        conn.commit()


def get_number_of_rows_from_table(conn, table_name):
    """
    WARNING: This operation may use up your AWS burst quota. Use in moderation.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
                SELECT count(*) FROM `{table_name}`;
            """
        )
        return cur.fetchone()[0]  # cur.fetchone returns a 1-tuple


def get_pppodata_conditioned(
    property_type,
    sql_cutoff_earliest_date,
    sql_cutoff_latest_date,
    north,
    south,
    east,
    west
):
    """
    Only rows that satisfy all the following conditions are fetched:
    - rows whose property_type (column name) matches property_type (argument to this function)
    - rows where sql_cutoff_earliest_date <= date_of_transfer (column name) <= sql_cutoff_latest_date
    - rows where south <= latitude (column name) <= north
    - rows where west <= longitude (column name) <= east

    :param property_type: a string, one of "D/S/T/F/O".
    :param sql_cutoff_earliest_date: a string in ISO format.
    :param sql_cutoff_latest_date: a string in ISO format.
    :param north: an int or float.
    :param south: an int or float.
    :param east: an int or float.
    :param west: an int or float.
    """
    with create_connection(database=constants.DATABASE_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT price, date_of_transfer, pp_data.postcode, property_type, new_build_flag, tenure_type, ppd_category_type, usertype, positional_quality_indicator, latitude, longitude FROM pp_data
                INNER JOIN postcode_data ON pp_data.postcode = postcode_data.postcode
                WHERE
                    pp_data.property_type = '{property_type}'
                    AND pp_data.date_of_transfer >= '{sql_cutoff_earliest_date}'
                    AND pp_data.date_of_transfer <= '{sql_cutoff_latest_date}'
                    AND postcode_data.longitude >= {west}
                    AND postcode_data.longitude <= {east}
                    AND postcode_data.latitude >= {south}
                    AND postcode_data.latitude <= {north};
                """
            )
            aws_query_result = cur.fetchall()

    return pd.DataFrame(
        aws_query_result,
        columns=[
            "price",
            "date_of_transfer",
            "postcode",
            "property_type",
            "new_build_flag",
            "tenure_type",
            "ppd_category_type",
            "usertype",
            "positional_quality_indicator",
            "latitude",
            "longitude"
        ]
    )
