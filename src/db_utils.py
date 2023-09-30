import os
from enum import Enum

import greenplumpython as gp
import pandas


class TABLE_NAME(Enum):
    model_weights = "model_weights"
    model_results = "model_results"


def get_db_credentials_from_env():
    params = dict(
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DBNAME'],
        host=os.environ['POSTGRES_DBHOST'],
        port=os.environ['POSTGRES_DBPORT'],
    )
    return params


def get_db_credentials():
    return get_db_credentials_from_env()


def read_db_table(db: gp.Database, table_name: TABLE_NAME):
    return db.create_dataframe(table_name=table_name.value)


def write_results(db: gp.Database, dataf: pandas.DataFrame):
    columns = dataf.columns.tolist()
    cols = ", ".join(columns)
    records = dataf.to_records(index=False).tolist()
    # print("RECS:", records)
    vals = ", ".join(list(map(str, records[0])))

    # Prepare the SQL statement
    sql = f"INSERT INTO {TABLE_NAME.model_results.value} ({cols}) VALUES ({vals})"

    # Execute the SQL statement to insert the data
    cur = db._conn.cursor()
    cur.execute(sql)
    db._conn.commit()

    # Close the cursor and connection
    cur.close()
