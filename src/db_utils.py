import os
from enum import Enum
from typing import Dict, List

import greenplumpython as gp
import pandas

from ansible_credential_utils import read_credentials_from_file

from logger import Logger

logger = Logger(show=True).get_logger(__name__)

ANSIBLE_DB_CREDENTIALS_FILEPATH = 'db.credentials'


class TABLE_NAME(Enum):
    model_weights = "model_weights"
    model_results = "model_results"


def db_credential_to_dict(user, password, dbname, host, port) -> Dict:
    params = dict(
        user=user,
        password=password,
        dbname=dbname,
        host=host,
        port=port,
    )
    return params


def get_db_credentials_from_env() -> List[str]:
    return [os.environ['POSTGRES_USER'],
            os.environ['POSTGRES_PASSWORD'],
            os.environ['POSTGRES_DBNAME'],
            os.environ['POSTGRES_DBHOST'],
            os.environ['POSTGRES_DBPORT'],
            ]


def get_db_credentials_from_vault(ansible_password) -> List[str]:
    data = read_credentials_from_file(ANSIBLE_DB_CREDENTIALS_FILEPATH, ansible_password)
    return data.split()


def get_db_credentials(ansible_password=None) -> Dict:
    if ansible_password is None:
        logger.warning('Ansible password was not passed! '
                       'Trying to get DB credentials from ENV')
        credentials = get_db_credentials_from_env()
    else:
        logger.info('Using ansible to get DB credentials')
        credentials = get_db_credentials_from_vault(ansible_password)
    return db_credential_to_dict(*credentials)


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
