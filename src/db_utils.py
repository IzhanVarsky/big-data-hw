import greenplumpython as gp
import pandas


def write_results(db: gp.Database, dataf: pandas.DataFrame):
    columns = dataf.columns.tolist()
    cols = ", ".join(columns)
    records = dataf.to_records(index=False).tolist()

    # Prepare the SQL statement
    sql = f"INSERT INTO model_results (%s) VALUES (%s)"

    # Execute the SQL statement to insert the data
    cur = db._conn.cursor()
    cur.executemany(sql, [columns, records])
    db._conn.commit()

    # Close the cursor and connection
    cur.close()
