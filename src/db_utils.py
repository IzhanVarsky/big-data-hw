import greenplumpython as gp
import pandas


def write_results(db: gp.Database, dataf: pandas.DataFrame):
    columns = dataf.columns.tolist()
    cols = ", ".join(columns)
    records = dataf.to_records(index=False).tolist()
    # print("RECS:", records)
    vals = ", ".join(list(map(str, records[0])))

    # Prepare the SQL statement
    sql = f"INSERT INTO model_results ({cols}) VALUES ({vals})"

    # Execute the SQL statement to insert the data
    cur = db._conn.cursor()
    cur.execute(sql)
    db._conn.commit()

    # Close the cursor and connection
    cur.close()
