This quiz covers the whole Data Engineering Fundamentals module: pipeline design and idempotency, ETL and ELT, ingestion and file formats, validation, preprocessing, API collection, batch processing, Airflow and Spark. Aim for at least 13 out of 16 before moving on to Machine Learning Fundamentals.

```quiz
[
  {
    "q": "A daily load step appends rows to a table. It fails after writing half the rows and is retried. What is the most likely result?",
    "options": [
      "The table is correct because retries are atomic",
      "Some rows for that day are duplicated",
      "The table is emptied",
      "The retry is skipped automatically"
    ],
    "answer": 1,
    "why": "Appending is not idempotent: rows written before the failure are written again. Overwrite the day's slice or upsert instead."
  },
  {
    "q": "Which three techniques make a load step idempotent?",
    "options": [
      "Compression, caching, indexing",
      "Appending, sampling, sorting",
      "Overwriting a partition, upserting by key, deduplicating on read",
      "Retries, timeouts, logging"
    ],
    "answer": 2,
    "why": "Each ensures that replaying the same input leaves the same final state."
  },
  {
    "q": "In an ELT pipeline, where does the transformation usually run?",
    "options": [
      "Inside the target warehouse, typically as SQL over raw tables",
      "In the source database before extraction",
      "In the browser",
      "Nowhere: ELT skips transformation"
    ],
    "answer": 0,
    "why": "ELT loads raw data first and transforms it inside the target, which keeps the raw data available for future changes."
  },
  {
    "q": "What does this SQLite statement do when a row with `id = 2` already exists: `INSERT INTO t (id, v) VALUES (2, 'x') ON CONFLICT (id) DO UPDATE SET v = excluded.v`?",
    "options": [
      "Raises an integrity error",
      "Inserts a second row with id 2",
      "Deletes the existing row",
      "Updates the existing row's `v` to `'x'`"
    ],
    "answer": 3,
    "why": "On a key conflict the upsert updates the existing row using the proposed values in `excluded`."
  },
  {
    "q": "An incremental job saves its new watermark before loading the batch, and the load fails. What happens next run?",
    "options": [
      "The failed batch is skipped permanently",
      "The batch is loaded twice",
      "The job performs a full load",
      "Nothing changes"
    ],
    "answer": 0,
    "why": "The next run only asks for rows beyond the saved watermark. Save it only after a successful load."
  },
  {
    "q": "`pd.read_csv` turns the product code `00451` into `451`. What is the best fix?",
    "options": [
      "Convert back with `str.zfill(5)` in every report",
      "Pass `dtype={\"product_code\": \"string\"}` when reading",
      "Store the file as Excel",
      "Use `parse_dates`"
    ],
    "answer": 1,
    "why": "CSV carries no types, so declare identifier columns as strings at read time."
  },
  {
    "q": "Which format is columnar, typed and compressed, and therefore usually best for analytical queries?",
    "options": [
      "CSV",
      "JSON Lines",
      "Parquet",
      "Plain text"
    ],
    "answer": 2,
    "why": "Parquet stores types and compresses by column, so readers fetch only the columns they need."
  },
  {
    "q": "A batch arrives without a required column. What should validation do?",
    "options": [
      "Quarantine every row and continue",
      "Fill the column with nulls",
      "Ignore it if the row count looks normal",
      "Stop the run and alert, because the batch is structurally broken"
    ],
    "answer": 3,
    "why": "Structural failures make the whole batch untrustworthy. Quarantine is for a minority of bad records."
  },
  {
    "q": "What is a data contract?",
    "options": [
      "An agreement in which a data producer promises schema, meaning, allowed values and freshness to consumers",
      "A licence for buying datasets",
      "A database transaction",
      "A type of Airflow operator"
    ],
    "answer": 0,
    "why": "The contract states expectations; validation code checks them automatically."
  },
  {
    "q": "Why split train and test sets by a hash of the user ID rather than by `df.sample(frac=0.8)`?",
    "options": [
      "Hashing is faster than sampling",
      "It guarantees exactly 80% training rows",
      "Each user always lands on the same side, so no user leaks across splits and assignments stay stable as data grows",
      "It removes duplicate rows"
    ],
    "answer": 2,
    "why": "A deterministic, entity-level split prevents leakage and keeps examples from moving between splits over time."
  },
  {
    "q": "An API returns 429 with `Retry-After: 10`. What should a well-behaved collector do?",
    "options": [
      "Retry immediately",
      "Wait about 10 seconds, then retry",
      "Treat it as a permanent failure",
      "Switch to a new API key"
    ],
    "answer": 1,
    "why": "429 means too many requests; the server is telling you how long to wait before trying again."
  },
  {
    "q": "Which statement about robots.txt is correct?",
    "options": [
      "If robots.txt allows a path, you may use its content for any purpose",
      "robots.txt only applies to search engines",
      "robots.txt replaces a site's terms of service",
      "You should not fetch disallowed paths, and allowed paths may still be restricted by terms of service, copyright or privacy law"
    ],
    "answer": 3,
    "why": "robots.txt is a minimum courtesy, not a grant of rights over the content."
  },
  {
    "q": "Chunk A has 10 rows with mean 2 and chunk B has 90 rows with mean 1. What is the overall mean?",
    "options": [
      "1.1",
      "1.5",
      "3.0",
      "2.0"
    ],
    "answer": 0,
    "why": "Weighted by row count: (10 * 2 + 90 * 1) / 100 = 1.1. Averaging the means gives the wrong answer, 1.5."
  },
  {
    "q": "In Airflow, why should tasks return file paths rather than large DataFrames?",
    "options": [
      "DataFrames cannot be pickled",
      "Return values are stored as XComs in the metadata database, which is meant for small values",
      "Paths run faster on executors",
      "Airflow only supports string return values"
    ],
    "answer": 1,
    "why": "Large XComs bloat the metadata database and slow the scheduler. Store data elsewhere and pass references."
  },
  {
    "q": "What does `catchup=True` cause when a daily DAG with a `start_date` two weeks ago is first unpaused?",
    "options": [
      "Only today's run is created",
      "The DAG runs once and stops",
      "Runs are created for every missed daily interval since the start date",
      "The start date is reset to today"
    ],
    "answer": 2,
    "why": "Catchup schedules all intervals between `start_date` and now that have not yet run."
  },
  {
    "q": "A PySpark script runs `a = df.filter(F.col(\"x\") > 0)`, then `b = a.groupBy(\"k\").count()`, then `c = b.withColumn(\"y\", F.lit(1))`, then `c.show()`. Which line triggers computation?",
    "options": [
      "The `filter` line",
      "The `groupBy` line",
      "The `withColumn` line",
      "The `show` line"
    ],
    "answer": 3,
    "why": "Transformations are lazy; `show` is an action, so it triggers execution of the whole plan."
  }
]
```
