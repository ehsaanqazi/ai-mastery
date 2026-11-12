SQLite is perfect for learning and for single-process tools, but most production AI applications store their data in **PostgreSQL** (often just "Postgres"). It is open source, runs as a server that many clients share, enforces types strictly, has a rich set of data types including `JSONB` for documents, and supports extensions, most notably **pgvector**, which turns it into a vector database for retrieval-augmented generation. Managed versions are offered by every major cloud provider. This lesson gets you working with Postgres: running it, using `psql`, creating the shop schema with proper types, managing roles, connecting from Python, storing JSON, and a first look at vector search.

Everything in the earlier lessons applies unchanged unless noted. The PostgreSQL examples need a running server, so unlike the rest of the module they are shown without output. The two Python snippets that do show output (a SQLite JSON query and a NumPy distance check) were run as written.

## Running PostgreSQL locally

The quickest route is a container (Docker is covered fully in [Docker and containers](/learn/docker-and-containers)). This starts PostgreSQL with the pgvector extension available, listening on the standard port 5432:

```bash
docker run --name pg -e POSTGRES_PASSWORD=devpassword -p 5432:5432 -d pgvector/pgvector:pg16
```

Alternatively, install it with your system package manager (`brew install postgresql@16` on macOS, `sudo apt install postgresql` on Debian and Ubuntu). Either way you get a **cluster**: one server process managing several databases.

```mermaid
flowchart TD
    S["PostgreSQL server (cluster)<br/>port 5432"] --> R["Roles<br/>shared by the whole cluster"]
    S --> D1["Database: shop"]
    S --> D2["Database: analytics"]
    D1 --> SC1["Schema: public"]
    D1 --> SC2["Schema: ml"]
    SC1 --> T1["Tables: customers, orders, ..."]
    SC2 --> T2["Tables: predictions, embeddings"]
```

A **schema** is a namespace inside a database. Tables go in `public` by default; larger systems use schemas to separate concerns, such as raw ingested data, cleaned data and model outputs, and refer to tables as `ml.predictions`.

## psql: the command-line client

`psql` is PostgreSQL's interactive terminal. Connect with:

```bash
psql -h localhost -U postgres
```

You can type any SQL statement, ending it with `;`. Commands starting with a backslash are **meta-commands** handled by psql itself:

| Command | Does |
| --- | --- |
| `\l` | List databases |
| `\c shop` | Connect to database `shop` |
| `\dt` | List tables in the current schema |
| `\d orders` | Describe a table: columns, types, indexes, constraints |
| `\du` | List roles |
| `\x` | Toggle expanded display (one column per line, good for wide rows) |
| `\timing` | Show how long each statement takes |
| `\i setup.sql` | Run SQL from a file |
| `\?` and `\h SELECT` | Help on meta-commands, and on an SQL statement |
| `\q` | Quit |

Create a database and connect to it:

```sql
CREATE DATABASE shop;
\c shop
```

## The shop schema, PostgreSQL style

Here is the schema rewritten with PostgreSQL's types:

```sql
CREATE TABLE customers (
    customer_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    country     CHAR(2) NOT NULL,
    signup_date DATE NOT NULL
);

CREATE TABLE products (
    product_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      NUMERIC(10, 2) NOT NULL CHECK (price >= 0)
);

CREATE TABLE tickets (
    ticket_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers (customer_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    subject         TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    true_label      TEXT,
    details         JSONB
);
```

`orders` follows the same pattern, with `order_date DATE` and an identity key. What changed compared with SQLite:

| Feature | SQLite | PostgreSQL |
| --- | --- | --- |
| Auto-generated keys | `INTEGER PRIMARY KEY` | `GENERATED ALWAYS AS IDENTITY`; older code uses `SERIAL` |
| Type checking | Affinity; `STRICT` tables opt in | Always strict: `'hello'` in an `INTEGER` column is an error |
| Money | `REAL` or integer cents | `NUMERIC(10, 2)`, exact |
| Dates and times | ISO 8601 text | `DATE`, `TIMESTAMPTZ` (timestamp with time zone) |
| Booleans | 0 and 1 | `BOOLEAN` with `TRUE` and `FALSE` |
| Case-insensitive match | `LIKE` already is, for ASCII | `ILIKE` |
| Casting | `CAST(x AS INTEGER)` | `CAST(x AS INTEGER)` or `x::integer` |
| Date arithmetic | `date(d, '+1 day')`, `julianday` | `d + 1`, `now() - interval '7 days'` |
| Foreign keys | Need `PRAGMA foreign_keys = ON` | Always enforced |

> [!TIP]
> Use `TIMESTAMPTZ` rather than `TIMESTAMP` for events. PostgreSQL stores it as an absolute instant and converts to the session's time zone for display, which avoids an entire class of daylight-saving and multi-region bugs.

## Roles and privileges

PostgreSQL controls access with **roles**. A role with the `LOGIN` attribute is effectively a user; a role without it acts as a group. You grant privileges on objects to roles, and grant group roles to users.

The principle is **least privilege**: each program gets exactly the access it needs. For an AI application this matters more than usual. A service that lets an LLM write SQL from a user's question should connect as a role that can only read the tables it needs, so that a bad generation or a prompt-injection attack cannot run `DELETE`.

```sql
-- Group roles: no login, just bundles of privileges
CREATE ROLE readonly NOLOGIN;
GRANT CONNECT ON DATABASE shop TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly;

CREATE ROLE app_writer NOLOGIN;
GRANT readonly TO app_writer;
GRANT INSERT, UPDATE ON tickets TO app_writer;

-- Login roles for actual programs and people
CREATE ROLE ticket_service LOGIN PASSWORD 'change-me';
GRANT app_writer TO ticket_service;

CREATE ROLE sql_agent LOGIN PASSWORD 'change-me-too';
GRANT readonly TO sql_agent;
```

```mermaid
flowchart LR
    TS["ticket_service<br/>LOGIN"] --> AW["app_writer<br/>INSERT, UPDATE on tickets"]
    AW --> RO["readonly<br/>SELECT on all tables"]
    SA["sql_agent<br/>LOGIN"] --> RO
    AN["analyst<br/>LOGIN"] --> RO
```

`GRANT SELECT ON ALL TABLES` covers only tables that exist now. `ALTER DEFAULT PRIVILEGES` makes future tables created in that schema readable too. Revoke with `REVOKE ... FROM role`, and check with `\du` and `\dp tablename` in psql.

## Connecting from Python

The standard PostgreSQL driver for Python is **psycopg** (version 3, installed with `pip install "psycopg[binary]"`; much existing code uses its predecessor, `psycopg2`). It follows the same DB-API shape as `sqlite3`, but its placeholder is `%s`:

```python
import os
import psycopg

dsn = os.environ["DATABASE_URL"]   # e.g. postgresql://ticket_service:...@localhost:5432/shop

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO tickets (customer_id, subject, predicted_label, confidence)
               VALUES (%s, %s, %s, %s)
               RETURNING ticket_id, created_at""",
            (3, "GPU credits not applied", "billing", 0.86),
        )
        ticket_id, created_at = cur.fetchone()

        cur.execute(
            "SELECT subject, confidence FROM tickets WHERE predicted_label = %s AND confidence < %s",
            ("billing", 0.9),
        )
        for subject, confidence in cur.fetchall():
            print(subject, confidence)
# leaving the 'with' block commits the transaction (or rolls back on an exception) and closes
```

Three habits to carry over: read credentials from the environment, never from source code; always pass values as parameters (`%s` is a psycopg placeholder, **not** Python string formatting, so never use the `%` operator to build the query); and let the `with` block manage the transaction. `NUMERIC` columns arrive in Python as `decimal.Decimal` and `TIMESTAMPTZ` as timezone-aware `datetime` objects. For DataFrames, `pandas.read_sql` accepts a SQLAlchemy engine connected to Postgres.

## JSONB: documents inside a relational table

Model outputs often have a flexible shape: per-class scores, latency, model version, the prompt used. Rather than add a column for every field, store them in a `JSONB` column. PostgreSQL parses `JSONB` into a binary form that can be queried and indexed.

```sql
UPDATE tickets
SET details = '{"model": "clf-v3", "scores": {"billing": 0.88, "account": 0.07}, "latency_ms": 41}'
WHERE ticket_id = 2;

SELECT ticket_id,
       details ->> 'model'                AS model,
       (details -> 'scores' ->> 'billing')::numeric AS billing_score
FROM tickets
WHERE details @> '{"model": "clf-v3"}'
  AND (details ->> 'latency_ms')::int < 50;
```

| Operator | Returns | Example |
| --- | --- | --- |
| `->` | A JSON value | `details -> 'scores'` |
| `->>` | Text | `details ->> 'model'` gives `clf-v3` |
| `@>` | Boolean: left contains right | `details @> '{"model": "clf-v3"}'` |
| `?` | Boolean: key exists | `details ? 'latency_ms'` |

`->>` always returns text, hence the casts with `::numeric` and `::int` before comparing numbers. A GIN index makes containment queries fast on large tables:

```sql
CREATE INDEX idx_tickets_details ON tickets USING GIN (details);
```

SQLite (3.38 and later) has `->` and `->>` too, applied to JSON stored as text, with JSON-path arguments. This version runs anywhere:

```python
import json
import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE predictions (ticket_id INTEGER PRIMARY KEY, details TEXT)")
conn.executemany("INSERT INTO predictions VALUES (?, ?)", [
    (2, json.dumps({"model": "clf-v3", "scores": {"billing": 0.88, "account": 0.07}, "latency_ms": 41})),
    (3, json.dumps({"model": "clf-v3", "scores": {"shipping": 0.61, "product": 0.30}, "latency_ms": 38})),
    (9, json.dumps({"model": "clf-v2", "scores": {"shipping": 0.58, "product": 0.35}, "latency_ms": 55})),
])
rows = conn.execute("""
    SELECT ticket_id, details ->> '$.model', details ->> '$.scores.product'
    FROM predictions
    WHERE details ->> '$.latency_ms' < 50
""").fetchall()
print(rows)
```

```output
[(2, 'clf-v3', None), (3, 'clf-v3', 0.3)]
```

A missing key gives `NULL` (`None`). Unlike PostgreSQL, SQLite's `->>` returns a number when the JSON value is a number. SQLite has no `@>` operator and no GIN indexes.

> [!NOTE]
> `JSONB` is for genuinely variable data. If every row has a field and you filter or join on it often, make it a proper column with a type and constraints. A table that is mostly one JSON blob gives up the guarantees that made you choose a relational database.

## pgvector: a preview of vector search

An **embedding** is a list of numbers that represents the meaning of a piece of text, so that similar texts have nearby vectors; [vector embeddings](/learn/vector-databases/vector-embeddings) covers how they are produced. The **pgvector** extension adds a `vector` column type and distance operators, so you can store embeddings next to the rows they describe and search them with SQL.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE ticket_embeddings (
    ticket_id BIGINT PRIMARY KEY REFERENCES tickets (ticket_id),
    embedding vector(3)          -- real models produce hundreds or thousands of dimensions
);

INSERT INTO ticket_embeddings VALUES
    (1, '[0.9, 0.1, 0.0]'),      -- Cannot log in after reset
    (2, '[0.1, 0.9, 0.1]'),      -- Charged twice for order
    (6, '[0.0, 0.8, 0.3]'),      -- Refund still not received
    (7, '[0.1, 0.1, 0.9]');      -- Package stuck in transit

-- The three tickets most similar to a new one, "I was billed two times"
SELECT t.ticket_id, t.subject, e.embedding <=> '[0.0, 0.9, 0.2]' AS distance
FROM ticket_embeddings AS e
JOIN tickets AS t USING (ticket_id)
ORDER BY e.embedding <=> '[0.0, 0.9, 0.2]'
LIMIT 3;
```

| Operator | Distance |
| --- | --- |
| `<->` | Euclidean (L2) distance |
| `<=>` | Cosine distance, $1 - \cos\theta$ |
| `<#>` | Negative inner product |

Cosine distance between vectors $\mathbf{a}$ and $\mathbf{b}$ is $1 - \frac{\mathbf{a}^\top \mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$: 0 for vectors pointing the same way, larger as they diverge. You can check the ordering the query produces with NumPy:

```python
import numpy as np

embeddings = {
    1: np.array([0.9, 0.1, 0.0]),   # Cannot log in after reset
    2: np.array([0.1, 0.9, 0.1]),   # Charged twice for order
    6: np.array([0.0, 0.8, 0.3]),   # Refund still not received
    7: np.array([0.1, 0.1, 0.9]),   # Package stuck in transit
}
query = np.array([0.0, 0.9, 0.2])   # "I was billed two times"

def cosine_distance(a, b):
    return 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

ranked = sorted(embeddings, key=lambda t: cosine_distance(embeddings[t], query))
for t in ranked:
    print(t, round(float(cosine_distance(embeddings[t], query)), 4))
```

```output
6 0.0098
2 0.0118
7 0.6785
1 0.8922
```

The two billing tickets come first, so the SQL query returns tickets 6, 2 and 7. Without an index, pgvector compares the query with every row, which is exact but slow at scale. An approximate nearest-neighbour index trades a little accuracy for large speed-ups:

```sql
CREATE INDEX ON ticket_embeddings USING hnsw (embedding vector_cosine_ops);
```

The real attraction is combining vector search with everything else in this module in one query: filter by customer, date or category with `WHERE`, join to other tables, and enforce access with roles. Phase 8 builds on this in [similarity search](/learn/vector-databases/similarity-search) and [metadata filtering](/learn/vector-databases/metadata-filtering).

> [!KEY]
> PostgreSQL can be your relational database, your document store (`JSONB`) and your vector store (pgvector) at once. For many AI applications, one well-run Postgres instance is simpler and more reliable than three separate systems.

## Exercises

**1.** In psql, which commands would you use to (a) see the columns and indexes of `tickets`, (b) list all roles, and (c) display a very wide row readably?

<details>
<summary>Show solution</summary>

(a) `\d tickets`, (b) `\du`, (c) `\x` to toggle expanded display, then run the query.

</details>

**2.** Create a login role `report_bot` that can read `orders` and `products` but nothing else, and cannot write anything.

<details>
<summary>Show solution</summary>

```sql
CREATE ROLE report_bot LOGIN PASSWORD 'change-me';
GRANT CONNECT ON DATABASE shop TO report_bot;
GRANT USAGE ON SCHEMA public TO report_bot;
GRANT SELECT ON orders, products TO report_bot;
```

Granting it the `readonly` group role would be too broad, since that includes `customers` and `tickets`, which contain personal data.

</details>

**3.** Rewrite this SQLite query for PostgreSQL, assuming `created_at` is `TIMESTAMPTZ`: `SELECT * FROM tickets WHERE subject LIKE '%refund%' AND created_at >= date('now', '-7 day')`.

<details>
<summary>Show solution</summary>

```sql
SELECT *
FROM tickets
WHERE subject ILIKE '%refund%'
  AND created_at >= now() - interval '7 days';
```

`ILIKE` restores the case-insensitive matching that SQLite's `LIKE` gave, and interval arithmetic replaces SQLite's `date` modifiers.

</details>

```quiz
[
  {
    "q": "What is the psycopg placeholder for a parameter in a PostgreSQL query from Python?",
    "options": [
      "`?`",
      "`{}`",
      "`:1`",
      "`%s`"
    ],
    "answer": 3,
    "why": "psycopg uses `%s` for every parameter type; values are still passed separately, not formatted into the string."
  },
  {
    "q": "In PostgreSQL, what does `details ->> 'model'` return when `details` is JSONB?",
    "options": [
      "A JSONB value",
      "Text",
      "A boolean",
      "An integer"
    ],
    "answer": 1,
    "why": "`->>` extracts a field as text; `->` returns JSONB. Cast the text when you need numbers."
  },
  {
    "q": "An LLM-powered feature turns user questions into SQL. Which role setup is safest?",
    "options": [
      "Connect as the `postgres` superuser so every query works",
      "Connect as a login role that only has SELECT on the specific tables it needs",
      "Connect as the application's read-write role",
      "Disable roles for that connection"
    ],
    "answer": 1,
    "why": "Least privilege limits the damage from a wrong or malicious generated query to reading permitted tables."
  },
  {
    "q": "Which pgvector operator gives cosine distance?",
    "options": [
      "`<->`",
      "`<#>`",
      "`<=>`",
      "`@>`"
    ],
    "answer": 2,
    "why": "`<=>` is cosine distance, `<->` is Euclidean distance, `<#>` is negative inner product; `@>` is JSONB containment."
  }
]
```
