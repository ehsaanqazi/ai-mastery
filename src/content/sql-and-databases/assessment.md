This quiz covers the whole SQL and Databases module: relational design, CRUD statements, filtering and grouping, joins, subqueries, CTEs, window functions, indexing, PostgreSQL and optimisation. Questions refer to the shop schema (`customers`, `products`, `orders`, `tickets`) used throughout. Aim for at least 13 out of 16 before moving on to Data Engineering Fundamentals.

```quiz
[
  {
    "q": "Which ACID property guarantees that if the second of two statements in a transaction fails, the first one's change is undone?",
    "options": [
      "Isolation",
      "Consistency",
      "Atomicity",
      "Durability"
    ],
    "answer": 2,
    "why": "Atomicity makes a transaction all or nothing."
  },
  {
    "q": "A product can carry many tags and a tag can apply to many products. How should you model this?",
    "options": [
      "A junction table `product_tags (product_id, tag_id)` with a composite primary key",
      "A `tag_id` foreign key on `products`",
      "A comma-separated `tags` column on `products`",
      "A `product_id` foreign key on `tags`"
    ],
    "answer": 0,
    "why": "Many-to-many relationships need a junction table with one row per pair."
  },
  {
    "q": "On a fresh `sqlite3` connection, you insert an order whose `customer_id` does not exist in `customers`. What happens by default?",
    "options": [
      "A FOREIGN KEY constraint error",
      "The order's customer_id is set to NULL",
      "The customer is created automatically",
      "The row is inserted, because SQLite only enforces foreign keys after `PRAGMA foreign_keys = ON`"
    ],
    "answer": 3,
    "why": "SQLite parses REFERENCES clauses but enforces them only when the pragma is enabled on that connection. PostgreSQL always enforces them."
  },
  {
    "q": "What does `UPDATE tickets SET true_label = 'billing';` do?",
    "options": [
      "Updates the most recent ticket",
      "Sets true_label to 'billing' on every ticket",
      "Fails because WHERE is required",
      "Updates only tickets where true_label is NULL"
    ],
    "answer": 1,
    "why": "Without WHERE, UPDATE applies to every row. Write the WHERE first as a SELECT and check the count."
  },
  {
    "q": "Which query correctly finds tickets that have not been reviewed?",
    "options": [
      "`SELECT * FROM tickets WHERE true_label = NULL`",
      "`SELECT * FROM tickets WHERE true_label IS NULL`",
      "`SELECT * FROM tickets WHERE true_label = ''`",
      "`SELECT * FROM tickets WHERE NOT true_label`"
    ],
    "answer": 1,
    "why": "Comparisons with NULL are unknown, never true. Use IS NULL."
  },
  {
    "q": "In which logical order are these clauses evaluated?",
    "options": [
      "SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY",
      "FROM, GROUP BY, WHERE, SELECT, HAVING, ORDER BY",
      "FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY",
      "WHERE, FROM, SELECT, GROUP BY, ORDER BY, HAVING"
    ],
    "answer": 2,
    "why": "Rows are chosen and filtered, then grouped and group-filtered, then SELECT computes outputs, then sorting happens."
  },
  {
    "q": "What does this return: `SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) >= 3`?",
    "options": [
      "Customers with at least three orders, and their order counts",
      "Every order placed by a customer with ID 3 or more",
      "The first three orders of each customer",
      "An error, because HAVING needs WHERE"
    ],
    "answer": 0,
    "why": "GROUP BY makes one row per customer; HAVING keeps the groups whose count is at least 3."
  },
  {
    "q": "Which query lists every customer, including those with no orders, with their order count?",
    "options": [
      "`SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.name`",
      "`SELECT c.name, COUNT(o.order_id) FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.name`",
      "`SELECT c.name, COUNT(*) FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.name`",
      "`SELECT c.name, COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.name`"
    ],
    "answer": 3,
    "why": "A LEFT JOIN from customers keeps customers without orders, and COUNT(o.order_id) gives them 0. COUNT(*) would count their NULL row as 1."
  },
  {
    "q": "A customer has 4 orders and 3 tickets. You join `customers` to both `orders` and `tickets` on customer_id and compute `COUNT(o.order_id)`. What do you get for this customer?",
    "options": [
      "4",
      "3",
      "7",
      "12"
    ],
    "answer": 3,
    "why": "Each order row pairs with each ticket row (fan-out): 4 × 3 = 12 rows. Aggregate each child table separately before joining."
  },
  {
    "q": "Why can `WHERE x NOT IN (SELECT y FROM t)` return no rows at all?",
    "options": [
      "If any y is NULL, every NOT IN comparison is unknown, so no row qualifies",
      "NOT IN only works with constants",
      "Subqueries in WHERE are not allowed",
      "NOT IN is evaluated before FROM"
    ],
    "answer": 0,
    "why": "`x <> NULL` is unknown, which poisons the whole NOT IN. Use NOT EXISTS or filter out NULLs in the subquery."
  },
  {
    "q": "What is the defining feature of a recursive CTE?",
    "options": [
      "It is computed once per session",
      "It must contain a window function",
      "It can only be used in PostgreSQL",
      "An anchor query plus a recursive query that references the CTE itself, combined with UNION ALL"
    ],
    "answer": 3,
    "why": "The anchor produces starting rows; the recursive part generates new rows from the previous ones until none are produced."
  },
  {
    "q": "Which query returns each customer's most recent order, one row per customer?",
    "options": [
      "`SELECT * FROM orders ORDER BY order_date DESC LIMIT 1`",
      "A CTE computing `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn`, then `WHERE rn = 1`",
      "`SELECT customer_id, MAX(order_id) FROM orders`",
      "`SELECT * FROM orders WHERE ROW_NUMBER() OVER (PARTITION BY customer_id) = 1`"
    ],
    "answer": 1,
    "why": "Number the rows within each customer's partition, then filter in an outer query. Window functions cannot appear in WHERE."
  },
  {
    "q": "An index exists on `orders (customer_id, order_date)`. Which query can search it?",
    "options": [
      "`WHERE order_date = '2024-06-10'`",
      "`WHERE customer_id = 2 AND order_date >= '2024-06-10'`",
      "`WHERE status = 'shipped'`",
      "`WHERE lower(status) = 'shipped'`"
    ],
    "answer": 1,
    "why": "A composite index helps queries that constrain its leading column; here equality on customer_id plus a range on order_date."
  },
  {
    "q": "In PostgreSQL, which statement matches subjects containing 'refund' regardless of case?",
    "options": [
      "`WHERE subject LIKE '%refund%'`",
      "`WHERE subject = 'refund'`",
      "`WHERE subject ILIKE '%refund%'`",
      "`WHERE subject @> 'refund'`"
    ],
    "answer": 2,
    "why": "PostgreSQL's LIKE is case-sensitive; ILIKE is the case-insensitive version."
  },
  {
    "q": "A Python loop runs `SELECT ... FROM orders WHERE customer_id = ?` once for each of 5,000 customers. What is the best fix?",
    "options": [
      "Replace the loop with one query using a JOIN, GROUP BY or `IN (...)`",
      "Add more indexes to orders",
      "Open a new connection for each query",
      "Use EXPLAIN ANALYZE inside the loop"
    ],
    "answer": 0,
    "why": "This is the N+1 problem: the cost is thousands of round trips, not any one slow query."
  },
  {
    "q": "Which line safely runs a query with a user-supplied email using psycopg?",
    "options": [
      "`cur.execute(\"SELECT * FROM customers WHERE email = '%s'\" % email)`",
      "`cur.execute(f\"SELECT * FROM customers WHERE email = '{email}'\")`",
      "`cur.execute(\"SELECT * FROM customers WHERE email = \" + email)`",
      "`cur.execute(\"SELECT * FROM customers WHERE email = %s\", (email,))`"
    ],
    "answer": 3,
    "why": "Only the parameterised call sends the value separately from the SQL. The `%` operator version is string formatting and is just as vulnerable as an f-string."
  }
]
```
