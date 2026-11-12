This quiz covers the whole Pandas module: Series and DataFrames, loading files, cleaning, missing values, filtering, grouping, merging, transformation and feature preparation. Aim for at least 13 out of 16 before moving on to Data Visualization.

```quiz
[
  {
    "q": "`a = pd.Series([1, 2], index=[\"x\", \"y\"])` and `b = pd.Series([10, 20], index=[\"y\", \"z\"])`. What is `(a + b)[\"y\"]`?",
    "options": ["`12`", "`22`", "`NaN`", "`21`"],
    "answer": 0,
    "why": "pandas aligns on labels: `a[\"y\"]` is 2 and `b[\"y\"]` is 10, giving 12. Labels `x` and `z` produce `NaN`."
  },
  {
    "q": "After `df = df.sort_values(\"score\")`, what is the difference between `df.loc[0]` and `df.iloc[0]`?",
    "options": ["There is none after sorting", "`df.loc[0]` is the row labelled 0; `df.iloc[0]` is the row now in first position", "`df.loc[0]` is the first row; `df.iloc[0]` is the row labelled 0", "`df.loc[0]` raises an error after sorting"],
    "answer": 1,
    "why": "Sorting keeps the original labels. `.loc` selects by label and `.iloc` by position."
  },
  {
    "q": "Which `read_csv` call correctly reads `1.234,50` as the number 1234.5?",
    "options": ["`pd.read_csv(f, decimal=\".\", thousands=\",\")`", "`pd.read_csv(f, sep=\",\")`", "`pd.read_csv(f, dtype=float)`", "`pd.read_csv(f, sep=\";\", decimal=\",\", thousands=\".\")`"],
    "answer": 3,
    "why": "The European format uses a comma as the decimal mark and a dot for thousands, and such files usually use `;` as the separator."
  },
  {
    "q": "You need one row per order line from API records shaped like `{\"order_id\": 7, \"items\": [{...}, {...}]}`. Which call fits?",
    "options": ["`pd.read_json(records)`", "`pd.json_normalize(records)`", "`pd.json_normalize(records, record_path=\"items\", meta=[\"order_id\"])`", "`pd.DataFrame(records).explode(\"order_id\")`"],
    "answer": 2,
    "why": "`record_path` makes one row per child record and `meta` copies parent fields onto each row."
  },
  {
    "q": "After `s = pd.to_numeric(raw, errors=\"coerce\")`, what is the most important next step?",
    "options": ["Inspect which original values became `NaN`, to catch formats you have not handled", "Convert `s` to `int64`", "Drop all rows where `s` is missing", "Nothing: coercion guarantees clean numbers"],
    "answer": 0,
    "why": "Coercion turns every unparseable value into `NaN`, including values like `\"80.5 GBP\"` that should have been cleaned rather than discarded."
  },
  {
    "q": "What does `df.sort_values(\"updated_at\").drop_duplicates(subset=\"user_id\", keep=\"last\")` return?",
    "options": ["Every row except the newest per user", "The oldest row per user", "Only users who appear once", "The most recent row per user"],
    "answer": 3,
    "why": "After sorting by time, the last row of each `user_id` is the newest, and `keep=\"last\"` keeps it."
  },
  {
    "q": "Why does `df[df[\"x\"] == np.nan]` return no rows?",
    "options": ["`NaN` compares unequal to everything, including itself", "`np.nan` must be written `None`", "Filtering requires `.loc`", "pandas stores missing floats as `pd.NA`"],
    "answer": 0,
    "why": "Use `df[df[\"x\"].isna()]` to find missing values."
  },
  {
    "q": "Which code fills missing `price` values with the median price of the row's own `category`?",
    "options": ["`df[\"price\"].fillna(df[\"price\"].median())`", "`df[\"price\"].fillna(df.groupby(\"category\")[\"price\"].transform(\"median\"))`", "`df[\"price\"].fillna(df.groupby(\"category\")[\"price\"].median())`", "`df.groupby(\"category\")[\"price\"].bfill()`"],
    "answer": 1,
    "why": "`transform` returns a Series aligned with the original rows, so each row is filled with its own group's median. The plain `median()` result is indexed by category, not by row."
  },
  {
    "q": "Which statement about missing values in a training pipeline is correct?",
    "options": ["Fill values should be computed on the full dataset for accuracy", "Backward fill is safe for forecasting because it uses real values", "Fill values should be computed on the training set and reused on test and production data", "Dropping every row with any missing value is always the safest option"],
    "answer": 2,
    "why": "Fill values are learned parameters. Computing them on test rows leaks information; `bfill` uses future values; `dropna()` can discard most of a dataset."
  },
  {
    "q": "What does this print? `df = pd.DataFrame({\"a\": [1, 2, 3]}); sub = df[df[\"a\"] > 1]; sub[\"a\"] = 0; print(df[\"a\"].tolist())` (pandas 3)",
    "options": ["`[1, 0, 0]`", "`[0, 0, 0]`", "It raises an error", "`[1, 2, 3]`"],
    "answer": 3,
    "why": "Under copy-on-write, `sub` behaves as an independent copy, so `df` is unchanged. Use `df.loc[df[\"a\"] > 1, \"a\"] = 0` to modify `df`."
  },
  {
    "q": "Why must each condition be parenthesised in `df[(df[\"a\"] > 1) & (df[\"b\"] < 5)]`?",
    "options": ["`&` binds more tightly than comparison operators", "Parentheses make the mask faster", "Python's `and` requires them", "pandas cannot compare two columns without them"],
    "answer": 0,
    "why": "Without parentheses Python evaluates `1 & df[\"b\"]` first, which is not what you meant."
  },
  {
    "q": "Which `groupby` method returns a result with exactly the same number of rows as the input?",
    "options": ["`agg`", "`transform`", "`filter`", "`size`"],
    "answer": 1,
    "why": "`transform` broadcasts each group's result back to its rows. `agg` and `size` return one row per group; `filter` returns a subset."
  },
  {
    "q": "Orders are left-joined to customers on `customer_id`, and the result has more rows than `orders`. What is the most likely cause?",
    "options": ["Some orders have no matching customer", "The key columns have different dtypes", "The join should have been `how=\"inner\"`", "Some `customer_id` values appear more than once in the customer table"],
    "answer": 3,
    "why": "Duplicate keys on the right multiply matching rows. `validate=\"many_to_one\"` would have raised a `MergeError`. Unmatched orders keep their row count in a left join."
  },
  {
    "q": "What does `indicator=True` add to a merge result?",
    "options": ["A `_merge` column with `both`, `left_only` or `right_only` for each row", "A count of matched rows per key", "A boolean column marking duplicate keys", "An index of the original row positions"],
    "answer": 0,
    "why": "The `_merge` column shows where each row came from, which makes unmatched rows easy to count and inspect."
  },
  {
    "q": "You want a feature holding each store's average sales over the previous 7 days, to predict today's sales. Which is correct?",
    "options": ["`df.groupby(\"store\")[\"sales\"].transform(lambda s: s.rolling(7).mean())`", "`df[\"sales\"].rolling(7).mean()`", "`df.groupby(\"store\")[\"sales\"].transform(lambda s: s.shift(1).rolling(7).mean())`", "`df.groupby(\"store\")[\"sales\"].transform(lambda s: s.shift(-1).rolling(7).mean())`"],
    "answer": 2,
    "why": "Shifting by one excludes today's value from the window, and grouping keeps stores separate. Without the shift, today's target leaks into its own feature; `shift(-1)` pulls in tomorrow."
  },
  {
    "q": "A fitted `OneHotEncoder(handle_unknown=\"ignore\")` sees a category at prediction time that was not in the training data. What happens?",
    "options": ["It raises an error", "It adds a new column for the category", "It encodes the row as all zeros in that feature's columns", "It assigns the row to the most frequent category"],
    "answer": 2,
    "why": "Unknown categories become all zeros, so the output keeps exactly the columns learned during `fit`."
  }
]
```
