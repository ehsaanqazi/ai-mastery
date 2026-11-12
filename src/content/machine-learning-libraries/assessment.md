This quiz covers the whole Machine Learning Libraries module: the scikit-learn estimator API and pipelines, XGBoost and LightGBM in practice, and handling imbalanced data with imbalanced-learn. Aim for at least 12 out of 16 before moving on to Neural Network Fundamentals.

```quiz
[
  {
    "q": "In scikit-learn, where should an estimator's hyperparameters be set?",
    "options": [
      "In `fit`",
      "In the constructor",
      "In `predict`",
      "In a global configuration file"
    ],
    "answer": 1,
    "why": "Hyperparameters are constructor arguments; `fit` learns parameters from data and stores them in attributes ending in an underscore."
  },
  {
    "q": "What does calling `predict` on a fitted `Pipeline` of `[scaler, classifier]` do to the scaler?",
    "options": [
      "Refits it on the new data",
      "Calls `fit_transform` on the new data",
      "Skips it",
      "Calls only `transform`, using statistics learned during `fit`"
    ],
    "answer": 3,
    "why": "Prediction never refits preprocessing; it applies what was learned from the training data."
  },
  {
    "q": "Which object applies different preprocessing to numeric and categorical columns of a DataFrame?",
    "options": [
      "`ColumnTransformer`",
      "`FeatureUnion`",
      "`FunctionTransformer`",
      "`make_pipeline`"
    ],
    "answer": 0,
    "why": "`ColumnTransformer` routes column subsets to different transformers and concatenates the outputs."
  },
  {
    "q": "A custom transformer's `__init__` computes `self.threshold = threshold * 2`. What breaks?",
    "options": [
      "Nothing",
      "Only `transform`",
      "`get_params`, `clone` and grid search, because the stored value no longer matches the argument",
      "Only pickling"
    ],
    "answer": 2,
    "why": "scikit-learn rebuilds estimators from `get_params()`, so `__init__` must store each argument unchanged under the same name."
  },
  {
    "q": "What is the safest attitude to a `.joblib` model file downloaded from an unknown website?",
    "options": [
      "Load it in a notebook to inspect it",
      "Load it with `mmap_mode='r'` to make it safe",
      "Load it only if the file is small",
      "Do not load it; unpickling can execute arbitrary code"
    ],
    "answer": 3,
    "why": "joblib files are pickles and can run code when loaded. Memory-mapping does not change that."
  },
  {
    "q": "Selecting the top features on the full dataset and then cross-validating gave 85% accuracy on random labels. What went wrong?",
    "options": [
      "Feature selection saw the validation folds' labels, leaking information",
      "Too few folds",
      "Logistic regression cannot use 20 features",
      "The random seed was not fixed"
    ],
    "answer": 0,
    "why": "Any step that learns from labels must be fitted inside each training fold, which is what a `Pipeline` guarantees."
  },
  {
    "q": "In XGBoost's scikit-learn API, how is early stopping configured in current versions?",
    "options": [
      "`early_stopping_rounds` in the constructor plus an `eval_set` in `fit`",
      "A callback only",
      "`n_estimators=\"auto\"`",
      "It cannot be used with the scikit-learn API"
    ],
    "answer": 0,
    "why": "Recent XGBoost versions take `early_stopping_rounds` (and `eval_metric`) as constructor arguments and monitor the `eval_set` passed to `fit`."
  },
  {
    "q": "After `xgb.train(..., early_stopping_rounds=50)`, how do you predict with only the best trees?",
    "options": [
      "`booster.predict(dtest)` always does this",
      "`booster.trim()`",
      "`booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))`",
      "Retrain with `num_boost_round=best_iteration`; there is no other way"
    ],
    "answer": 2,
    "why": "The native booster keeps every tree it built; `iteration_range` selects how many to use."
  },
  {
    "q": "XGBoost's `weight` importance ranks a random ID column near the top. Which tool gives a more trustworthy view?",
    "options": [
      "`cover` importance",
      "Permutation importance on held-out data",
      "Counting how many trees there are",
      "`total_gain` on the training set"
    ],
    "answer": 1,
    "why": "Permutation importance measures how much held-out performance drops when a feature is shuffled, so overfitted noise features score near zero."
  },
  {
    "q": "What does `xgb.XGBClassifier(tree_method=\"hist\", device=\"cuda\")` request?",
    "options": [
      "CPU training with exact splits",
      "Approximate training on the CPU with CUDA-style threads",
      "Training on multiple machines",
      "Histogram-based training on an NVIDIA GPU"
    ],
    "answer": 3,
    "why": "`device=\"cuda\"` moves training to the GPU in recent XGBoost versions, replacing the older `gpu_hist` tree method."
  },
  {
    "q": "Why is `num_leaves` LightGBM's main complexity control?",
    "options": [
      "LightGBM ignores `max_depth`",
      "Trees grow leaf-wise, always splitting the best leaf, so depth alone does not bound tree size well",
      "It sets the number of trees",
      "It controls histogram bins"
    ],
    "answer": 1,
    "why": "Leaf-wise growth can produce deep, lopsided trees, so limiting the number of leaves is the natural size control."
  },
  {
    "q": "How is early stopping added to `LGBMClassifier.fit` in LightGBM 4?",
    "options": [
      "`callbacks=[lgb.early_stopping(50)]` with validation data",
      "`early_stopping_rounds=50` in `fit`",
      "`stop_early=True` in the constructor",
      "`lgb.cv` only"
    ],
    "answer": 0,
    "why": "LightGBM 4 uses the `early_stopping` callback and monitors the validation data you pass."
  },
  {
    "q": "In LightGBM's scikit-learn API, `subsample=0.8` is set but nothing changes. Why?",
    "options": [
      "`subsample` only works on GPU",
      "`subsample` must be an integer",
      "`subsample_freq` defaults to 0, which disables row bagging",
      "Row sampling requires `num_leaves` above 255"
    ],
    "answer": 2,
    "why": "Bagging runs only when its frequency (`subsample_freq` or `bagging_freq`) is positive."
  },
  {
    "q": "What does a sampler's `fit_resample(X, y)` return?",
    "options": [
      "A fitted model",
      "A new `X` and `y`, possibly with a different number of rows",
      "Only the minority-class rows",
      "Class weights"
    ],
    "answer": 1,
    "why": "Samplers change the dataset itself, which is why they need imbalanced-learn's own pipeline rather than scikit-learn's."
  },
  {
    "q": "Where should SMOTE be applied when cross-validating?",
    "options": [
      "To the full dataset before splitting",
      "To the test set before final evaluation",
      "To the validation folds only",
      "To the training part of each fold only, via `imblearn.pipeline.Pipeline`"
    ],
    "answer": 3,
    "why": "Validation and test data must keep the real class balance and must not influence the synthetic training points."
  },
  {
    "q": "Which metric is the most informative single summary for a model on data with 1% positives?",
    "options": [
      "Accuracy",
      "Average precision (area under the precision-recall curve)",
      "Training loss",
      "The number of trees"
    ],
    "answer": 1,
    "why": "Accuracy is dominated by the majority class. Average precision focuses on how well positives are ranked and is sensitive to false positives among a small positive class."
  }
]
```
