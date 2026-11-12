This quiz covers the whole Model Evaluation module: classification metrics, regression metrics, cross-validation and hyperparameter search. Aim for at least 13 out of 16 before moving on to Machine Learning Libraries.

```quiz
[
  {
    "q": "A test set has 1,000 transactions, 10 of them fraudulent. A model that always predicts \"legitimate\" has what accuracy and what recall?",
    "options": ["99% accuracy, 0% recall", "1% accuracy, 100% recall", "99% accuracy, 99% recall", "50% accuracy, 0% recall"],
    "answer": 0,
    "why": "It is right on all 990 legitimate transactions (99% accuracy) but catches none of the 10 frauds (recall 0). This is the accuracy paradox."
  },
  {
    "q": "A model flags 40 emails as spam; 30 really are spam. There were 60 spam emails in total. What are its precision and recall?",
    "options": ["Precision 0.5, recall 0.75", "Precision 0.75, recall 0.5", "Precision 0.75, recall 0.75", "Precision 0.5, recall 0.5"],
    "answer": 1,
    "why": "Precision = 30 / 40 = 0.75 (of the flagged emails); recall = 30 / 60 = 0.5 (of all spam)."
  },
  {
    "q": "Lowering a classifier's decision threshold from 0.5 to 0.2 will usually:",
    "options": [
      "Raise precision and lower recall",
      "Leave precision and recall unchanged",
      "Raise recall and lower precision",
      "Raise both precision and recall"
    ],
    "answer": 2,
    "why": "More examples are flagged, so more positives are caught (recall up), but more false alarms appear (precision tends to fall)."
  },
  {
    "q": "With TP = 20, FP = 10 and FN = 30, what is the F1-score?",
    "options": ["0.4", "0.5", "0.67", "0.33"],
    "answer": 1,
    "why": "F1 = 2TP / (2TP + FP + FN) = 40 / (40 + 10 + 30) = 40 / 80 = 0.5."
  },
  {
    "q": "A three-class model scores per-class F1 of 0.95, 0.90 and 0.10, and the last class is rare. Which summary hides the rare-class failure most?",
    "options": ["Macro F1", "The per-class report", "Balanced accuracy", "Weighted F1"],
    "answer": 3,
    "why": "Weighted averaging weights each class by its support, so a rare class contributes little. Macro gives every class equal weight."
  },
  {
    "q": "After `tn, fp, fn, tp = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]).ravel()`, what does `print(tp, fp)` show?",
    "options": ["2 1", "1 2", "2 2", "1 1"],
    "answer": 0,
    "why": "The true positives are rows 3 and 4 (2), and the single false positive is row 2 (1). `ravel` returns TN, FP, FN, TP."
  },
  {
    "q": "A fraud model has ROC-AUC of 0.95, but its fraud alerts are mostly false alarms. How can both be true?",
    "options": [
      "ROC-AUC was computed on the training set",
      "With very few positives, many false alarms are still a tiny false positive rate, while precision suffers",
      "ROC-AUC measures precision, not recall",
      "It cannot happen; one of the numbers must be wrong"
    ],
    "answer": 1,
    "why": "FPR divides by the huge number of negatives, so ROC-AUC barely notices false alarms that dominate the alert queue. Average precision reflects them."
  },
  {
    "q": "Why pass `predict_proba(X)[:, 1]` rather than `predict(X)` to `roc_auc_score`?",
    "options": [
      "ROC-AUC needs scores to rank examples across all thresholds",
      "`predict` output is not a NumPy array",
      "It makes the computation faster",
      "`roc_auc_score` only accepts floats"
    ],
    "answer": 0,
    "why": "Hard 0/1 labels give a single threshold, collapsing the ROC curve to one point and producing a misleading AUC."
  },
  {
    "q": "House-price errors (in thousands) are 2, -2, 4, -4 and 40. Which statement is true?",
    "options": [
      "MAE and RMSE are equal",
      "MAE is larger than RMSE",
      "The mean error is zero",
      "RMSE is much larger than MAE, because squaring makes the 40 dominate"
    ],
    "answer": 3,
    "why": "MAE = 52 / 5 = 10.4, while RMSE = sqrt(1640 / 5) ≈ 18.1. The single large error dominates the squared metric. The mean error is 8, not zero."
  },
  {
    "q": "If a model had to predict one constant for every example, which constant minimises MSE, and which minimises MAE?",
    "options": [
      "Median for MSE, mean for MAE",
      "Mean for both",
      "Mean for MSE, median for MAE",
      "Mode for both"
    ],
    "answer": 2,
    "why": "Squared deviations are minimised by the mean; absolute deviations by the median, which is why MAE is more robust to outliers."
  },
  {
    "q": "`cross_val_score(model, X, y, scoring=\"neg_mean_absolute_error\")` returns values around -12.5. What is the MAE?",
    "options": ["-12.5", "12.5", "156.25", "It cannot be determined"],
    "answer": 1,
    "why": "scikit-learn negates error metrics so that higher is always better. Flip the sign: MAE is 12.5."
  },
  {
    "q": "Your data contains many sessions per user, and the model will serve new users. Plain `KFold` gives 0.97 accuracy; `GroupKFold` by user gives 0.71. Which should you trust?",
    "options": [
      "`GroupKFold`, because it keeps each user entirely in training or validation, as in production",
      "`KFold`, because it uses more data per fold",
      "The average of the two",
      "Neither; use the training accuracy"
    ],
    "answer": 0,
    "why": "With `KFold` the model sees other sessions of the same user during training and can memorise users, inflating the score."
  },
  {
    "q": "Which splitter should you use to evaluate a model that forecasts next week's demand from past sales?",
    "options": ["`StratifiedKFold`", "`KFold(shuffle=True)`", "`GroupKFold`", "`TimeSeriesSplit`"],
    "answer": 3,
    "why": "Time-ordered data must always train on the past and test on the future; shuffled splits let the model learn from the future."
  },
  {
    "q": "A team tries 400 hyperparameter settings and reports the best mean CV score as the expected production accuracy. What is wrong?",
    "options": [
      "Nothing; cross-validation removes all bias",
      "The best of many noisy estimates is optimistically biased; they need an untouched test set or nested CV",
      "They should have reported the worst score instead",
      "CV scores cannot be used to choose hyperparameters"
    ],
    "answer": 1,
    "why": "Selecting the maximum rewards lucky folds as well as genuinely good settings, so the winning score overstates true performance."
  },
  {
    "q": "In a pipeline `Pipeline([(\"scale\", StandardScaler()), (\"svc\", SVC())])`, which `GridSearchCV` grid is valid?",
    "options": [
      "`{\"C\": [0.1, 1, 10]}`",
      "`{\"svc.C\": [0.1, 1, 10]}`",
      "`{\"svc__C\": [0.1, 1, 10], \"svc__gamma\": [0.01, 0.1]}`",
      "`{\"SVC__C\": [0.1, 1, 10]}`"
    ],
    "answer": 2,
    "why": "Pipeline parameters are addressed as step name, double underscore, parameter name, using the step names you chose."
  },
  {
    "q": "With 5 hyperparameters, why is `RandomizedSearchCV(n_iter=50)` usually a better use of budget than a grid with 2 values each (32 combinations)?",
    "options": [
      "Random search uses more folds",
      "The grid tests only 2 distinct values of each hyperparameter, while random search tests up to 50, exploring the important ones much better",
      "Random search does not need a scoring metric",
      "Grids cannot be parallelised"
    ],
    "answer": 1,
    "why": "Usually only a few hyperparameters matter. Random sampling gives a new value of each on every trial, whereas a grid keeps repeating the same few."
  }
]
```
