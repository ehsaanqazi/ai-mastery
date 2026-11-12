This quiz covers the whole Machine Learning Fundamentals module: the learning setup, the main learning paradigms, data splitting and leakage, overfitting and the bias-variance tradeoff, and feature engineering, selection and preprocessing. Aim for at least 13 out of 16 before moving on to Supervised Learning Algorithms.

```quiz
[
  {
    "q": "Training a model minimises the empirical risk. What do you actually want to be small?",
    "options": [
      "The training loss",
      "The expected loss on new data from the same distribution (the true risk)",
      "The number of parameters",
      "The size of the training set"
    ],
    "answer": 1,
    "why": "Empirical risk is a stand-in. Generalisation means low error on unseen data, which is what the true risk measures."
  },
  {
    "q": "Which statement about AI, machine learning and deep learning is correct?",
    "options": [
      "Every AI system learns from data",
      "Deep learning includes all of machine learning",
      "A route planner using A-star search is machine learning",
      "Deep learning is a subset of machine learning, which is a subset of AI"
    ],
    "answer": 3,
    "why": "The fields are nested. Symbolic AI such as search and expert systems is AI without learning."
  },
  {
    "q": "Predicting which of five product categories a support ticket belongs to is:",
    "options": ["Regression", "Multiclass classification", "Clustering", "Reinforcement learning"],
    "answer": 1,
    "why": "The label is one category from a fixed set of more than two, and labelled examples are available."
  },
  {
    "q": "k-means inertia keeps falling as k increases. What does that imply?",
    "options": [
      "You cannot choose k simply by minimising inertia",
      "The largest k is always best",
      "The data has no clusters",
      "k-means has failed to converge"
    ],
    "answer": 0,
    "why": "More centres always reduce distances to the nearest centre. Use the elbow, the silhouette score, or usefulness to choose k."
  },
  {
    "q": "In a semi-supervised experiment, self-training scores lower than a model trained on the labelled data alone. What is the most likely cause?",
    "options": [
      "The test set was too large",
      "Unlabelled data cannot be used by any method",
      "Confidently wrong pseudo-labels reinforced the initial model's errors",
      "The labelled set was stratified"
    ],
    "answer": 2,
    "why": "Self-training trusts the model's own confident predictions. When the starting model is weak, mistakes become training labels."
  },
  {
    "q": "In Q-learning, after taking action a in state s, the agent receives reward r = 0 and lands in s', where the best Q-value is 0.8. With gamma = 0.9, what is the update target?",
    "options": ["0.8", "0.9", "0", "0.72"],
    "answer": 3,
    "why": "The target is r + gamma * max Q(s', a') = 0 + 0.9 * 0.8 = 0.72."
  },
  {
    "q": "In RLHF, what plays the role of the policy?",
    "options": [
      "The language model being tuned",
      "The reward model",
      "The human raters",
      "The pretraining dataset"
    ],
    "answer": 0,
    "why": "The LLM chooses actions (tokens) given states (the text so far). The reward model supplies the reward."
  },
  {
    "q": "Why keep a test set separate from the validation set?",
    "options": [
      "The test set is used to fit the hyperparameters",
      "Selecting models on validation scores makes those scores optimistic, so an untouched set is needed for an honest final estimate",
      "scikit-learn requires three splits",
      "The validation set is only for regression problems"
    ],
    "answer": 1,
    "why": "Every decision made using a dataset leaks a little information. The test set must play no part in any decision."
  },
  {
    "q": "What does this code demonstrate?\n\n`X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)`\n`cross_val_score(LogisticRegression(), X_sel, y, cv=5)`",
    "options": [
      "Correct use of a pipeline",
      "Stratified sampling",
      "Selection leakage: features were chosen using the validation folds' labels",
      "Time-series cross-validation"
    ],
    "answer": 2,
    "why": "The selector saw every row, including each fold's validation rows, so the cross-validated score is inflated. Put the selector in a pipeline."
  },
  {
    "q": "You are forecasting daily demand. Which splitting strategy is appropriate?",
    "options": [
      "A random shuffle with `train_test_split`",
      "Stratified k-fold on the demand values",
      "Putting every Monday in the test set",
      "`TimeSeriesSplit`, training on earlier days and validating on later ones"
    ],
    "answer": 3,
    "why": "The model will only ever predict the future from the past, so evaluation must respect time order."
  },
  {
    "q": "A model scores training MSE 0.01 and validation MSE 0.45. The noise variance is about 0.04. What is the diagnosis?",
    "options": [
      "Overfitting: high variance",
      "Underfitting: high bias",
      "A good fit",
      "Irreducible error dominates"
    ],
    "answer": 0,
    "why": "Training error is below the noise floor and validation error is far above it: the model has fitted the noise."
  },
  {
    "q": "In the decomposition expected error = bias^2 + variance + noise, which change typically lowers variance without increasing bias?",
    "options": [
      "Lowering the polynomial degree",
      "Increasing Ridge `alpha`",
      "Collecting more training data from the same source",
      "Using a shallower tree"
    ],
    "answer": 2,
    "why": "More data pins the fit down without restricting what the model can represent. The other options trade bias for variance."
  },
  {
    "q": "Why encode hour of day as a sine and cosine pair?",
    "options": [
      "So that hour 23 and hour 0 end up close together",
      "To reduce the number of features",
      "Because models cannot read integers",
      "To remove the need for a test set"
    ],
    "answer": 0,
    "why": "Mapping the cycle onto a circle makes adjacent hours adjacent in feature space, including across midnight."
  },
  {
    "q": "Which feature selection method can detect that two near-identical columns are redundant?",
    "options": [
      "A variance threshold",
      "Forward sequential selection with cross-validation",
      "Ranking features by their individual F-test score",
      "Ranking features by mutual information with the target"
    ],
    "answer": 1,
    "why": "Wrappers judge each feature in the context of those already chosen, so a duplicate adds almost nothing and is skipped. Filters score features one at a time."
  },
  {
    "q": "What is wrong with calling `StandardScaler().fit_transform(X)` and then `train_test_split` on the result?",
    "options": [
      "Nothing, as long as `random_state` is set",
      "Scaling must always come after the model",
      "`StandardScaler` cannot be used with `train_test_split`",
      "The scaler's mean and standard deviation include test rows, which leaks information"
    ],
    "answer": 3,
    "why": "Preprocessing must be fitted on training data only. Split first, or use a pipeline."
  },
  {
    "q": "Which model would give the same predictions whether or not you standardise its numeric features?",
    "options": [
      "k-nearest neighbours",
      "Logistic regression with L2 regularisation",
      "A decision tree",
      "k-means clustering"
    ],
    "answer": 2,
    "why": "Tree splits depend only on the order of a feature's values, and standardisation preserves that order."
  }
]
```
