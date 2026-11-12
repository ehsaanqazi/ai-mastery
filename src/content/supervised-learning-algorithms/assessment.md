This quiz covers the whole module: linear and logistic models, decision trees and forests, SVMs, KNN, naive Bayes and the gradient boosting family. Aim for at least 13 out of 16 before moving on to Unsupervised Learning.

```quiz
[
  {
    "q": "Which equation gives the least-squares parameters of a linear regression in closed form?",
    "options": ["theta = X^T y", "theta = (X^T X)^-1 X^T y", "theta = (X X^T)^-1 y", "theta = X^-1 y^T"],
    "answer": 1,
    "why": "Setting the gradient of the squared error to zero gives the normal equation X^T X theta = X^T y."
  },
  {
    "q": "A regression model scores R^2 = -0.2 on the test set. What does that tell you?",
    "options": ["It explains 20% of the variance", "The features are perfectly correlated", "It predicts worse than always predicting the mean of the test targets", "The calculation must be wrong, since R^2 cannot be negative"],
    "answer": 2,
    "why": "R^2 compares the model's squared error with that of a constant mean prediction. On held-out data a bad model can do worse than that baseline, giving a negative value."
  },
  {
    "q": "Two nearly identical features are fitted with plain least squares and get weights +8.1 and -6.0. What is the best remedy if you need stable, interpretable weights?",
    "options": ["Add a ridge (L2) penalty or drop one of the features", "Increase the learning rate", "Fit a deeper model", "Remove the intercept"],
    "answer": 0,
    "why": "Multicollinearity makes the split of credit between correlated features unstable. Ridge shares it sensibly; removing a duplicate also works."
  },
  {
    "q": "With `z = 0.0`, what does `1 / (1 + np.exp(-z))` evaluate to?",
    "options": ["0.0", "1.0", "0.25", "0.5"],
    "answer": 3,
    "why": "The sigmoid of 0 is 1 / (1 + 1) = 0.5, which is why the logistic regression decision boundary is where the linear score equals zero."
  },
  {
    "q": "In logistic regression, a feature has weight 1.1. What happens when that feature increases by one unit?",
    "options": ["The probability rises by 1.1", "The odds are multiplied by about e^1.1, roughly 3", "The log-odds are multiplied by 1.1", "The prediction threshold moves to 1.1"],
    "answer": 1,
    "why": "Weights add to the log-odds, so the odds are multiplied by e raised to the weight."
  },
  {
    "q": "A node holds 6 examples of class A and 2 of class B. What is its Gini impurity?",
    "options": ["0.375", "0.25", "0.5", "0.75"],
    "answer": 0,
    "why": "Proportions are 0.75 and 0.25, so G = 1 - 0.5625 - 0.0625 = 0.375."
  },
  {
    "q": "Why do decision trees, random forests and boosted trees not need feature scaling?",
    "options": ["They scale features internally", "They only work with binary features", "Splits compare a feature against a threshold, which is unaffected by order-preserving rescaling", "They use cosine distance"],
    "answer": 2,
    "why": "Rescaling moves thresholds but not which examples fall on each side of a split."
  },
  {
    "q": "In a random forest, what is the main purpose of `max_features` being smaller than the total number of features?",
    "options": ["To make each tree faster to predict", "To decorrelate the trees so that averaging reduces more variance", "To prevent trees from growing deep", "To enable out-of-bag scoring"],
    "answer": 1,
    "why": "Variance of the average is rho sigma^2 + (1 - rho) sigma^2 / B; lowering correlation rho makes averaging more effective."
  },
  {
    "q": "Which statement about the number of trees is correct?",
    "options": ["Adding trees to a random forest eventually overfits badly, while boosting never overfits", "Both benefit from as many trees as possible without validation", "Neither is affected by the number of trees", "Adding trees to a random forest only plateaus, while too many boosting rounds can overfit without early stopping"],
    "answer": 3,
    "why": "Forest trees are averaged independently, so more only stabilises. Boosting keeps fitting smaller residuals, including noise."
  },
  {
    "q": "An RBF-kernel SVM has 100% training accuracy and poor test accuracy. Which pair of changes is most likely to help?",
    "options": ["Lower `gamma` and lower `C`", "Raise `gamma` and raise `C`", "Remove feature scaling", "Switch to `probability=True`"],
    "answer": 0,
    "why": "High gamma and high C both produce a boundary that wraps tightly around training points. Lowering them smooths it."
  },
  {
    "q": "A KNN classifier on unscaled data, with one feature ranging in the thousands and the rest below 1, performs poorly. Why?",
    "options": ["k is too small", "KNN cannot handle more than one feature", "The large-range feature dominates the Euclidean distance", "KNN needs categorical features"],
    "answer": 2,
    "why": "Distances sum squared differences, so the feature with the largest range decides who the neighbours are unless features are standardised."
  },
  {
    "q": "For the points labelled red at distances 0.5 and 1.6 and blue at distances 2.1, 2.9 and 3.2 from a query, what do k=3 and k=5 with uniform votes predict?",
    "options": ["red and red", "red and blue", "blue and red", "blue and blue"],
    "answer": 1,
    "why": "The 3 nearest are red, red, blue, so red wins. The 5 nearest are 2 red and 3 blue, so blue wins. The choice of k matters."
  },
  {
    "q": "Why does multinomial naive Bayes use Laplace smoothing?",
    "options": ["To make features independent", "To speed up counting", "To calibrate the output probabilities", "So that a word never seen in a class does not make that class's probability exactly zero"],
    "answer": 3,
    "why": "Without smoothing, one zero count zeroes the whole product, letting a single unseen word veto all other evidence."
  },
  {
    "q": "In gradient boosting with squared error loss, each new tree is fitted to what?",
    "options": ["The residuals y minus the current prediction", "The original targets y", "A bootstrap sample of the features", "The predictions of the previous tree"],
    "answer": 0,
    "why": "For squared error, the negative gradient with respect to the prediction is the residual, so each tree corrects what the ensemble still gets wrong."
  },
  {
    "q": "In XGBoost, a leaf has gradient sum G = -2, hessian sum H = 3 and lambda = 1. What is its optimal value?",
    "options": ["-0.5", "0.667", "0.5", "2"],
    "answer": 2,
    "why": "w* = -G / (H + lambda) = 2 / 4 = 0.5."
  },
  {
    "q": "Which pairing of library and characteristic idea is correct?",
    "options": ["LightGBM: symmetric oblivious trees; CatBoost: leaf-wise growth", "XGBoost: ordered target statistics; LightGBM: Newton leaf values only", "CatBoost: ordered target statistics for categories; LightGBM: leaf-wise growth with histograms", "XGBoost: no regularisation; CatBoost: no categorical support"],
    "answer": 2,
    "why": "CatBoost is built around leakage-free ordered target statistics and symmetric trees; LightGBM grows trees leaf-wise over histogram bins; XGBoost adds a regularised second-order objective."
  }
]
```
