This quiz covers the whole Probability and Statistics module: summary statistics, probability rules, Bayes' theorem, distributions, sampling, hypothesis testing, correlation and statistical significance. Aim for at least 13 out of 16 before moving on to Essential Mathematics for Machine Learning.

```quiz
[
  {
    "q": "Model latencies in milliseconds are `[110, 120, 115, 125, 3000]`. Which summary best describes a typical request?",
    "options": ["The mean, 694 ms", "The median, 120 ms", "The maximum, 3000 ms", "The range, 2890 ms"],
    "answer": 1,
    "why": "One extreme value drags the mean far from the bulk of the data. The median depends only on order, so it stays with the typical request."
  },
  {
    "q": "What does `statistics.stdev` compute that `statistics.pstdev` does not?",
    "options": ["The sample standard deviation, dividing by n - 1", "The variance instead of the standard deviation", "A standard deviation that ignores outliers", "The standard error of the mean"],
    "answer": 0,
    "why": "`stdev` divides by n - 1 (Bessel's correction) to estimate a population's spread from a sample; `pstdev` divides by n, treating the data as the whole population."
  },
  {
    "q": "Why must a `StandardScaler` be fitted on the training data only?",
    "options": ["Fitting on more data is slower", "The test set has no standard deviation", "Using test data to compute the mean and standard deviation leaks information about the test set into training", "scikit-learn raises an error otherwise"],
    "answer": 2,
    "why": "The scaling statistics are part of the model. Computing them from test data lets test information influence training, which makes evaluation over-optimistic."
  },
  {
    "q": "An agent completes a task in 8 independent steps, each succeeding with probability 0.9. What is the probability the whole task succeeds?",
    "options": ["0.9", "0.8", "About 0.72", "About 0.43"],
    "answer": 3,
    "why": "All 8 steps must succeed, so multiply: 0.9^8 ≈ 0.43."
  },
  {
    "q": "In a confusion matrix, which conditional probability is precision?",
    "options": ["P(actually positive | predicted positive)", "P(predicted positive | actually positive)", "P(predicted negative | actually negative)", "P(actually positive)"],
    "answer": 0,
    "why": "Precision conditions on the prediction: of everything predicted positive, what fraction was really positive. Recall conditions the other way."
  },
  {
    "q": "A disease affects 2% of people. A test has 90% sensitivity and a 10% false positive rate. Roughly what is P(disease | positive)?",
    "options": ["90%", "50%", "About 16%", "About 2%"],
    "answer": 2,
    "why": "Per 1,000 people: 20 are ill and 18 test positive; 980 are healthy and 98 test positive. 18 / (18 + 98) ≈ 0.155."
  },
  {
    "q": "What assumption makes Naive Bayes \"naive\"?",
    "options": ["All classes have equal priors", "Features (such as words) are conditionally independent given the class", "The data is normally distributed", "Every word appears in every document"],
    "answer": 1,
    "why": "Assuming conditional independence lets the likelihood of a whole document be written as a product of per-word probabilities, which is easy to estimate."
  },
  {
    "q": "With `exps = [math.exp(z) for z in [0.0, 0.0]]`, what is `[e / sum(exps) for e in exps]`?",
    "options": ["`[0.0, 0.0]`", "`[1.0, 1.0]`", "`[0.5, 0.5]`", "`[1.0, 0.0]`"],
    "answer": 2,
    "why": "This is softmax. Equal logits give equal exponentials (both 1.0), which normalise to a uniform distribution."
  },
  {
    "q": "Which distribution best models the number of requests an API receives per second, arriving independently at a steady average rate?",
    "options": ["Bernoulli", "Poisson", "Uniform", "Categorical"],
    "answer": 1,
    "why": "The Poisson distribution models counts of independent events in a fixed interval at a constant average rate."
  },
  {
    "q": "A metric is roughly normal with mean 50 and standard deviation 4. About what fraction of values fall between 42 and 58?",
    "options": ["68%", "95%", "99.7%", "50%"],
    "answer": 1,
    "why": "42 and 58 are two standard deviations either side of the mean, which covers about 95% of a normal distribution."
  },
  {
    "q": "Why do accuracy scores across many random test sets tend to look normally distributed, even though each example is scored 0 or 1?",
    "options": ["Because accuracy is always normal by definition", "Because test sets are chosen to be normal", "Because of the central limit theorem: accuracy is a mean of many independent 0/1 values", "Because of the law of total probability"],
    "answer": 2,
    "why": "The CLT says the mean of many independent values is approximately normal, whatever the distribution of the individual values."
  },
  {
    "q": "A sentiment dataset is built from product reviews, but most customers who were mildly satisfied never write one. What problem does this create?",
    "options": ["Sampling bias: the data over-represents strong opinions, and more reviews will not fix it", "Random sampling error that shrinks with more data", "Multicollinearity", "A Type II error"],
    "answer": 0,
    "why": "Self-selection is a systematic bias in how the sample was collected. Collecting more data the same way gives a more precise estimate of the wrong population."
  },
  {
    "q": "An A/B test of two model versions gives p = 0.20. What is the correct interpretation?",
    "options": ["The versions perform identically", "There is a 20% chance the new version is better", "The data does not provide convincing evidence of a difference; the test may lack power", "The new version is 20% better"],
    "answer": 2,
    "why": "Failing to reject the null hypothesis is not evidence that it is true. A confidence interval would show which effect sizes remain plausible."
  },
  {
    "q": "Which statement about Pearson correlation is true?",
    "options": ["A correlation of 0 means the variables are independent", "It changes if you convert a feature from metres to centimetres", "A high correlation proves one variable causes the other", "It measures the strength of a linear relationship on a scale from -1 to 1"],
    "answer": 3,
    "why": "Pearson's r is unit-free and bounded by -1 and 1. It misses curved relationships and says nothing about causation."
  },
  {
    "q": "Users who use a model's \"explain\" button churn less. Engagement drives both button use and retention. What is engagement here?",
    "options": ["A confounder", "A p-value", "A prior", "A stratum"],
    "answer": 0,
    "why": "A confounder influences both variables, creating a correlation between them even if neither causes the other."
  },
  {
    "q": "A team tries 30 hyperparameter settings, each tested against the baseline at alpha = 0.05, and reports only the one with p = 0.03. What is the main problem?",
    "options": ["The p-value should have been computed with a t-test", "Multiple comparisons: with 30 tries, a false positive below 0.05 is likely even if nothing helps", "Hyperparameters cannot be tested statistically", "0.03 is not below 0.05"],
    "answer": 1,
    "why": "With 30 independent useless settings, the chance of at least one p below 0.05 is 1 - 0.95^30 ≈ 0.79. Bonferroni would require p < 0.05 / 30 ≈ 0.0017."
  }
]
```
