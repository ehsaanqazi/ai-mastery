This quiz covers the whole module: loss functions, optimisation algorithms, maximum likelihood estimation, entropy, cross-entropy and KL divergence. Aim for at least 13 out of 16 before moving on to NumPy.

```quiz
[
  {
    "q": "A regression dataset has a handful of grossly wrong labels. Which loss keeps them from dominating the gradient while staying smooth near zero error?",
    "options": [
      "MSE",
      "Huber",
      "Hinge",
      "Categorical cross-entropy"
    ],
    "answer": 1,
    "why": "Huber is quadratic for small residuals and linear for large ones, so outliers contribute a bounded gradient."
  },
  {
    "q": "Minimising mean absolute error makes a regression model predict which statistic of the target?",
    "options": [
      "The conditional mean",
      "The conditional mode",
      "The conditional median",
      "The conditional variance"
    ],
    "answer": 2,
    "why": "Absolute error is minimised by the median; squared error by the mean."
  },
  {
    "q": "With labels in {-1, +1}, which example has zero hinge loss?",
    "options": [
      "y = +1, score = 1.5",
      "y = +1, score = 0.5",
      "y = -1, score = 0.2",
      "y = -1, score = -0.5"
    ],
    "answer": 0,
    "why": "Hinge loss is max(0, 1 - y·s). Only a margin y·s of at least 1 gives zero: 1 × 1.5 = 1.5."
  },
  {
    "q": "Plain gradient descent on f(w) = 25w² (so f''(w) = 50) is stable only if the learning rate is below what value?",
    "options": [
      "0.5",
      "0.02",
      "0.1",
      "0.04"
    ],
    "answer": 3,
    "why": "The update multiplies w by (1 - 50η), which must have magnitude below 1, so η < 2/50 = 0.04."
  },
  {
    "q": "Which optimiser keeps running averages of both the gradient and the squared gradient, with bias correction?",
    "options": [
      "Adam",
      "SGD with momentum",
      "RMSProp",
      "Plain SGD"
    ],
    "answer": 0,
    "why": "Adam combines a momentum-like first moment with an RMSProp-like second moment and corrects both for their zero initialisation."
  },
  {
    "q": "Why is a linear learning-rate warm-up commonly used when training transformers with Adam?",
    "options": [
      "It reduces memory use",
      "Early second-moment estimates are unreliable and freshly initialised networks are fragile, so large early steps can diverge",
      "It removes the need for weight decay",
      "It makes the final learning rate higher"
    ],
    "answer": 1,
    "why": "Starting with small steps while the optimiser's statistics settle avoids early blow-ups."
  },
  {
    "q": "In this code, what is `m_hat` on the first step?\n\n`g = 4.0; m = 0.9 * 0.0 + 0.1 * g; m_hat = m / (1 - 0.9 ** 1)`",
    "options": [
      "0.4",
      "0.36",
      "4.0",
      "40.0"
    ],
    "answer": 2,
    "why": "m = 0.4 and 1 - 0.9 = 0.1, so m_hat = 0.4 / 0.1 = 4.0: bias correction recovers the gradient exactly on step 1."
  },
  {
    "q": "You flip a coin 20 times and see 15 heads. What is the maximum likelihood estimate of the probability of heads?",
    "options": [
      "0.5",
      "0.15",
      "0.25",
      "0.75"
    ],
    "answer": 3,
    "why": "For a Bernoulli model the MLE is the observed fraction, k/n = 15/20."
  },
  {
    "q": "Assuming Gaussian noise with fixed variance, maximising the likelihood of a regression model is equivalent to:",
    "options": [
      "Minimising the sum of squared residuals",
      "Minimising the sum of absolute residuals",
      "Maximising the number of exact predictions",
      "Minimising hinge loss"
    ],
    "answer": 0,
    "why": "The Gaussian negative log-likelihood is a constant plus the sum of squared residuals divided by 2σ²."
  },
  {
    "q": "Adding an L2 penalty on the weights to a negative log-likelihood is equivalent to which estimate?",
    "options": [
      "Unbiased estimation",
      "MAP estimation with a zero-mean Gaussian prior on the weights",
      "MLE with Laplace noise",
      "MAP estimation with a uniform prior"
    ],
    "answer": 1,
    "why": "MAP adds -log p(θ) to the NLL, and for a Gaussian prior that term is proportional to the squared norm of θ."
  },
  {
    "q": "What is the entropy of a fair eight-sided die, in bits?",
    "options": [
      "8",
      "2",
      "3",
      "2.079"
    ],
    "answer": 2,
    "why": "A uniform distribution over 8 outcomes has entropy log2(8) = 3 bits (2.079 is the value in nats)."
  },
  {
    "q": "A training run logs a cross-entropy of 0.6931 nats. What is that in bits?",
    "options": [
      "0.48",
      "0.6931",
      "1.4427",
      "1.0"
    ],
    "answer": 3,
    "why": "Divide nats by ln 2 ≈ 0.6931 to get bits: 0.6931 / 0.6931 = 1 bit."
  },
  {
    "q": "A classifier gives the correct class a probability of 0.25. What is the cross-entropy loss for that example in bits?",
    "options": [
      "2",
      "0.25",
      "1.386",
      "4"
    ],
    "answer": 0,
    "why": "-log2(0.25) = 2 bits (or about 1.386 nats)."
  },
  {
    "q": "A language model's validation cross-entropy is ln(12) nats per token. What is its perplexity?",
    "options": [
      "ln 12",
      "12",
      "144",
      "Cannot be determined without the vocabulary size"
    ],
    "answer": 1,
    "why": "Perplexity is exp of the cross-entropy in nats, so exp(ln 12) = 12: as uncertain as a uniform choice among 12 tokens."
  },
  {
    "q": "Which statement about KL divergence is false?",
    "options": [
      "It is never negative",
      "It is zero when the two distributions are identical",
      "It can be infinite if q assigns zero probability where p does not",
      "It is symmetric in its two arguments"
    ],
    "answer": 3,
    "why": "KL(p || q) and KL(q || p) generally differ, which is why the direction is a modelling choice."
  },
  {
    "q": "In the RLHF objective E[r(x, y)] - β·KL(π || π_ref), what happens as β becomes very large?",
    "options": [
      "The policy stays almost identical to the reference model",
      "The policy ignores the reference and maximises reward only",
      "The reward model is retrained",
      "The KL estimate becomes negative"
    ],
    "answer": 0,
    "why": "A large β makes any drift from π_ref very costly, so the policy learns little from the reward. β = 0 removes the constraint entirely."
  }
]
```
