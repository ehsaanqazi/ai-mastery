This quiz covers the whole MLOps module: the model lifecycle, tracking, versioning and registries, monitoring, drift and retraining, reproducibility, and the MLflow, Weights & Biases, DVC and Evidently tools, including their LLMOps uses. Aim for at least 13 out of 16 before moving on to CI/CD.

```quiz
[
  {
    "q": "Which stage of the model lifecycle feeds back into data collection and retraining?",
    "options": ["Register", "Monitor", "Evaluate", "Deploy"],
    "answer": 1,
    "why": "Monitoring detects degradation or new conditions in production and triggers a new pass through data and training."
  },
  {
    "q": "Which set of information best defines a reproducible model version?",
    "options": [
      "The weights file and its file name",
      "Code commit, data version, config, environment and the weights' content hash",
      "The accuracy on the test set",
      "The name of the person who trained it"
    ],
    "answer": 1,
    "why": "A version must link to everything that produced it; the weights alone cannot be explained or rebuilt."
  },
  {
    "q": "A serving application loads `models:/fraud@champion`. How do you roll back to the previous version?",
    "options": [
      "Retrain the previous model",
      "Edit the serving code to point at the old file",
      "Move the `champion` alias back to the previous version in the registry",
      "Delete the newest version"
    ],
    "answer": 2,
    "why": "Aliases are movable pointers; rollback is a registry change, and immutable versions mean the old one still exists."
  },
  {
    "q": "In LLMOps, which artefact plays the role of a regression test suite for prompt and model changes?",
    "options": ["The system prompt", "A versioned eval set with scoring", "The token counter", "The API key"],
    "answer": 1,
    "why": "Every prompt or model change is scored against the same versioned eval set, so regressions show up before release."
  },
  {
    "q": "What is the two-sample KS statistic?",
    "options": [
      "The difference in means divided by the pooled standard deviation",
      "The largest absolute gap between the two empirical CDFs",
      "The sum of squared differences between histograms",
      "The share of rows with missing values"
    ],
    "answer": 1,
    "why": "$D = \\sup_t |F_{ref}(t) - F_{cur}(t)|$, which reacts to shifts in location, spread and shape."
  },
  {
    "q": "Given bin shares $e = [0.5, 0.5]$ (reference) and $a = [0.5, 0.5]$ (current), what is the PSI?",
    "options": ["1", "0.5", "0", "It is undefined"],
    "answer": 2,
    "why": "Each term is $(a_i - e_i) \\ln(a_i / e_i)$, and identical shares make every term zero."
  },
  {
    "q": "With 500,000 rows per window, a KS test gives p < 1e-10 but PSI is 0.003. What should you conclude?",
    "options": [
      "Severe drift: roll back immediately",
      "A statistically detectable but practically negligible difference",
      "The test is broken",
      "The labels have changed"
    ],
    "answer": 1,
    "why": "Large samples make tiny differences significant; the effect size shows the change is negligible."
  },
  {
    "q": "Fraudsters change tactics so that transactions that used to be safe are now fraudulent, while the input distribution looks the same. What is this?",
    "options": ["Covariate shift", "Label shift", "Concept drift", "Data leakage"],
    "answer": 2,
    "why": "The relationship $P(y \\mid x)$ changed; input-based drift detectors cannot see it, only labels can."
  },
  {
    "q": "Labels for a loan model arrive 12 months after each decision. Which design choice makes it possible to measure accuracy later?",
    "options": [
      "Logging every prediction with a request id and model version",
      "Retraining every day",
      "Raising the decision threshold",
      "Disabling drift detection"
    ],
    "answer": 0,
    "why": "The request id lets outcomes be joined back to predictions, and the version attributes them to the right model."
  },
  {
    "q": "Why is p95 or p99 latency preferred over mean latency for a model service?",
    "options": [
      "It is always lower",
      "It captures the slow tail that the mean and median hide",
      "It needs no logging",
      "It measures model accuracy"
    ],
    "answer": 1,
    "why": "Latency distributions are long-tailed; percentiles show what the slowest users experience."
  },
  {
    "q": "What does shadow deployment let you measure that offline evaluation cannot, with zero user impact?",
    "options": [
      "The challenger's effect on user clicks",
      "The challenger's latency, errors and prediction differences on real traffic",
      "The champion's training loss",
      "The cost of labelling"
    ],
    "answer": 1,
    "why": "The challenger processes copies of live requests but its outputs are not shown, so system behaviour is observed without risk."
  },
  {
    "q": "Read this routing function. What property does it guarantee?\n\n`bucket = int(sha256(user_id).hexdigest(), 16) % 100; return \"challenger\" if bucket < 5 else \"champion\"`",
    "options": [
      "Exactly 5 users get the challenger",
      "Each user is consistently routed to the same model, with about 5% on the challenger",
      "Users are re-randomised on every request",
      "The challenger receives the slowest requests"
    ],
    "answer": 1,
    "why": "The hash of a user id never changes, so assignment is sticky, and buckets are spread roughly uniformly."
  },
  {
    "q": "Why can two GPU training runs with the same seed produce slightly different models?",
    "options": [
      "GPUs ignore seeds",
      "Parallel reductions add floating-point numbers in varying orders, and floating-point addition is not associative",
      "The dataset is reshuffled by the driver",
      "Seeds expire after one run"
    ],
    "answer": 1,
    "why": "Non-deterministic operation order changes rounding; deterministic settings help but cost speed and may not hold across hardware."
  },
  {
    "q": "In recent MLflow 3 releases, which tracking URI works for a simple local setup without extra opt-in?",
    "options": ["`file:./mlruns`", "`sqlite:///mlflow.db`", "`git://localhost`", "`s3://bucket` only"],
    "answer": 1,
    "why": "The file-based tracking backend is in maintenance mode and requires an explicit opt-in; a SQLite database works directly."
  },
  {
    "q": "After `dvc add data/train.csv`, a teammate clones the repo and the CSV is missing. What must they run?",
    "options": ["`dvc init`", "`git pull`", "`dvc pull`", "`dvc repro --force`"],
    "answer": 2,
    "why": "Git only holds the pointer; `dvc pull` fetches the content from the DVC remote, provided it was pushed."
  },
  {
    "q": "Which statement about the tools in this module is accurate?",
    "options": [
      "Evidently provides a model registry with aliases",
      "W&B can run offline with `mode=\"offline\"` and upload later with `wandb sync`",
      "DVC replaces Git for versioning code",
      "MLflow cannot store prompts"
    ],
    "answer": 1,
    "why": "W&B supports offline mode and later syncing. Evidently has no registry, DVC works alongside Git, and MLflow has a prompt registry."
  }
]
```
