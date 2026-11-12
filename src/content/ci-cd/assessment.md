This quiz covers the whole CI/CD module: GitHub Actions, automated testing for ML and LLM systems, continuous integration, continuous delivery and deployment, automated deployment strategies, and end-to-end pipelines including continuous training. Aim for at least 12 out of 16 before moving on to Cloud Computing.

```quiz
[
  {
    "q": "In GitHub Actions, what is the relationship between a job and a step?",
    "options": [
      "A step contains several jobs that run in parallel",
      "A job is a group of steps that run in order on one runner",
      "Jobs and steps are interchangeable names",
      "A step runs on its own fresh virtual machine"
    ],
    "answer": 1,
    "why": "Steps inside a job run sequentially and share the runner's file system; separate jobs run on separate runners."
  },
  {
    "q": "What does `cache: pip` on `actions/setup-python` do?",
    "options": [
      "Saves and restores pip's download cache, keyed on the dependency files, to speed up installs",
      "Uploads installed packages as a downloadable artefact",
      "Pins every package to its latest version",
      "Installs packages without network access"
    ],
    "answer": 0,
    "why": "The cache key is a hash of the dependency files, so the cache is reused until they change."
  },
  {
    "q": "A deploy job logs in to a cloud with OIDC. Which permission must the job have?",
    "options": [
      "`packages: write`",
      "`pull-requests: write`",
      "`id-token: write`",
      "`contents: write`"
    ],
    "answer": 2,
    "why": "`id-token: write` allows the job to request the signed OIDC token that the cloud exchanges for short-lived credentials."
  },
  {
    "q": "A workflow sets the concurrency group to `${{ github.workflow }}-${{ github.ref }}` with `cancel-in-progress: true`. What happens when a new commit is pushed to a pull request whose previous run is still in progress?",
    "options": [
      "Both runs continue in parallel",
      "The new run waits until the old one finishes",
      "The new run is cancelled",
      "The in-progress run for the same workflow and ref is cancelled and the new one runs"
    ],
    "answer": 3,
    "why": "Runs share a group per workflow and ref; `cancel-in-progress: true` cancels the older run in favour of the latest."
  },
  {
    "q": "Which test layer would catch training data containing the label `Billing` when only lowercase `billing` is allowed?",
    "options": [
      "Smoke test",
      "Data validation",
      "Load test",
      "Linting"
    ],
    "answer": 1,
    "why": "Data validation checks schema, allowed values, nulls and duplicates in the data itself."
  },
  {
    "q": "An LLM eval gate uses a floor of 0.85 and a maximum drop of 0.02 from a baseline of 0.90. The candidate scores 0.87. What is the verdict?",
    "options": [
      "Fail: it dropped 0.03 from the baseline",
      "Pass: it is above the floor",
      "Pass: drops below 0.05 are always ignored",
      "Fail: it is below the floor"
    ],
    "answer": 0,
    "why": "0.87 clears the floor, but the 0.03 drop exceeds the allowed 0.02, so the relative check fails."
  },
  {
    "q": "Why should unit tests for an LLM-backed endpoint use a fake LLM client?",
    "options": [
      "The real API cannot be called from Python",
      "Fakes produce better summaries",
      "Unit tests must be fast, free and deterministic; real model behaviour is checked by the eval suite",
      "GitHub Actions blocks outbound network access"
    ],
    "answer": 2,
    "why": "Separating code correctness (unit tests with fakes) from model behaviour (evals) keeps each check reliable."
  },
  {
    "q": "Which practice best fits continuous integration?",
    "options": [
      "Feature branches that live for a month and merge before release",
      "Short-lived branches merged to `main` at least daily, each verified by automated checks",
      "Testing only on the release branch",
      "Merging without review when CI is slow"
    ],
    "answer": 1,
    "why": "Trunk-based development keeps integrations small and frequent, which is the core of CI."
  },
  {
    "q": "What does a required status check in branch protection do?",
    "options": [
      "Runs the check on a schedule",
      "Hides failing checks from reviewers",
      "Automatically fixes lint errors",
      "Blocks merging a pull request until the named check passes"
    ],
    "answer": 3,
    "why": "Required checks turn CI from advice into an enforced rule on the protected branch."
  },
  {
    "q": "A team deploys every change that passes the pipeline to production automatically, with no human step. What are they practising?",
    "options": [
      "Continuous deployment",
      "Continuous delivery",
      "Continuous integration only",
      "Blue-green deployment"
    ],
    "answer": 0,
    "why": "Continuous deployment removes the human decision; continuous delivery keeps changes releasable but a person chooses when."
  },
  {
    "q": "Why deploy an image by digest rather than by a tag such as `latest`?",
    "options": [
      "Digests download faster",
      "A digest identifies exact, immutable image content, so staging and production run the same bytes",
      "Tags cannot be used with Kubernetes",
      "Digests include the model weights automatically"
    ],
    "answer": 1,
    "why": "Tags can be moved; a digest cannot. This makes promotion and rollback exact."
  },
  {
    "q": "During a rolling deployment, why must database migrations be backward-compatible?",
    "options": [
      "Migrations run after all old instances stop",
      "Databases reject non-compatible migrations",
      "Old and new versions of the code run against the same database at the same time",
      "Backward-compatible migrations are faster"
    ],
    "answer": 2,
    "why": "Expand and contract keeps each step safe while both versions are live, and keeps code rollbacks possible."
  },
  {
    "q": "A canary has handled 1,000 requests with an error rate three times the stable version's, and similar latency. What should an automated canary check do?",
    "options": [
      "Promote, because latency is fine",
      "Wait for more traffic indefinitely",
      "Increase the canary to 50% to gather data faster",
      "Roll back by sending no traffic to the canary"
    ],
    "answer": 3,
    "why": "The point of a canary is to limit exposure; a clear error-rate regression triggers rollback."
  },
  {
    "q": "Why do many teams version and deploy model artefacts separately from the application image?",
    "options": [
      "Models retrain on a different schedule from code, and separate versioning allows model-only promotion and rollback",
      "Docker images cannot contain binary files",
      "Model registries are required by GitHub Actions",
      "It removes the need to evaluate models"
    ],
    "answer": 0,
    "why": "Decoupling means a retrain does not need a code release, and a bad model can be rolled back by moving an alias."
  },
  {
    "q": "Your service calls an LLM through a provider alias that can be updated without notice. What change makes model upgrades go through your eval gate?",
    "options": [
      "Raising the eval floor",
      "Pinning the exact model version in configuration, so an upgrade becomes a commit",
      "Running unit tests twice",
      "Removing the canary stage"
    ],
    "answer": 1,
    "why": "Once the version is pinned, changing it is a code or config change that triggers CI and the eval gate. A scheduled eval catches remaining drift."
  },
  {
    "q": "How does continuous training (CT) relate to CI/CD?",
    "options": [
      "CT replaces CI/CD for ML projects",
      "CT deploys code changes without tests",
      "CI/CD tests and ships the training code; CT runs it on new data to produce model versions that pass gates before promotion",
      "CT is only used for LLMs"
    ],
    "answer": 2,
    "why": "The training pipeline is code, shipped through CI/CD; CT then runs it automatically, with evaluation gates and the same promotion machinery."
  }
]
```
