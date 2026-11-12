A CI pipeline is only as trustworthy as the tests it runs. For an ordinary web service, "tests pass" means the code does what it should. For an AI service it is not enough: the code can be perfect while the training data has silently changed shape, the retrained model is worse than last week's, or a prompt tweak has made the LLM start inventing ticket numbers. Automated testing for AI systems therefore checks three things: the **code**, the **data** and the **model's behaviour**. This lesson shows how to organise those checks so they run automatically, catch real regressions and still keep CI fast.

You already know how to write tests with [pytest](/learn/software-engineering/pytest), and the difference between [unit tests](/learn/software-engineering/unit-testing) and [integration tests](/learn/software-engineering/integration-testing). Here we focus on what to automate, where it runs, and how to make model and LLM checks into pass-or-fail gates.

## The test pyramid for ML and LLM systems

The classic test pyramid says: many fast, cheap unit tests at the bottom, fewer slow integration tests above them, and a handful of end-to-end tests at the top. AI systems add layers that test data and model quality.

```mermaid
flowchart TD
    A["Smoke tests<br/>after each deploy, seconds"] --> B["LLM eval regression<br/>golden cases with thresholds"]
    B --> C["Model quality gates<br/>metrics on a fixed test set"]
    C --> D["Data validation<br/>schema, ranges, nulls, leakage"]
    D --> E["Integration tests<br/>app + database + fake LLM"]
    E --> F["Unit tests<br/>pure functions, many and fast"]
```

| Layer | What it checks in `ticket-triage` | Typical cost | When it runs |
| --- | --- | --- | --- |
| Unit | Text cleaning, label mapping, prompt building, Pydantic models | Milliseconds each | Every push |
| Integration | `/classify` end-to-end with a real database and a **stubbed** LLM | Seconds | Every pull request |
| Data validation | Training and eval data: columns, types, allowed labels, duplicates | Seconds | When data or pipeline code changes; before training |
| Model quality gate | Accuracy, F1 on a frozen test set against floors and the current production model | Minutes | When the model or training code changes |
| LLM eval regression | Golden prompts scored on correctness, format, refusals | Minutes and API cost | When prompts, model version or LLM code change; nightly |
| Smoke | `/health` and one real request against the deployed service | Seconds | After every deployment |

The shape matters because the top layers are slow, expensive and sometimes noisy. If you push every check into the top, CI takes an hour and nobody waits for it.

## Unit and integration tests in CI

Unit tests must not touch the network. The LLM client is a dependency, so inject it and replace it with a fake in tests (see [dependency injection](/learn/fastapi/dependency-injection) and [API testing](/learn/fastapi/api-testing)):

```python
from fastapi.testclient import TestClient
from app.main import app, get_llm_client

class FakeLLM:
    def summarise(self, text: str) -> str:
        return "Customer cannot log in after password reset."

def test_summarise_returns_summary():
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    client = TestClient(app)
    response = client.post("/summarise", json={"text": "I reset my password and now..."})
    assert response.status_code == 200
    assert response.json()["summary"].startswith("Customer")
    app.dependency_overrides.clear()
```

Integration tests need real collaborators such as PostgreSQL. GitHub Actions can start them as **service containers** next to the job:

```yaml
jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 5s
          --health-retries 10
    env:
      DATABASE_URL: postgresql://postgres:test@localhost:5432/postgres
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/integration -q
```

The health options make the job wait until the database actually accepts connections, a common source of "works locally, fails in CI".

## Data validation tests

Bad data is the most common cause of a bad model, and it does not raise exceptions. Write explicit checks and run them in CI whenever the data or the data pipeline changes, and again at the start of every training run.

```python
import pandas as pd

ALLOWED_LABELS = {"billing", "login", "bug", "other"}

def validate_tickets(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = {"ticket_id", "text", "label"} - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if df["ticket_id"].duplicated().any():
        errors.append(f"{df['ticket_id'].duplicated().sum()} duplicate ticket_id values")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        errors.append("empty ticket text")
    bad = set(df["label"]) - ALLOWED_LABELS
    if bad:
        errors.append(f"unknown labels: {sorted(bad)}")
    return errors

df = pd.DataFrame({
    "ticket_id": [1, 2, 2, 4],
    "text": ["Card declined", "Cannot log in", "Cannot log in", "  "],
    "label": ["billing", "login", "login", "Billing"],
})
for problem in validate_tickets(df):
    print(problem)
```

```output
1 duplicate ticket_id values
empty ticket text
unknown labels: ['Billing']
```

In a test file this becomes `assert validate_tickets(load_training_data()) == []`. Libraries such as Pandera or Great Expectations let you express the same rules declaratively; the principle is the same.

## Model quality gates

A **quality gate** turns a metric into a pass-or-fail decision. Two kinds of threshold are useful together:

- an **absolute floor**: the model must reach, say, macro F1 of at least 0.80 on the frozen test set, whatever happened before;
- a **relative check** against the current production model: the candidate must not be worse by more than a small tolerance.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                           n_classes=4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

candidate = LogisticRegression(max_iter=1000).fit(X_train, y_train)
f1 = f1_score(y_test, candidate.predict(X_test), average="macro")

F1_FLOOR = 0.60
PRODUCTION_F1 = 0.63      # read from the model registry in a real pipeline
TOLERANCE = 0.01

print(f"candidate macro F1 = {f1:.3f}")
print("passes floor:", f1 >= F1_FLOOR)
print("no regression:", f1 >= PRODUCTION_F1 - TOLERANCE)
```

```output
candidate macro F1 = 0.682
passes floor: True
no regression: True
```

The test set must be **frozen** (versioned, never trained on) or the gate is meaningless. Metrics come from the [model evaluation](/learn/model-evaluation/f1-score) module; the production number would come from your [model registry](/learn/mlops/model-registry).

## LLM eval regression with thresholds

LLM outputs are not deterministic strings you can compare with `==`. Instead, keep a **golden set** of cases (inputs plus what a good answer must contain or satisfy), score the current system on each, aggregate the scores per metric, and compare against floors and the last accepted baseline. Scoring can be exact checks (valid JSON, required fields), string or regex checks, or an LLM judge for qualities such as faithfulness.

The gate itself is ordinary Python:

```python
baseline  = {"faithfulness": 0.91, "json_valid": 0.99, "refuses_unsafe": 0.95}
candidate = {"faithfulness": 0.88, "json_valid": 1.00, "refuses_unsafe": 0.96}
floors    = {"faithfulness": 0.85, "json_valid": 0.98, "refuses_unsafe": 0.90}
MAX_DROP = 0.02

failures = []
for metric, score in candidate.items():
    if score < floors[metric]:
        failures.append(f"{metric}: {score:.2f} below floor {floors[metric]:.2f}")
    elif baseline[metric] - score > MAX_DROP:
        failures.append(f"{metric}: dropped {baseline[metric] - score:.2f} from baseline")

print("PASS" if not failures else "FAIL")
for f in failures:
    print(" ", f)
```

```output
FAIL
  faithfulness: dropped 0.03 from baseline
```

Step through the loop to see why only one metric fails:

```trace
{
  "title": "LLM eval gate",
  "code": [
    "failures = []",
    "for metric, score in candidate.items():",
    "    if score < floors[metric]:",
    "        failures.append('below floor')",
    "    elif baseline[metric] - score > MAX_DROP:",
    "        failures.append('regressed')",
    "print('PASS' if not failures else 'FAIL')"
  ],
  "steps": [
    { "line": 1, "vars": { "failures": "[]" }, "note": "Start with no failures. MAX_DROP is 0.02." },
    { "line": 3, "vars": { "metric": "'faithfulness'", "score": "0.88", "failures": "[]" }, "note": "0.88 is above the floor of 0.85, so the absolute check passes." },
    { "line": 6, "vars": { "metric": "'faithfulness'", "score": "0.88", "failures": "['regressed']" }, "note": "But the baseline was 0.91: a drop of 0.03, more than the allowed 0.02. Recorded as a regression." },
    { "line": 5, "vars": { "metric": "'json_valid'", "score": "1.00", "failures": "['regressed']" }, "note": "Above its floor and higher than the baseline of 0.99. Nothing recorded." },
    { "line": 5, "vars": { "metric": "'refuses_unsafe'", "score": "0.96", "failures": "['regressed']" }, "note": "Above floor, and improved on 0.95. Nothing recorded." },
    { "line": 7, "vars": { "failures": "['regressed']" }, "out": ["FAIL"], "note": "One failure, so the gate fails and the script should exit non-zero to fail the CI job." }
  ]
}
```

In CI, `run_evals.py` ends with `sys.exit(1)` when there are failures, which fails the step. Upload the per-case results as an artefact so a reviewer can see which cases got worse.

> [!WARNING]
> Eval scores are noisy. With 50 golden cases, one case flipping moves a metric by 0.02. Choose `MAX_DROP` with the size of your golden set in mind, use temperature 0 where the provider allows it, and grow the golden set over time from real failures in production.

## Smoke tests after deployment

A **smoke test** asks one question: is the thing we just deployed alive and wired up correctly? It runs against the real deployed URL immediately after a deployment:

```bash
curl --fail --silent --show-error "$SERVICE_URL/health"
curl --fail --silent --show-error -X POST "$SERVICE_URL/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "I was charged twice this month"}'
```

`--fail` makes `curl` exit non-zero on an HTTP error status, failing the step. If the smoke test fails, the pipeline triggers a rollback, covered in [automated deployment](/learn/ci-cd/automated-deployment).

## Keeping CI fast and trustworthy

Developers stop waiting for, and then stop trusting, slow CI. Some techniques:

- **Tier the suite.** Mark slow tests with `@pytest.mark.slow` and run `pytest -m "not slow"` on every push; run the full suite on pull requests to `main` or nightly.
- **Run only what changed.** Workflow `paths` filters can start the expensive eval job only when `app/llm.py`, `prompts/` or `evals/` changed.
- **Parallelise.** Split independent jobs (lint, unit, integration) so they run at the same time; `pytest-xdist` (`pytest -n auto`) spreads tests across CPU cores.
- **Cache dependencies** and use small fixtures instead of full datasets.

**Flaky tests** pass and fail on the same code. Common causes are unseeded randomness, real network calls, timing (`sleep` instead of waiting for a condition), test order dependence and shared state. A flaky test is worse than no test, because it teaches people to click "re-run" without reading failures. Quarantine it (mark and skip with a tracking issue), find the cause, and fix it. Do not paper over it with automatic retries forever.

> [!KEY]
> Every check in CI should be deterministic enough that a failure means something. For model and LLM gates, that means frozen test sets, fixed seeds, explicit tolerances and enough cases that noise does not dominate.

## Exercises

**1.** Extend `validate_tickets` so it also reports when any ticket text is longer than 5,000 characters.

<details>
<summary>Show solution</summary>

Add before `return errors`:

```python
too_long = (df["text"].str.len() > 5000).sum()
if too_long:
    errors.append(f"{too_long} tickets longer than 5000 characters")
```

</details>

**2.** Your LLM eval gate fails about one run in five on `main` with no code changes. Give two likely causes and a fix for each.

<details>
<summary>Show solution</summary>

Non-deterministic sampling (fix: temperature 0 or a fixed seed if the provider supports it, or average over several samples); a small golden set so one flipped case crosses `MAX_DROP` (fix: more cases, or a tolerance that reflects the noise); an LLM judge that is itself inconsistent (fix: stricter rubric, deterministic checks where possible). A changed model version behind an alias is another cause: pin the model version.

</details>

**3.** Decide which layer of the pyramid each check belongs to: (a) the `/health` endpoint returns 200 on the new deployment; (b) `clean_text` strips HTML tags; (c) no label outside the allowed set appears in training data.

<details>
<summary>Show solution</summary>

(a) smoke test, (b) unit test, (c) data validation.

</details>

```quiz
[
  {
    "q": "Why combine an absolute floor with a relative check against the production model in a quality gate?",
    "options": [
      "The floor stops a bad model even if production is also bad; the relative check stops gradual regressions above the floor",
      "Because CI tools require two thresholds",
      "The relative check is only for LLMs",
      "The floor makes training faster"
    ],
    "answer": 0,
    "why": "Each catches a failure the other misses: the floor sets a minimum standard, the relative check prevents quietly shipping something worse than what users already have."
  },
  {
    "q": "In unit tests for `/summarise`, how should the LLM be handled?",
    "options": [
      "Call the real API with a test key",
      "Skip the endpoint entirely",
      "Replace the client with a fake through dependency injection",
      "Run the LLM locally on the CI runner"
    ],
    "answer": 2,
    "why": "Unit tests must be fast, free and deterministic, so external services are replaced with fakes. Real LLM behaviour is checked separately by the eval suite."
  },
  {
    "q": "What is a smoke test?",
    "options": [
      "A load test that pushes the service to its limit",
      "A quick check against the freshly deployed service that it is alive and correctly wired",
      "A test that checks the model for bias",
      "A unit test with random inputs"
    ],
    "answer": 1,
    "why": "Smoke tests are small, fast checks run right after deployment; a failure usually triggers a rollback."
  },
  {
    "q": "What is the right response to a flaky test?",
    "options": [
      "Add automatic retries and move on",
      "Delete the test",
      "Increase the CI timeout",
      "Quarantine it with a tracking issue, find the cause and fix it"
    ],
    "answer": 3,
    "why": "Flaky tests erode trust in CI. Quarantining stops the noise while the root cause, often randomness, network or timing, is fixed."
  }
]
```
