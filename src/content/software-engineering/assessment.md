This quiz covers the whole Software Engineering module: clean code, architecture, design patterns, logging, testing with pytest, error monitoring, Git branching and code review, all in the context of AI services. Aim for at least 13 out of 16 before moving on to Application Development.

```quiz
[
  {
    "q": "A function creates its own OpenAI client, reads `os.environ`, builds a prompt, calls the model and parses JSON. Which refactoring step most improves its testability?",
    "options": [
      "Adding a docstring",
      "Passing the model client (or a completion function) in as a parameter and extracting prompt building and parsing into pure functions",
      "Wrapping everything in `try`/`except Exception`",
      "Renaming the function"
    ],
    "answer": 1,
    "why": "Injected dependencies and pure functions can be tested without network calls. This is dependency inversion and single responsibility in practice."
  },
  {
    "q": "What does low coupling mean?",
    "options": [
      "A module contains many unrelated functions",
      "A unit depends on few, narrow interfaces of other units rather than their internals",
      "Every class has exactly one method",
      "Modules never import each other"
    ],
    "answer": 1,
    "why": "Low coupling means changes in one unit rarely force changes in another, because they interact through small contracts."
  },
  {
    "q": "In a ports and adapters architecture, where does the code that converts an Anthropic or OpenAI response into a plain string belong?",
    "options": ["The domain core", "An adapter", "The API routes", "The test suite"],
    "answer": 1,
    "why": "Adapters translate between a specific technology and the port the domain defines, keeping SDK details out of the core."
  },
  {
    "q": "Which file is the only place that should know about every concrete adapter?",
    "options": [
      "The domain service",
      "`utils.py`",
      "The composition root that builds and wires objects at start-up",
      "Each FastAPI route"
    ],
    "answer": 2,
    "why": "Centralising construction in the composition root lets the rest of the code depend on abstractions."
  },
  {
    "q": "You need to add retries with exponential backoff to several LLM-calling functions without editing their bodies. Which pattern fits?",
    "options": ["Repository", "Decorator", "Observer", "Factory"],
    "answer": 1,
    "why": "A decorator wraps a function to add behaviour around it, which suits cross-cutting concerns like retries, caching and timing."
  },
  {
    "q": "What does this code print?\n\n```python\nfrom functools import lru_cache\ncalls = 0\n@lru_cache(maxsize=None)\ndef embed(t):\n    global calls\n    calls += 1\n    return len(t)\nembed(\"a\"); embed(\"b\"); embed(\"a\")\nprint(calls)\n```",
    "options": ["1", "2", "3", "0"],
    "answer": 1,
    "why": "The second call with \"a\" is served from the cache, so the function body runs only for \"a\" and \"b\"."
  },
  {
    "q": "Why are structured (JSON) logs preferable to free-text log messages in a production AI service?",
    "options": [
      "They are shorter",
      "Fields such as `model`, `latency_ms` and `request_id` can be indexed and queried directly",
      "They cannot contain errors",
      "They do not need log levels"
    ],
    "answer": 1,
    "why": "Structured fields make logs machine-queryable for dashboards, cost reports and investigations."
  },
  {
    "q": "Which of these should never be written to application logs?",
    "options": [
      "A request's correlation ID",
      "Input and output token counts",
      "The user's API key from the Authorization header",
      "The prompt version used"
    ],
    "answer": 2,
    "why": "Secrets in logs are effectively leaked. A key that reaches the logs must be rotated."
  },
  {
    "q": "A `Mock` has `side_effect = [TimeoutError(), \"ok\"]`. What happens on the first and second calls?",
    "options": [
      "Both return \"ok\"",
      "The first raises `TimeoutError`; the second returns \"ok\"",
      "Both raise `TimeoutError`",
      "The first returns the list"
    ],
    "answer": 1,
    "why": "An iterable side effect supplies one item per call, raising items that are exceptions and returning the others."
  },
  {
    "q": "Which test double is a working, simplified implementation, such as an in-memory document store?",
    "options": ["Dummy", "Stub", "Fake", "Mock"],
    "answer": 2,
    "why": "Fakes behave realistically with a simpler implementation. Stubs return canned answers; mocks record calls for assertions."
  },
  {
    "q": "Why is RAG retrieval quality best checked by an integration test rather than a unit test?",
    "options": [
      "It requires the real embedding model, chunking and index working together on real data",
      "Unit tests cannot use assertions",
      "Integration tests are faster",
      "Retrieval never changes"
    ],
    "answer": 0,
    "why": "Whether the right documents come back is a property of the real components combined, which a fake index cannot reveal."
  },
  {
    "q": "In pytest, a fixture with `scope=\"session\"` returns a dictionary that one test modifies. What is the risk?",
    "options": [
      "pytest raises an error",
      "Later tests see the modified dictionary, so results depend on test order",
      "The fixture is recreated for every test anyway",
      "Coverage drops to zero"
    ],
    "answer": 1,
    "why": "Session fixtures are shared across the whole run. Shared mutable state couples tests together."
  },
  {
    "q": "Which pytest feature runs the same test body over a table of inputs, reporting each case separately?",
    "options": ["`@pytest.fixture`", "`@pytest.mark.parametrize`", "`monkeypatch`", "`--setup-show`"],
    "answer": 1,
    "why": "Parametrize generates one test per row of arguments, each with its own ID and result."
  },
  {
    "q": "Your SLO is 99.9% of requests succeeding over 30 days. Roughly how many minutes of full outage does the error budget allow?",
    "options": ["4.3", "43.2", "432", "7.2"],
    "answer": 1,
    "why": "0.1% of 43,200 minutes in 30 days is 43.2 minutes."
  },
  {
    "q": "Why should you avoid rebasing a branch that teammates have already pulled?",
    "options": [
      "Rebase deletes the branch",
      "Rebase rewrites commits with new hashes, so teammates' copies diverge from the rewritten history",
      "Rebase only works on `main`",
      "Rebase disables CI"
    ],
    "answer": 1,
    "why": "Rewriting shared history forces others to reconcile two versions of the same commits."
  },
  {
    "q": "A pull request changes the answer prompt. Overall eval accuracy rises slightly, but one category drops sharply on a small eval set. What is the best review response?",
    "options": [
      "Approve, because the overall number improved",
      "Reject the PR permanently",
      "Ask for the failing examples in that category and a larger evaluation before approving",
      "Ask the author to reformat the prompt"
    ],
    "answer": 2,
    "why": "Headline averages can hide regressions, and small eval sets give noisy differences. Inspecting failures and gathering more evidence is the right step."
  }
]
```
