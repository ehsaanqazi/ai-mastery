This quiz covers the whole FastAPI module: running an app, routing, Pydantic, request validation, response models, dependency injection, authentication, background tasks, async endpoints and testing. Aim for at least 13 out of 16 before moving on to Software Engineering.

```quiz
[
  {
    "q": "What does `--reload` do in `uvicorn main:app --reload`, and where should it be used?",
    "options": [
      "Reloads the model on every request; use it in production",
      "Restarts the server when source files change; use it only in development",
      "Caches responses between requests; use it everywhere",
      "Runs several worker processes; use it in production"
    ],
    "answer": 1,
    "why": "`--reload` watches the file system and restarts on changes, which is convenient while coding and wasteful and risky in production."
  },
  {
    "q": "Given `@app.get(\"/models/{name}\")` and `def get(name: str, limit: int = 5)`, a request to `/models/v2?limit=abc` returns what?",
    "options": [
      "200 with `limit` set to 5",
      "404 Not Found",
      "422, with an error located at `['query', 'limit']`",
      "500 Internal Server Error"
    ],
    "answer": 2,
    "why": "`limit` is a query parameter of type `int`; `abc` cannot be parsed, so validation fails with 422 before the function runs."
  },
  {
    "q": "A router is declared as `APIRouter(prefix=\"/jobs\", tags=[\"jobs\"])` and included with `app.include_router(router, prefix=\"/v2\")`. What is the path of `@router.get(\"/{job_id}\")`?",
    "options": ["`/jobs/{job_id}`", "`/v2/{job_id}`", "`/{job_id}`", "`/v2/jobs/{job_id}`"],
    "answer": 3,
    "why": "Include prefix, router prefix and route path are joined in that order."
  },
  {
    "q": "Which Pydantic v2 code correctly converts a model instance `m` into a JSON string?",
    "options": ["`m.json()`", "`m.model_dump_json()`", "`json.dumps(m)`", "`m.to_json()`"],
    "answer": 1,
    "why": "`model_dump_json()` is the v2 method. `.json()` is the deprecated v1 name, and `json.dumps` cannot serialise a model directly."
  },
  {
    "q": "Which tool checks that `end_date` is after `start_date` in a Pydantic model?",
    "options": [
      "`Field(gt=start_date)`",
      "`ConfigDict(extra=\"forbid\")`",
      "A `@model_validator(mode=\"after\")`",
      "A `Query(ge=...)` marker"
    ],
    "answer": 2,
    "why": "Cross-field rules belong in a model validator, which runs after each field has been validated individually."
  },
  {
    "q": "With `class PredictRequest(BaseModel): model_config = ConfigDict(extra=\"forbid\"); text: str`, what happens to a body `{\"text\": \"hi\", \"temprature\": 0.2}`?",
    "options": [
      "422: `temprature` is an extra input that is not permitted",
      "The typo is silently ignored",
      "`temprature` is renamed to the closest field",
      "500 Internal Server Error"
    ],
    "answer": 0,
    "why": "`extra=\"forbid\"` turns unknown fields into validation errors, which surfaces client typos immediately."
  },
  {
    "q": "How does FastAPI read the parameter `x_api_version: Annotated[str, Header()] = \"1\"`?",
    "options": [
      "From the query parameter `x_api_version`",
      "From a JSON body field",
      "From a cookie",
      "From the `X-API-Version` request header"
    ],
    "answer": 3,
    "why": "Header parameters convert underscores to hyphens, and header names are case-insensitive."
  },
  {
    "q": "An endpoint declares `response_model=PublicPrediction` but returns an object with extra internal fields. What reaches the client?",
    "options": [
      "Only the fields declared in `PublicPrediction`",
      "All fields, since the object is valid",
      "A 422 error",
      "Nothing; the response is empty"
    ],
    "answer": 0,
    "why": "Response models validate and filter output, so undeclared fields are dropped."
  },
  {
    "q": "Deep inside model code you detect that the model is not loaded. What is the cleanest way to return 503 to the client?",
    "options": [
      "Return the string `\"503\"` from the model function",
      "Raise a custom `ModelNotReadyError` and map it to a 503 response with `@app.exception_handler`",
      "Call `sys.exit(503)`",
      "Return `None` and let FastAPI decide"
    ],
    "answer": 1,
    "why": "A domain exception keeps model code free of HTTP concerns, and a single handler controls the response format and headers."
  },
  {
    "q": "A yield dependency prints `open`, then does `yield db` inside a `try` block whose `finally` prints `close`. The endpoint that uses it prints `handler`. In what order do the three lines appear for one request?",
    "options": [
      "open, close, handler",
      "handler, open, close",
      "open, handler, close",
      "open, handler (close only at shutdown)"
    ],
    "answer": 2,
    "why": "Code before `yield` runs first, the endpoint runs with the yielded value, then the cleanup after `yield` runs."
  },
  {
    "q": "Where should a large model be loaded so every request reuses it?",
    "options": [
      "In the `lifespan` function at startup, stored on `app.state`",
      "Inside each endpoint",
      "In a dependency with `use_cache=False`",
      "In a background task"
    ],
    "answer": 0,
    "why": "`lifespan` runs once at startup. The deprecated `on_event(\"startup\")` hook did the same job in older code."
  },
  {
    "q": "A user with a valid token but without the admin role calls an admin route. A missing token on the same route should return a different status. Which pair is correct?",
    "options": [
      "Valid token without role: 401; missing token: 403",
      "Valid token without role: 404; missing token: 401",
      "Both: 403",
      "Valid token without role: 403; missing token: 401"
    ],
    "answer": 3,
    "why": "401 means not authenticated; 403 means authenticated but not permitted."
  },
  {
    "q": "Which statement about JWT access tokens is true?",
    "options": [
      "Their payload is encrypted, so it is safe to store passwords in it",
      "Their payload can be read by anyone; the signature only prevents undetected tampering",
      "They never expire unless revoked",
      "The server must look them up in a database on every request"
    ],
    "answer": 1,
    "why": "JWTs are signed, not encrypted. Keep secrets out of them and keep expiry short."
  },
  {
    "q": "Which job is the best fit for FastAPI's `BackgroundTasks`?",
    "options": [
      "A two-hour GPU fine-tuning run",
      "A billing record that must never be lost",
      "Appending a prediction to a monitoring log after responding",
      "A batch job that other services poll for status"
    ],
    "answer": 2,
    "why": "Background tasks are best-effort and in-process: no persistence, retries or shared status. Durable or heavy work belongs in a job queue with workers."
  },
  {
    "q": "An `async def` endpoint calls a synchronous LLM SDK that blocks for 3 seconds. What is the effect, and a correct fix?",
    "options": [
      "Only that request is slow; no fix needed",
      "FastAPI automatically runs it in a thread; no fix needed",
      "The request times out at 1 second; increase the timeout",
      "The event loop is blocked so all requests stall; use the SDK's async client or declare the endpoint with plain `def`"
    ],
    "answer": 3,
    "why": "Blocking calls inside `async def` hold the event loop thread. Await an async client, offload with `run_in_threadpool`, or use `def` so FastAPI runs it in its thread pool."
  },
  {
    "q": "In a pytest fixture, why is `app.dependency_overrides.clear()` placed after `yield` rather than at the end of each test?",
    "options": [
      "Teardown after `yield` runs even if the test fails, so overrides cannot leak into later tests",
      "pytest forbids clearing overrides inside tests",
      "Overrides only take effect after `yield`",
      "It makes the tests run in parallel"
    ],
    "answer": 0,
    "why": "If an assertion fails, the rest of the test body is skipped, but fixture teardown still runs."
  }
]
```
