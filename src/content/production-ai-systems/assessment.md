This quiz covers the whole Production AI Systems module: serving architectures, inference optimisation, quantisation, batching, caching, GPU utilisation, latency, cost, scaling, load balancing, observability and reliability. Aim for at least 13 out of 16 before moving on to Advanced LLM Engineering.

```quiz
[
  {
    "q": "Predictions for every product are needed once a day for an email campaign. Which serving pattern fits best?",
    "options": ["Online serving behind a GPU server", "Batch serving on a schedule", "Streaming serving from an event stream", "Edge serving on users' devices"],
    "answer": 1,
    "why": "The inputs are known in advance and nobody waits for individual predictions, so a scheduled batch job is simplest and cheapest."
  },
  {
    "q": "A request spends 40% of its time in the model and 60% elsewhere. What is the maximum end-to-end speed-up from optimising only the model?",
    "options": ["About 1.67x", "2.5x", "40x", "Unlimited"],
    "answer": 0,
    "why": "By Amdahl's law the ceiling is 1 / (1 - 0.4) = 1 / 0.6, about 1.67."
  },
  {
    "q": "Roughly how much memory do the weights of a 30-billion-parameter model need at 8-bit?",
    "options": ["About 7.5 GB", "About 15 GB", "About 30 GB", "About 60 GB"],
    "answer": 2,
    "why": "30e9 parameters x 1 byte = 30e9 bytes, about 30 GB, before quantisation scales and runtime overhead."
  },
  {
    "q": "Which test best catches a quantisation regression that average accuracy might hide?",
    "options": [
      "Measuring tokens per second",
      "Measuring agreement with the full-precision model on a fixed evaluation set and inspecting changed answers",
      "Checking that the file size shrank",
      "Running the model once on a sample prompt"
    ],
    "answer": 1,
    "why": "Some answers can improve while others regress; agreement and inspection of flips expose changes that cancel out in an average."
  },
  {
    "q": "In dynamic batching with a maximum batch size of 32 and a maximum wait of 10 ms, when does a batch run?",
    "options": [
      "Only when 32 requests are waiting",
      "Every 10 ms regardless of the queue",
      "When 32 requests are waiting or the oldest has waited 10 ms, whichever comes first",
      "When the GPU is idle for 10 ms"
    ],
    "answer": 2,
    "why": "The size limit maximises efficiency under load; the wait limit bounds latency when traffic is light."
  },
  {
    "q": "Why can batching reduce latency, not just raise throughput, near capacity?",
    "options": [
      "Larger batches make each input compute faster",
      "Without batching, queues grow as load approaches capacity; batching adds capacity and keeps queues short",
      "Batching skips the pre-processing step",
      "Batching removes network round trips"
    ],
    "answer": 1,
    "why": "Near saturation, queueing delay dominates latency, and the extra capacity from batching removes it."
  },
  {
    "q": "What does this cache key function get wrong?\n\n`key = hashlib.sha256(user_message.encode()).hexdigest()`",
    "options": [
      "SHA-256 is too slow for caching",
      "It ignores the model, system prompt version, parameters and tenant, so different requests can share an answer",
      "Hex digests cannot be used as dictionary keys",
      "It should use MD5 instead"
    ],
    "answer": 1,
    "why": "Everything that affects the answer must be in the key; otherwise stale or other users' answers are served."
  },
  {
    "q": "Why is semantic caching risky for questions such as \"how do I cancel my plan\" versus \"how do I not cancel my plan\"?",
    "options": [
      "Embeddings cannot represent questions",
      "The two can be highly similar in embedding space while needing opposite answers",
      "Semantic caches only work in English",
      "The cache key is too long"
    ],
    "answer": 1,
    "why": "Similarity is not equivalence; negations, numbers and names often score high while changing the meaning."
  },
  {
    "q": "Why is LLM decoding at batch size 1 typically memory-bandwidth bound?",
    "options": [
      "Each step reads all the weights but performs little arithmetic per byte read",
      "The tokeniser uses most of the memory",
      "The GPU's compute units are disabled during decoding",
      "Decoding requires writing the weights back to memory"
    ],
    "answer": 0,
    "why": "Arithmetic intensity is about one operation per byte, far below the ridge point of modern accelerators."
  },
  {
    "q": "A service's p50 latency is 200 ms and its p99 is 2.5 s. What does that tell you?",
    "options": [
      "The service is uniformly slow",
      "Typical requests are fast, but 1 in 100 takes at least 2.5 s, pointing to a slow path such as cache misses or retries",
      "The mean latency must be 2.5 s",
      "The p95 must also be 2.5 s"
    ],
    "answer": 1,
    "why": "A large gap between median and p99 indicates a tail caused by a subset of requests taking a slower path."
  },
  {
    "q": "An LLM call has TTFT 300 ms, TPOT 20 ms and 201 output tokens. About how long does the whole response take?",
    "options": ["0.3 s", "4.0 s", "4.3 s", "6.3 s"],
    "answer": 2,
    "why": "300 + 200 x 20 = 4,300 ms."
  },
  {
    "q": "A self-hosted GPU costs C per hour and processes T tokens per second when busy. What happens to the cost per token if average utilisation falls from 80% to 20%?",
    "options": ["It stays the same", "It doubles", "It quadruples", "It halves"],
    "answer": 2,
    "why": "Cost per token is inversely proportional to utilisation, and 0.8 / 0.2 = 4."
  },
  {
    "q": "A service receives 30 requests per second, each taking 4 seconds. Each replica runs 20 concurrent requests. At 100% utilisation, how many replicas are needed?",
    "options": ["2", "6", "8", "120"],
    "answer": 1,
    "why": "Little's law gives 30 x 4 = 120 requests in flight; 120 / 20 = 6 replicas, before adding headroom."
  },
  {
    "q": "Which load-balancing strategy best handles LLM requests whose generation lengths vary widely?",
    "options": ["Round robin", "Random", "Least outstanding requests", "Hashing on the client IP address"],
    "answer": 2,
    "why": "It routes on work in progress, so replicas stuck on long generations receive less new work."
  },
  {
    "q": "In OpenTelemetry, what lets spans from an API service and a separate retrieval service appear in the same trace?",
    "options": [
      "A shared log file",
      "Context propagation of the trace ID in request headers such as `traceparent`",
      "Matching timestamps",
      "Running both services in one container"
    ],
    "answer": 1,
    "why": "The trace context travels with the request, so each service's spans attach to the same trace."
  },
  {
    "q": "In this code, which errors will be retried?\n\n`if status in (429, 500, 502, 503, 504): retry_with_jittered_backoff()`",
    "options": [
      "Only authentication failures",
      "Rate-limit and server-side errors, which are often transient",
      "Every error, including 400 Bad Request",
      "None, because jitter disables retries"
    ],
    "answer": 1,
    "why": "429 and 5xx errors are commonly transient; client errors such as 400 or 401 would fail the same way again."
  }
]
```
