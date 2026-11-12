This quiz covers the whole AI System Design module: the design method and estimation, LLM application and RAG architecture, distributed inference, serving platforms, availability, data stores, cost and latency, reliability and observability. Many questions are short design scenarios. Aim for at least 15 out of 18 to finish the curriculum with confidence.

```quiz
[
  {
    "q": "In a design interview you are asked to build an AI meeting summariser. What should you do first?",
    "options": [
      "Draw the architecture with a vector database and an agent framework",
      "Clarify requirements: users and volume, latency (real-time or after the meeting), quality bar, privacy and data residency",
      "Pick the largest available model",
      "Estimate GPU count from the model's parameter count"
    ],
    "answer": 1,
    "why": "Requirements decide everything else; real-time versus post-meeting summarisation alone changes the whole architecture."
  },
  {
    "q": "An app expects 1.2 million requests per day spread evenly over 24 hours, with a peak factor of 2. What is peak QPS?",
    "options": ["About 14", "About 28", "About 50", "About 280"],
    "answer": 1,
    "why": "1,200,000 / 86,400 ≈ 13.9 average QPS; times 2 ≈ 28."
  },
  {
    "q": "How many bytes do 10 million 768-dimensional float32 embeddings occupy, before index overhead?",
    "options": ["About 3 GB", "About 7.7 GB", "About 31 GB", "About 77 GB"],
    "answer": 2,
    "why": "10,000,000 × 768 × 4 bytes = 30.72 × 10^9 bytes, about 31 GB."
  },
  {
    "q": "Which code best describes how GPU count should be estimated for a self-hosted model?",
    "options": [
      "`gpus = params / 1e9`",
      "`replicas = ceil(peak_output_tps * headroom / measured_tps_per_replica_at_slo); gpus = replicas * gpus_per_replica`",
      "`gpus = users / 1000`",
      "`gpus = context_length / 4096`"
    ],
    "answer": 1,
    "why": "Throughput per replica must be measured at the latency target; memory decides GPUs per replica, traffic decides replicas."
  },
  {
    "q": "A support agent must issue refunds. Which design keeps a prompt-injected model from refunding another customer's order?",
    "options": [
      "A system prompt telling the model to check ownership",
      "A tool executor that checks the order belongs to the authenticated customer and requires idempotency keys and approval above a limit",
      "Using a larger model",
      "Lower temperature"
    ],
    "answer": 1,
    "why": "Controls must live in code that holds the real identity; the model's instructions can be overridden by injected text."
  },
  {
    "q": "An enterprise RAG assistant post-filters retrieved chunks by ACL and users in small teams often get empty answers. What is the best fix?",
    "options": [
      "Remove ACL checks",
      "Use filtered (pre-filter) search in the index, over-fetch, and keep a final source permission check on the top chunks",
      "Increase the LLM's temperature",
      "Build one index per employee"
    ],
    "answer": 1,
    "why": "Filtering during the search returns k permitted results; the final check guards against stale ACL metadata."
  },
  {
    "q": "Your company switches to a new embedding model. What is the safe migration?",
    "options": [
      "Re-embed documents in place within the live index",
      "Build a new index, backfill and dual-write, evaluate, then switch an alias and keep the old index for rollback",
      "Embed only new documents with the new model",
      "Re-embed queries only"
    ],
    "answer": 1,
    "why": "Vectors from different models are not comparable, so the switch must be atomic from the reader's point of view and reversible."
  },
  {
    "q": "A model's weights need 140 GB and the KV cache for your target concurrency needs 90 GB. With 80 GB GPUs, about 90% usable, what tensor-parallel degree do you choose?",
    "options": ["2", "3", "4", "8"],
    "answer": 2,
    "why": "230 GB / 72 GB per GPU ≈ 3.2, so at least 4 GPUs; 4 is also a valid tensor-parallel degree, unlike 3."
  },
  {
    "q": "Why is tensor parallelism usually kept within a single server?",
    "options": [
      "It needs a different model format",
      "It communicates within every layer for every token, so it needs the fastest GPU interconnect",
      "Servers cannot share memory",
      "It only works with quantised models"
    ],
    "answer": 1,
    "why": "Pipeline parallelism, with far less communication, is the usual choice across servers."
  },
  {
    "q": "A RAG workload has very long prompts and short answers, and token streams stutter under load. Which architectural change targets this directly?",
    "options": [
      "Disaggregating prefill and decode, or enabling chunked prefill",
      "Adding a response cache only",
      "Switching to a larger model",
      "Turning off streaming"
    ],
    "answer": 0,
    "why": "Long prefills interfere with decode steps on shared GPUs; separating or chunking them smooths token generation."
  },
  {
    "q": "In a multi-tenant LLM gateway, the HR tenant must never send data to external providers. The self-hosted model is down. What should happen?",
    "options": [
      "Fall back to an external provider to preserve availability",
      "Routing policy excludes external backends for this tenant, so the gateway returns a clear error or a permitted degraded option",
      "Retry the self-hosted model forever",
      "Send the request without the sensitive parts"
    ],
    "answer": 1,
    "why": "Data policy is a routing constraint that fallbacks must respect."
  },
  {
    "q": "Arrivals are 40 requests per second, average latency is 5 s, and each replica serves 25 concurrent sequences at the target latency. How many replicas are needed before headroom?",
    "options": ["2", "5", "8", "200"],
    "answer": 2,
    "why": "Little's law: 40 × 5 = 200 in flight; 200 / 25 = 8 replicas."
  },
  {
    "q": "A request path depends on four components in series, each 99.9% available. What is the approximate end-to-end availability?",
    "options": ["99.9%", "99.6%", "99.0%", "96%"],
    "answer": 1,
    "why": "0.999^4 ≈ 0.996. Series dependencies multiply, so the chain is less available than any single component."
  },
  {
    "q": "A real-time moderation service fails. Which design is generally best?",
    "options": [
      "Always fail open and publish everything",
      "Always fail closed and block all posts",
      "Defined degraded modes: fallback rules, hold high-risk items, re-check items decided by fallback once the full pipeline recovers",
      "Restart the service and hope"
    ],
    "answer": 2,
    "why": "Neither extreme fits every category; explicit degradation limits both harm and disruption."
  },
  {
    "q": "Which data in an AI application can usually tolerate the loosest RPO?",
    "options": ["User feedback labels", "Conversation history", "A response cache", "Billing records"],
    "answer": 2,
    "why": "A cache can start empty after a disaster; the others are original data that cannot be regenerated."
  },
  {
    "q": "Why use a transactional outbox to sync document changes to the vector index?",
    "options": [
      "It makes similarity search faster",
      "It records the change and the sync event atomically, so a worker can apply idempotent index updates with retries and nothing is lost on a crash",
      "It removes the need for a relational database",
      "It encrypts embeddings"
    ],
    "answer": 1,
    "why": "Two separate systems cannot be updated atomically; the outbox gives reliable eventual consistency."
  },
  {
    "q": "Routing 60% of traffic to a model one-tenth the price cuts cost sharply in your cost model. What must you do before adopting it?",
    "options": [
      "Nothing; the cost model proves it is better",
      "Measure quality on the evaluation set for the routed traffic and confirm it meets the quality bar",
      "Increase the cache TTL",
      "Raise the latency SLO"
    ],
    "answer": 1,
    "why": "Cost levers that change the model carry quality risk; only evaluation shows whether the saving is worth it."
  },
  {
    "q": "Which alert design best follows SLO-based alerting practice?",
    "options": [
      "Page whenever GPU utilisation exceeds 90%",
      "Page when the error-budget burn rate is high over both a long and a short window, with a linked runbook",
      "Email every failed request to the team",
      "Alert on average latency only"
    ],
    "answer": 1,
    "why": "Multiwindow burn-rate alerts catch significant, ongoing user-facing problems and ignore brief blips and harmless internal signals."
  }
]
```
