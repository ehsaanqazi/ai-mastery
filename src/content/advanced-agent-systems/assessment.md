This quiz covers the whole Advanced Agent Systems module: single- and multi-agent design, orchestration, tool selection, state, workflows, evaluation, error handling, guardrails and human approval. Aim for at least 13 out of 16 before moving on to Backend Development.

```quiz
[
  {
    "q": "A task has five tightly dependent sequential steps and fits easily in one context window. Which design is usually best?",
    "options": [
      "A supervisor with five worker agents",
      "One agent with good tools, or a fixed workflow",
      "A debate between two agents",
      "A hierarchical team of agents"
    ],
    "answer": 1,
    "why": "Splitting tightly coupled steps across agents adds lossy handoffs and extra calls without any parallelism benefit."
  },
  {
    "q": "In context engineering, which practice best stops tool output from crowding out the task?",
    "options": [
      "Raising the model's temperature",
      "Returning only the fields the model needs from each tool and eliding old results",
      "Putting all reference documents in the system prompt",
      "Removing the system prompt"
    ],
    "answer": 1,
    "why": "Trimming at the source and eliding stale results keep the history term of the context budget under control."
  },
  {
    "q": "Compared with one agent doing the same research sequentially, a supervisor with parallel workers usually...",
    "options": [
      "uses fewer tokens and takes longer",
      "uses more tokens in total but finishes sooner, with smaller per-call contexts",
      "uses the same tokens and the same time",
      "guarantees higher accuracy"
    ],
    "answer": 1,
    "why": "Coordination costs tokens (briefs and summaries); parallelism and context isolation are what it buys."
  },
  {
    "q": "What is the most effective cheap fix for details being lost when one agent hands off to another?",
    "options": [
      "A longer free-text summary",
      "A handoff schema with required fields plus a reference to the full transcript",
      "Letting the agents chat until they agree",
      "Using the same system prompt for both agents"
    ],
    "answer": 1,
    "why": "A schema forces the needed fields to be carried, and the transcript reference lets the receiver look up anything omitted."
  },
  {
    "q": "Which decision should be made by code rather than a model?",
    "options": [
      "Interpreting what a customer's free-text email asks for",
      "Choosing the wording of a reply",
      "Whether a refund amount exceeds the approval threshold",
      "Summarising a long document"
    ],
    "answer": 2,
    "why": "A numeric threshold is a rule that can be written down; code enforces it exactly and auditably."
  },
  {
    "q": "What does this code print?\n\n```python\nimport asyncio\nasync def slow():\n    await asyncio.sleep(1)\n    return 'done'\nasync def main():\n    r = await asyncio.gather(asyncio.wait_for(slow(), 0.1), return_exceptions=True)\n    print(type(r[0]).__name__)\nasyncio.run(main())\n```",
    "options": ["`str`", "`TimeoutError`", "`CancelledError`", "Nothing; it raises"],
    "answer": 1,
    "why": "`wait_for` times out and raises `TimeoutError`; `return_exceptions=True` puts the exception object in the results list."
  },
  {
    "q": "Why should one `Budget` object be shared between a supervisor and its workers?",
    "options": [
      "Budgets cannot be copied in Python",
      "So the whole request cannot exceed one limit, however many agents spend from it",
      "To make workers run in parallel",
      "To reduce tool-call errors"
    ],
    "answer": 1,
    "why": "Separate budgets add up; a shared one enforces a single ceiling for the request."
  },
  {
    "q": "A platform has 300 tools. Which approach scales best?",
    "options": [
      "Load all 300 definitions into every call",
      "Retrieve a short list of relevant tools per request, keep a few always-on fallbacks, and monitor recall at k",
      "Pick 10 tools at random for each request",
      "Let the model guess tool names without definitions"
    ],
    "answer": 1,
    "why": "Tool retrieval keeps context small and choices clear; recall at k shows whether the right tool reaches the model."
  },
  {
    "q": "An executor takes `user_id` from the model's tool arguments to decide whose data to return. What is the flaw?",
    "options": [
      "It is slower than reading from a cache",
      "The model, or injected text, can supply another user's ID; identity must come from the authenticated session",
      "Tool arguments cannot contain strings",
      "Nothing, as long as the system prompt forbids it"
    ],
    "answer": 1,
    "why": "Authorisation must rely on facts the model cannot control."
  },
  {
    "q": "A run crashes after calling the payment API but before saving its checkpoint. On resume, what prevents a second payment?",
    "options": [
      "A larger step limit",
      "Logging the error",
      "An idempotency key derived from the run ID and step name",
      "Using a newer model"
    ],
    "answer": 2,
    "why": "Checkpoints cannot close the gap between acting and saving; idempotency makes the repeated call harmless."
  },
  {
    "q": "Agent v2 cuts average steps by 20% and passes every case that checks only the final message. A new case checks the refunds ledger and fails. What does this show?",
    "options": [
      "The new case is wrong because v2 is cheaper",
      "Outcome checks on the environment catch failures that message checks miss",
      "Step count is the best quality metric",
      "Evaluation should only use easy cases"
    ],
    "answer": 1,
    "why": "Checking what the agent did, not just what it said, exposes unsafe or false claims of success."
  },
  {
    "q": "A case succeeds with probability 0.8 per run. What is its pass^3?",
    "options": ["0.992", "0.8", "0.512", "0.24"],
    "answer": 2,
    "why": "pass^k is the probability that all k attempts succeed: 0.8 cubed is 0.512. (0.992 is pass@3.)"
  },
  {
    "q": "Which error should be retried with exponential backoff?",
    "options": [
      "A 503 Service Unavailable from a search API",
      "A permission-denied error",
      "\"Unknown tool\" from the executor",
      "A schema validation failure on the model's arguments"
    ],
    "answer": 0,
    "why": "Transient infrastructure errors may succeed later; the others need a correction or a different path, not a retry."
  },
  {
    "q": "An agent calls `search(q=\"refund policy\")` with identical arguments again and again. Which mechanism detects this specifically?",
    "options": [
      "A loop detector counting repeated call signatures in a recent window",
      "Schema validation",
      "PII redaction",
      "An allowlist of hosts"
    ],
    "answer": 0,
    "why": "Loop detection compares recent tool calls; step limits and budgets remain the final backstop."
  },
  {
    "q": "An agent reads a web page containing hidden instructions to email out customer data. Which defence holds even if the text is not detected as an attack?",
    "options": [
      "A regular expression for \"ignore previous instructions\"",
      "A sentence in the system prompt asking the model to ignore embedded instructions",
      "Marking the context as tainted and requiring human approval for high-risk tools such as `send_email`",
      "Increasing the context window"
    ],
    "answer": 2,
    "why": "Restricting privileges after untrusted input does not depend on recognising the attack."
  },
  {
    "q": "Which is a sound rule for human approval workflows?",
    "options": [
      "Require approval for every action, including reads, to be safe",
      "Auto-approve requests that are not answered within an hour",
      "Execute exactly the approved (or approver-edited) action, check the approver's authority, and expire stale requests",
      "Let the model decide whether its action needs approval"
    ],
    "answer": 2,
    "why": "Approvals must be enforced by code, bound to a specific action, authorised and time-limited; everything else causes fatigue or gaps."
  }
]
```
