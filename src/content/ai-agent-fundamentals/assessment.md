This quiz covers the whole AI Agent Fundamentals module: what makes a system an agent, the agent loop and its architecture, tools and function calling, short-term and long-term memory, planning, reflection, workflow patterns and human oversight. Aim for at least 14 out of 18 before moving on to Agent Frameworks.

```quiz
[
  {
    "q": "Which description best distinguishes an agent from a workflow?",
    "options": [
      "An agent uses tools; a workflow does not",
      "In an agent the model chooses the next step; in a workflow your code fixes the sequence",
      "An agent always runs for longer",
      "A workflow cannot use more than one model call"
    ],
    "answer": 1,
    "why": "Both can call models and tools. The difference is who controls the sequence of steps."
  },
  {
    "q": "In the perceive-reason-act loop, what does the 'perceive' step correspond to in code?",
    "options": [
      "Sending the system prompt",
      "Appending the tool result to the conversation so the model sees it next time",
      "Choosing which model to call",
      "Validating the user's identity"
    ],
    "answer": 1,
    "why": "The agent perceives the world through tool results added to its context."
  },
  {
    "q": "Each step of a 12-step task succeeds independently with probability 0.95. Approximately how likely is the whole task to succeed?",
    "options": ["95%", "About 85%", "About 54%", "About 12%"],
    "answer": 2,
    "why": "0.95 to the power 12 is about 0.54. Errors compound over long chains of decisions."
  },
  {
    "q": "A tool raises an exception mid-run. What should a well-designed executor do?",
    "options": [
      "Terminate the whole agent run",
      "Return a specific error message to the model as the tool result",
      "Retry the identical call until it succeeds",
      "Hide the failure and return an empty result"
    ],
    "answer": 1,
    "why": "An actionable error message lets the model correct its arguments or choose another approach."
  },
  {
    "q": "Which rule belongs in code rather than only in the system prompt?",
    "options": [
      "\"Answer in a friendly tone\"",
      "\"Never issue refunds above 500\"",
      "\"Prefer short answers\"",
      "\"Use British spelling\""
    ],
    "answer": 1,
    "why": "Rules that must always hold need enforcement in code. Prompts make behaviour likely, not guaranteed."
  },
  {
    "q": "What does this loop-detection check do?\n\n`if state.tool_history.count(call) >= 2: stop()`",
    "options": [
      "Stops the agent on the first tool call",
      "Stops the agent when an identical tool call is about to run for the third time",
      "Stops the agent when any two different tools are used",
      "Counts tokens"
    ],
    "answer": 1,
    "why": "The call has already appeared twice in the history, so a third identical request triggers the stop."
  },
  {
    "q": "Which tool description is most useful to a model?",
    "options": [
      "`search`: \"Searches.\"",
      "`search_orders`: \"Find a customer's orders by email. Returns up to 10 orders, newest first. Use before refund_order.\"",
      "`tool_7`: \"Order tool.\"",
      "`orders`: \"Does order things.\""
    ],
    "answer": 1,
    "why": "Good descriptions state what the tool does, its inputs and outputs, and when to use it relative to other tools."
  },
  {
    "q": "Why give a `create_ticket` tool an idempotency key?",
    "options": [
      "To make it run faster",
      "So a retried or repeated call does not create duplicate tickets",
      "Because JSON Schema requires a key",
      "To encrypt the ticket"
    ],
    "answer": 1,
    "why": "Agents retry and resume. The key lets the tool recognise an action it already performed."
  },
  {
    "q": "In the Chat Completions format, the model returns two tool calls in one assistant message. What must the next request include?",
    "options": [
      "Only the first tool's result",
      "The assistant message plus one `tool` message per call, each with the matching `tool_call_id`",
      "A new system prompt",
      "Nothing extra; the API remembers the calls"
    ],
    "answer": 1,
    "why": "Every tool call must be answered by a result referencing its id, or the request is rejected."
  },
  {
    "q": "When trimming an agent's history to fit a token budget, which rule is essential?",
    "options": [
      "Always drop the system prompt first",
      "Keep an assistant tool-call message and its tool results together",
      "Drop the most recent messages first",
      "Remove every tool result"
    ],
    "answer": 1,
    "why": "Splitting a call from its result breaks the tool-calling protocol."
  },
  {
    "q": "A user's preference \"prefers Python examples\" is stored for future sessions. What kind of memory is it?",
    "options": ["Working memory", "Episodic memory", "Semantic memory", "Procedural memory"],
    "answer": 2,
    "why": "It is a durable fact about the user, which is semantic memory."
  },
  {
    "q": "A memory score is `w_s * sim + w_r * recency + w_i * importance`. What happens if `w_r` is set far too high?",
    "options": [
      "Old but relevant memories dominate",
      "Recent but irrelevant memories can outrank relevant older ones",
      "Retrieval stops returning anything",
      "User isolation is weakened"
    ],
    "answer": 1,
    "why": "Recency should break ties among relevant memories, not override relevance."
  },
  {
    "q": "What is memory poisoning?",
    "options": [
      "Embeddings degrading over time",
      "Malicious instructions from untrusted content being stored and later retrieved as trusted context",
      "Running out of memory on the server",
      "Users deleting their own memories"
    ],
    "answer": 1,
    "why": "It is a delayed prompt injection: planted text is saved and replayed in later sessions."
  },
  {
    "q": "In plan-and-execute, what should the replanner receive after a step fails?",
    "options": [
      "Only the original goal",
      "The goal, the results completed so far and the failure details",
      "Only the list of tools",
      "The entire previous transcript, but not the error"
    ],
    "answer": 1,
    "why": "With completed results and the error, it can plan only the remaining work and avoid repeating the mistake."
  },
  {
    "q": "Which statement about reflection loops is most accurate?",
    "options": [
      "Self-critique by the same model reliably catches factual errors",
      "External checks such as tests or schema validation give the most reliable feedback",
      "More rounds always improve the output",
      "Reflection reduces cost"
    ],
    "answer": 1,
    "why": "Objective evaluators cannot be persuaded by plausible output. Self-critique misses what the model does not know, and rounds add cost."
  },
  {
    "q": "Inputs fall into a few distinct categories that need different prompts and tools. Which workflow pattern fits best?",
    "options": ["Prompt chaining", "Routing", "Evaluator-optimiser", "Voting"],
    "answer": 1,
    "why": "A router classifies each input and sends it to a specialised handler."
  },
  {
    "q": "Why should an approval gate for a consequential action save state and return rather than block?",
    "options": [
      "Blocking would use too many tokens",
      "Human review can take minutes or days, and the run must survive restarts and resume later",
      "LLM APIs require it",
      "It prevents the model from reasoning"
    ],
    "answer": 1,
    "why": "Interrupts with serialised state let the process exit and resume whenever the decision arrives."
  },
  {
    "q": "An agent summarises incoming emails and can also send emails and access private files. What is the main risk of this combination?",
    "options": [
      "High token usage",
      "Instructions hidden in an incoming email could cause the agent to send private data outward",
      "Emails are too long to summarise",
      "The agent will route messages incorrectly"
    ],
    "answer": 1,
    "why": "Untrusted input, private data and outward actions together are the classic setup for a damaging prompt injection; gate consequential actions."
  }
]
```
