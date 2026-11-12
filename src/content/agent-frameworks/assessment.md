This quiz covers the whole Agent Frameworks module: LangChain's runnables and tools, LangGraph's state graphs, persistence and interrupts, LlamaIndex's data framework for RAG, AutoGen's conversation model and code-execution risks, CrewAI's roles and processes, and how to choose between a framework and plain code. Aim for at least 13 out of 16 before moving on to Advanced Agent Systems.

```quiz
[
  {
    "q": "Which statement best describes what all five frameworks in this module provide?",
    "options": [
      "New model architectures that outperform plain API calls",
      "Opinionated packaging of the same agent ingredients: a loop, tools, memory and state",
      "Guaranteed protection against prompt injection",
      "A replacement for model providers' APIs"
    ],
    "answer": 1,
    "why": "Each framework wraps the first principles from Module 31 with its own abstractions; none changes what the underlying model can do."
  },
  {
    "q": "What does this LCEL expression produce when invoked with a dict of variables? `chain = prompt | model | StrOutputParser()`",
    "options": [
      "A list of messages",
      "An `AIMessage` object",
      "A plain string with the model's reply text",
      "A dictionary of the original variables"
    ],
    "answer": 2,
    "why": "The prompt turns variables into messages, the model returns an `AIMessage`, and `StrOutputParser` extracts its text as a string."
  },
  {
    "q": "Which methods does every LangChain runnable provide?",
    "options": [
      "`invoke`, `batch` and `stream` (plus async versions)",
      "`fit`, `predict` and `score`",
      "`add_node`, `add_edge` and `compile`",
      "`kickoff` and `delegate`"
    ],
    "answer": 0,
    "why": "The shared Runnable interface is what makes `|` composition and uniform batching and streaming possible."
  },
  {
    "q": "After `reply = model.bind_tools([get_stock]).invoke(messages)`, how do you know whether the model wants a tool run?",
    "options": [
      "`reply.content` starts with `TOOL:`",
      "The tool runs automatically and `reply` contains its result",
      "`bind_tools` raises an exception",
      "`reply.tool_calls` is a non-empty list"
    ],
    "answer": 3,
    "why": "Tool-calling models return an `AIMessage` whose `tool_calls` lists the requested calls. Your loop or agent must execute them."
  },
  {
    "q": "In LangGraph, a node returns `{\"attempts\": 2}`. The state also has `draft` and `log` keys. What is the new state?",
    "options": [
      "`attempts` is 2 and `draft` and `log` keep their previous values",
      "Only `attempts` remains; the other keys are removed",
      "The graph raises an error because not all keys were returned",
      "`attempts` is added to its previous value"
    ],
    "answer": 0,
    "why": "Nodes return partial updates. Without a reducer, a returned key overwrites; unmentioned keys are unchanged."
  },
  {
    "q": "What does this conditional edge do? `builder.add_conditional_edges(\"review\", route, {\"retry\": \"write\", \"done\": END})`",
    "options": [
      "Runs `write` and `END` in parallel after `review`",
      "Calls `route` on the state after `review` and goes to `write` or finishes, depending on the label it returns",
      "Retries `review` until it returns `done`",
      "Removes the `review` node when `route` returns `done`"
    ],
    "answer": 1,
    "why": "The routing function returns a label and the mapping turns it into the next node, here creating a loop back to `write` or an exit."
  },
  {
    "q": "Why is `InMemorySaver` unsuitable as a production checkpointer?",
    "options": [
      "It cannot store message lists",
      "It does not support `thread_id`",
      "Saved state is lost when the process exits",
      "It only works with OpenAI models"
    ],
    "answer": 2,
    "why": "It keeps checkpoints in process memory. Production systems use a database-backed checkpointer so paused or long-running threads survive restarts."
  },
  {
    "q": "A graph paused at an `interrupt`. How do you continue it with the human's answer?",
    "options": [
      "Call `invoke` again with the original input and a new `thread_id`",
      "Edit the checkpoint file by hand",
      "Call `interrupt` again from outside the graph",
      "Call `invoke(Command(resume=answer), config)` with the same `thread_id`"
    ],
    "answer": 3,
    "why": "Resuming with `Command(resume=...)` on the same thread loads the saved state, and the interrupted node's `interrupt` call returns the answer."
  },
  {
    "q": "In LangGraph 1.x, what has replaced the prebuilt `create_react_agent` as the recommended one-line agent?",
    "options": [
      "`create_agent` from `langchain.agents`",
      "`AgentExecutor` from `langchain`",
      "`FunctionAgent` from LlamaIndex",
      "`Crew` from CrewAI"
    ],
    "answer": 0,
    "why": "`create_react_agent` is deprecated in the 1.x family in favour of `langchain.agents.create_agent`, which builds the same model-and-tools graph."
  },
  {
    "q": "Which LlamaIndex object combines a retriever with a response synthesiser?",
    "options": [
      "`SimpleDirectoryReader`",
      "A query engine",
      "`SentenceSplitter`",
      "`StorageContext`"
    ],
    "answer": 1,
    "why": "`index.as_query_engine()` retrieves the top-k nodes and synthesises an answer from them."
  },
  {
    "q": "A LlamaIndex app answers wrongly. `response.source_nodes` contains the correct passage with a high score. Where is the problem most likely?",
    "options": [
      "Document loading",
      "Chunking",
      "Generation: the prompt, response mode or model",
      "The embedding model"
    ],
    "answer": 2,
    "why": "Retrieval found the right text, so the failure happened when turning it into an answer."
  },
  {
    "q": "When is `index.as_retriever().retrieve(question)` preferable to a query engine?",
    "options": [
      "When you want retrieved chunks without an LLM call, for example to feed your own agent or to test retrieval",
      "When you want a shorter final answer",
      "When the index has not been built yet",
      "When you want the model to choose between several indexes"
    ],
    "answer": 0,
    "why": "A retriever returns scored nodes only. It is cheaper, deterministic for a fixed index and easy to evaluate."
  },
  {
    "q": "Why does every AutoGen-style multi-agent conversation need termination conditions?",
    "options": [
      "To save the transcript to disk",
      "To decide which agent speaks next",
      "To allow code execution",
      "Because agents have no built-in reason to stop and can loop indefinitely"
    ],
    "answer": 3,
    "why": "Combine a success signal such as a keyword with hard caps on turns or messages."
  },
  {
    "q": "Which is the most important safeguard for an agent that writes and executes code?",
    "options": [
      "Using a larger model",
      "Running the code in an isolated sandbox without secrets or network access, with time limits",
      "Asking the model to promise the code is safe",
      "Running code only on weekdays"
    ],
    "answer": 1,
    "why": "Model-written code is untrusted input and may be influenced by prompt injection. Isolation limits what a mistake or attack can do."
  },
  {
    "q": "In CrewAI, what is the difference between `Process.sequential` and `Process.hierarchical`?",
    "options": [
      "Sequential uses one agent, hierarchical uses several",
      "Sequential is asynchronous, hierarchical is synchronous",
      "Sequential runs tasks in the listed order; hierarchical has a manager LLM delegate tasks at run time",
      "Hierarchical runs every task twice for verification"
    ],
    "answer": 2,
    "why": "Hierarchical mode adds a manager (configured with `manager_llm` or `manager_agent`) that plans, delegates and reviews."
  },
  {
    "q": "You need an expense-approval agent that waits days for a manager's sign-off and then continues. Which choice fits best?",
    "options": [
      "LangGraph with a database-backed checkpointer and an interrupt",
      "A LlamaIndex query engine",
      "An AutoGen round-robin chat with no termination condition",
      "An LCEL chain with `batch`"
    ],
    "answer": 0,
    "why": "Durable state and pause-and-resume are LangGraph's design centre. The other options do not provide resumable long-running state."
  }
]
```
