This quiz covers the whole Application Development module: Streamlit, Gradio, React fundamentals, connecting a frontend to a FastAPI backend, and interface patterns for AI applications. Aim for at least 13 out of 16 before moving on to Docker and Containers.

```quiz
[
  {
    "q": "A Streamlit app starts with `history = []` and appends each message to it. Why does the history never contain more than one message?",
    "options": [
      "Streamlit lists are immutable",
      "Each interaction reruns the script, recreating the empty list; it should live in `st.session_state`",
      "`st.chat_input` clears all variables",
      "The list must be decorated with `st.cache_data`"
    ],
    "answer": 1,
    "why": "Ordinary variables are recreated on every rerun. Per-session values that must persist belong in `st.session_state`."
  },
  {
    "q": "What is the key difference between `st.cache_data` and `st.cache_resource`?",
    "options": [
      "`st.cache_data` returns a copy of a serialisable value; `st.cache_resource` returns the same shared object, suited to models and connections",
      "`st.cache_resource` is per user; `st.cache_data` is global",
      "`st.cache_data` only works with pandas",
      "There is no difference; one is an alias"
    ],
    "answer": 0,
    "why": "Data caching copies results so callers cannot affect each other; resource caching shares one object, avoiding repeated model loads."
  },
  {
    "q": "Which Streamlit call renders a generator of text chunks progressively and returns the full text?",
    "options": [
      "`st.chat_message`",
      "`st.markdown`",
      "`st.write_stream`",
      "`st.status`"
    ],
    "answer": 2,
    "why": "`st.write_stream` consumes the generator, displays each chunk as it arrives and returns the concatenated result."
  },
  {
    "q": "You want several Streamlit inputs to be submitted together, without a rerun on every keystroke. What do you use?",
    "options": [
      "`st.columns`",
      "`st.cache_data`",
      "`st.expander`",
      "`st.form` with `st.form_submit_button`"
    ],
    "answer": 3,
    "why": "Widgets inside a form only send their values when the form's submit button is pressed."
  },
  {
    "q": "In Gradio, when you click a button wired with `btn.click(fn, inputs, outputs)`, what runs?",
    "options": [
      "The whole Python file, top to bottom",
      "Only `fn`, with the current values of `inputs`; its return value updates `outputs`",
      "Every function registered in the Blocks",
      "Nothing until `demo.launch()` is called again"
    ],
    "answer": 1,
    "why": "Gradio is event-driven: only the function bound to the event runs, unlike Streamlit's full-script rerun."
  },
  {
    "q": "What arguments does the function passed to `gr.ChatInterface` receive?",
    "options": [
      "The new message and the conversation history, plus any additional inputs",
      "Only the full conversation as one string",
      "A request object and a response object",
      "The chatbot component itself"
    ],
    "answer": 0,
    "why": "The signature is `fn(message, history, *additional_inputs)`, with history as a list of role and content dictionaries."
  },
  {
    "q": "A Gradio app serves a model on one local GPU. What does setting a low concurrency limit achieve?",
    "options": [
      "It makes each request faster",
      "It disables the queue",
      "It limits simultaneous runs so the GPU is not overloaded, while the queue holds waiting requests",
      "It creates a public share link"
    ],
    "answer": 2,
    "why": "The queue absorbs bursts; the concurrency limit controls how many calls run at once on the constrained resource."
  },
  {
    "q": "What does this React code render when `items` is `[\"a\", \"b\"]`?\n\n`<ul>{items.map((x) => <li key={x}>{x.toUpperCase()}</li>)}</ul>`",
    "options": [
      "An empty list, because `map` returns nothing",
      "A list with one item reading `AB`",
      "A list with items `a` and `b`",
      "A list with items `A` and `B`"
    ],
    "answer": 3,
    "why": "`map` returns an array of `li` elements, one per item, each showing the upper-cased string."
  },
  {
    "q": "Why does `setMessages([...messages, newMessage])` work when `messages.push(newMessage); setMessages(messages)` does not?",
    "options": [
      "The spread creates a new array, so React sees a changed value and re-renders",
      "`push` is not allowed in TypeScript",
      "`setMessages` only accepts literal arrays",
      "The spread operator is faster"
    ],
    "answer": 0,
    "why": "React compares state by identity. Mutating and passing the same array looks like no change, so no re-render happens."
  },
  {
    "q": "An effect is declared as `useEffect(() => { loadDoc(docId); }, [docId])`. When does it run?",
    "options": [
      "Only once, on the first render",
      "After the first render and again whenever `docId` changes",
      "Before every render",
      "Only when the component is removed"
    ],
    "answer": 1,
    "why": "The dependency array lists the values the effect uses; React re-runs it when any of them change."
  },
  {
    "q": "A React app on `localhost:5173` can call FastAPI on `localhost:8000` from `curl` but not from the browser. What is the most likely cause?",
    "options": [
      "FastAPI does not accept JSON",
      "The browser blocks the response because the backend does not allow that origin via CORS",
      "Port 8000 is reserved by the browser",
      "`fetch` cannot make POST requests"
    ],
    "answer": 1,
    "why": "Different ports are different origins. Browsers enforce CORS; `curl` does not. Add the origin to `CORSMiddleware` or proxy through the dev server."
  },
  {
    "q": "Which value is safe to put in a Vite `VITE_` environment variable?",
    "options": [
      "The LLM provider's API key",
      "The database password",
      "The public base URL of your API",
      "The secret used to sign auth tokens"
    ],
    "answer": 2,
    "why": "`VITE_` variables are compiled into the public JavaScript bundle. Only non-secret configuration belongs there."
  },
  {
    "q": "When reading an SSE stream with `fetch` and a `ReadableStream`, why does the client keep a buffer of incomplete text?",
    "options": [
      "Network chunks do not align with event boundaries, so an event may be split across reads",
      "To slow the stream down for readability",
      "Because `TextDecoder` drops characters otherwise",
      "SSE requires the client to echo data back"
    ],
    "answer": 0,
    "why": "One chunk can hold part of an event or several events; only complete events, ending in a blank line, can be parsed."
  },
  {
    "q": "Which loading model best matches an AI chat request?",
    "options": [
      "A single boolean `isLoading`",
      "No loading state; streaming makes it unnecessary",
      "Only an error state",
      "Distinct states such as idle, waiting for the first token, streaming and error, with the ability to stop"
    ],
    "answer": 3,
    "why": "AI requests have a long wait before the first token, a streaming phase, mid-stream failures and user cancellation; each needs its own UI."
  },
  {
    "q": "Why is a \"Confidence: 87%\" badge computed from average token probability a poor uncertainty signal?",
    "options": [
      "Token probabilities are always 100%",
      "It is not calibrated to correctness, so users will over-trust fluent wrong answers",
      "Browsers cannot display percentages",
      "It makes responses slower"
    ],
    "answer": 1,
    "why": "Token probability reflects how predictable the wording is, not whether the answer is true. Show validated signals instead."
  },
  {
    "q": "An internal team wants a dashboard to explore evaluation results with charts and filters, built this week by Python developers. What is the most suitable choice?",
    "options": [
      "A custom React frontend and FastAPI backend",
      "A Gradio `Interface` with one text box",
      "Streamlit",
      "A WebSocket server"
    ],
    "answer": 2,
    "why": "Streamlit excels at data-heavy internal dashboards built quickly in Python; the scale and customisation limits do not matter here."
  }
]
```
