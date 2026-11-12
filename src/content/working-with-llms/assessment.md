This quiz covers the whole Working with LLMs module, from calling an API and loading open models to structured outputs, tool calling, streaming, cost and evaluation. Answer every question to see your score. Your best result is saved, and you can retry as often as you like. Aim for at least 14 out of 18 before moving on to Prompt Engineering.

```quiz
[
  {
    "q": "A chat application sends only the latest user message on each call to Chat Completions. What happens?",
    "options": [
      "The provider fills in the earlier turns from its memory",
      "The model has no knowledge of the earlier conversation",
      "The API rejects the request",
      "The model answers more accurately because the prompt is shorter"
    ],
    "answer": 1,
    "why": "The API is stateless. The only context the model sees is what you send, so the history must be resent (and trimmed) by your code."
  },
  {
    "q": "Which `finish_reason` tells you the reply was cut off by the output token limit?",
    "options": [
      "`stop`",
      "`tool_calls`",
      "`length`",
      "`content_filter`"
    ],
    "answer": 2,
    "why": "`length` means `max_tokens` was reached. `stop` is a natural end and `tool_calls` means the model wants to call a function."
  },
  {
    "q": "Which change lets code written for the `openai` SDK talk to a local Ollama or vLLM server?",
    "options": [
      "Install a different SDK",
      "Set `base_url` to the server's `/v1` address and use the server's model name",
      "Add `local=True` to every call",
      "Convert the messages to XML"
    ],
    "answer": 1,
    "why": "Both servers implement an OpenAI-compatible Chat Completions endpoint, so only the base URL and model name change."
  },
  {
    "q": "Roughly how much memory do the weights of an 8B-parameter model need in bfloat16?",
    "options": [
      "4 GB",
      "8 GB",
      "16 GB",
      "32 GB"
    ],
    "answer": 2,
    "why": "bfloat16 uses 2 bytes per parameter: 8 x 10^9 x 2 = 16 x 10^9 bytes. 32 GB would be float32 and 4 GB 4-bit."
  },
  {
    "q": "A model card says the licence is for non-commercial research only. What does this mean for a paid product?",
    "options": [
      "It can be used if you credit the authors",
      "It can be used after fine-tuning, because the result is a new model",
      "It should not be used in the product; choose a model whose licence permits commercial use",
      "Licences on the Hub are not legally binding"
    ],
    "answer": 2,
    "why": "Downloadable is not the same as licensed for commercial use, and fine-tuned derivatives usually inherit the base model's terms."
  },
  {
    "q": "Why is a `.safetensors` file safer to load than a pickled `.bin` file from an unknown author?",
    "options": [
      "It is compressed",
      "It stores only tensors and metadata, so loading it cannot execute code",
      "It is signed by Hugging Face",
      "It loads faster on CPUs"
    ],
    "answer": 1,
    "why": "Unpickling can run arbitrary code embedded in the file; safetensors cannot."
  },
  {
    "q": "What does this code print first?\n\n`out = model.generate(**inputs, max_new_tokens=50)`\n`print(tokenizer.decode(out[0], skip_special_tokens=True))`",
    "options": [
      "Only the model's reply",
      "The prompt text followed by the reply",
      "Token ids",
      "Nothing, because special tokens are skipped"
    ],
    "answer": 1,
    "why": "`generate` returns the prompt tokens plus the new ones. Slice off the first `input_ids.shape[1]` tokens to get only the reply."
  },
  {
    "q": "What is the purpose of `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`?",
    "options": [
      "To translate messages into English",
      "To format role-tagged messages with the exact special tokens the model was trained on, ending with the start of an assistant turn",
      "To count tokens for billing",
      "To compress the prompt"
    ],
    "answer": 1,
    "why": "Each model family has its own conversation format. Using the wrong one is a common cause of strange outputs."
  },
  {
    "q": "What does `'Return {\"label\": \"x\"} for: {text}'.format(text=\"hi\")` do?",
    "options": [
      "Returns the string with `hi` inserted",
      "Raises an error, because the JSON braces are read as a placeholder",
      "Returns the template unchanged",
      "Escapes the braces automatically"
    ],
    "answer": 1,
    "why": "`str.format` treats every brace pair as a field. Double the braces or use `string.Template` for prompts containing JSON."
  },
  {
    "q": "Where should per-request user data, such as a customer's order history, normally go?",
    "options": [
      "At the start of the system message",
      "In a clearly delimited block in a user message, after the stable instructions",
      "In the tool definitions",
      "In the model name"
    ],
    "answer": 1,
    "why": "Keeping the system message stable preserves prompt caching and keeps rules separate from data that might be partly user-controlled."
  },
  {
    "q": "You use `response_format={\"type\": \"json_object\"}`. What is still not guaranteed?",
    "options": [
      "That the output parses as JSON",
      "That the output has the fields, types and allowed values you need",
      "That the output is UTF-8",
      "That a response is returned"
    ],
    "answer": 1,
    "why": "JSON mode guarantees syntax only. Schema-constrained output and validation enforce the shape."
  },
  {
    "q": "Pydantic rejects a model's JSON because `rating` is 7 but the field has `le=5`. What is the most effective automatic recovery?",
    "options": [
      "Clip the value to 5 silently",
      "Resend the identical request",
      "Send the previous output and the validation error back and ask for corrected JSON, with a cap on attempts",
      "Disable the constraint"
    ],
    "answer": 2,
    "why": "Specific error feedback lets the model fix the problem; capping attempts prevents endless loops and leaves a fallback path."
  },
  {
    "q": "In the OpenAI tool-calling loop, which message must be appended before the tool results?",
    "options": [
      "A new system message",
      "The assistant message that contains the `tool_calls`",
      "A user message repeating the question",
      "Nothing; results can be sent alone"
    ],
    "answer": 1,
    "why": "Each `tool` result refers to a call by `tool_call_id`, so the assistant turn containing those calls must precede it in the history."
  },
  {
    "q": "A model proposes `refund_order(order_id=\"981\", amount=500)`. The arguments pass schema validation. What must your code still do?",
    "options": [
      "Nothing further",
      "Ask the model to confirm it is sure",
      "Check that the authenticated user owns order 981 and that the refund is allowed, possibly with human approval",
      "Convert the amount to a string"
    ],
    "answer": 2,
    "why": "Validation checks shape, not permission. Authorisation uses the real session identity, never model-produced arguments."
  },
  {
    "q": "When streaming with `stream_options={\"include_usage\": True}`, what is special about the final chunk?",
    "options": [
      "It repeats the whole answer",
      "It has `usage` set and an empty `choices` list",
      "It contains the first token",
      "It is sent before any text"
    ],
    "answer": 1,
    "why": "That is why streaming loops guard with `if chunk.choices:` before reading `choices[0].delta`."
  },
  {
    "q": "A team adds a per-request timestamp to the first line of a long, otherwise fixed system prompt. What happens to prompt caching?",
    "options": [
      "Nothing; caching ignores timestamps",
      "Cache hits disappear, because the prefix differs on every request",
      "Caching improves because the prompt is unique",
      "Only output tokens are affected"
    ],
    "answer": 1,
    "why": "Caches reuse an identical prefix. Variable content belongs at the end of the prompt."
  },
  {
    "q": "With illustrative prices of USD 2 per million input tokens and USD 8 per million output tokens, what does a call with 5,000 input and 500 output tokens cost?",
    "options": [
      "USD 0.010",
      "USD 0.014",
      "USD 0.040",
      "USD 14.00"
    ],
    "answer": 1,
    "why": "(5,000 x 2 + 500 x 8) / 1,000,000 = 14,000 / 1,000,000 = 0.014 USD."
  },
  {
    "q": "An LLM judge agrees with human graders on 85% of cases, but Cohen's kappa is low. What is the most likely explanation?",
    "options": [
      "The judge is excellent",
      "Most cases share one label, so high raw agreement is expected by chance; the judge may be passing almost everything",
      "Kappa cannot be used with two labels",
      "The human graders made errors"
    ],
    "answer": 1,
    "why": "Kappa corrects for chance agreement. A lenient judge on a mostly-pass set has high raw agreement but adds little information."
  }
]
```
