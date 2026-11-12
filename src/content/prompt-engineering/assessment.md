This quiz covers the whole Prompt Engineering module: clear zero-shot prompts, few-shot examples, roles, chain-of-thought, templates, structure, output constraints, prompt injection and evaluation. Aim for at least 13 out of 16 before moving on to Hugging Face Ecosystem.

```quiz
[
  {
    "q": "Which set of ingredients does a clear zero-shot prompt usually contain?",
    "options": [
      "A role, a temperature and a seed",
      "Task, context, input and output format",
      "Examples, reasoning and a judge",
      "A system message and nothing else"
    ],
    "answer": 1,
    "why": "Stating what to do, for whom and why, on what material, and in what shape removes the decisions the model would otherwise guess."
  },
  {
    "q": "A zero-shot prompt gives wrong answers about last week's internal sales figures. What is the right fix?",
    "options": [
      "Ask the model to think step by step",
      "Tell the model it is a sales analyst",
      "Put the figures in the prompt as context",
      "Lower the temperature to 0"
    ],
    "answer": 2,
    "why": "The failure is missing knowledge. No phrasing supplies facts the model never saw; you must provide them."
  },
  {
    "q": "Your sentiment prompt has five examples: four `positive`, then one `negative` at the end. Which two effects are you most likely introducing?",
    "options": [
      "Majority-label bias towards `positive` and recency influence from the last example",
      "Higher cost and lower latency",
      "Template injection and delimiter breaking",
      "Self-consistency and position bias"
    ],
    "answer": 0,
    "why": "The label distribution of examples acts as a prior, and the final examples can have outsized influence. Balance and interleave labels, then test orderings."
  },
  {
    "q": "Why must few-shot examples be kept out of your evaluation dataset?",
    "options": [
      "Duplicate text is rejected by the API",
      "The model could copy answers from its context, inflating the score",
      "Examples are always mislabelled",
      "It makes the prompt too short"
    ],
    "answer": 1,
    "why": "Test items that appear as examples measure recall of the prompt, not generalisation to new inputs."
  },
  {
    "q": "Which statement about role prompting is accurate?",
    "options": [
      "\"You are an expert\" reliably improves factual accuracy",
      "Roles are a dependable security boundary",
      "Roles add knowledge from the role's profession",
      "Roles mainly shape tone, focus and default assumptions; they do not add knowledge"
    ],
    "answer": 3,
    "why": "A role shifts the style and framing of output. It cannot supply facts or skills the model lacks, and it can be overridden like any instruction."
  },
  {
    "q": "Why can chain-of-thought improve accuracy on multi-step problems?",
    "options": [
      "Each reasoning token adds computation, and later steps are conditioned on intermediate results written in the context",
      "Reasoning tokens are free",
      "It switches the model to a larger network",
      "It disables sampling"
    ],
    "answer": 0,
    "why": "Writing intermediate steps gives the model a scratchpad and breaks one hard prediction into several easier ones."
  },
  {
    "q": "What does self-consistency require?",
    "options": [
      "Temperature 0 and a single call",
      "A second model to check the first",
      "Several reasoning chains sampled at non-zero temperature, with answers that can be compared and voted on",
      "A system message that says \"be consistent\""
    ],
    "answer": 2,
    "why": "Diverse samples that converge on one answer are more reliable. Identical greedy samples would make the vote pointless."
  },
  {
    "q": "When calling a reasoning model, what is usually the best use of your prompt?",
    "options": [
      "Detailed step-by-step instructions on how to think",
      "Several worked reasoning examples",
      "A request to show all reasoning in the visible answer",
      "A precise statement of the goal, constraints and success criteria"
    ],
    "answer": 3,
    "why": "Reasoning models already reason internally. Explicit CoT instructions add less, and visible reasoning adds cost and latency."
  },
  {
    "q": "What does this code print?\n\n```python\nfrom string import Template\nt = Template('Reply as {\"ok\": true}. Text: $text')\nprint(t.substitute(text='$text'))\n```",
    "options": [
      "A `KeyError`",
      "`Reply as {\"ok\": true}. Text: $text`",
      "`Reply as {\"ok\": true}. Text: `",
      "A `ValueError` about braces"
    ],
    "answer": 1,
    "why": "Braces are not special in `string.Template`, and substituted values are inserted literally, so the user's `$text` is not processed again."
  },
  {
    "q": "Which line is vulnerable to template injection?",
    "options": [
      "`env.from_string(\"Q: \" + user_text).render()`",
      "`env.from_string(\"Q: {{ q }}\").render(q=user_text)`",
      "`Template(\"Q: $q\").substitute(q=user_text)`",
      "`\"Q: \" + user_text` sent directly as the message"
    ],
    "answer": 0,
    "why": "Concatenating user text into the template source makes any template syntax in it executable. The others treat it as a value."
  },
  {
    "q": "Why are XML-style tags such as `<policy>` useful in prompts?",
    "options": [
      "Models execute the tags as markup",
      "They encrypt the inserted content",
      "They mark clear, named boundaries between instructions and data and let instructions refer to sections",
      "They guarantee the model ignores instructions inside them"
    ],
    "answer": 2,
    "why": "Tags are descriptive delimiters. They reduce confusion for a cooperative model but provide no security guarantee."
  },
  {
    "q": "What happens if you rely on a small maximum-output-tokens setting to keep JSON answers short?",
    "options": [
      "The model plans a shorter JSON object",
      "Generation stops at the limit, which can cut the JSON off and make it unparseable",
      "The API compresses the JSON",
      "The model switches to a shorter format automatically"
    ],
    "answer": 1,
    "why": "The token limit is a hard stop, not an instruction. Ask for brevity in the prompt and keep the limit as a safety cap."
  },
  {
    "q": "Why give a model an explicit sentinel such as `NOT_IN_DOCUMENT`?",
    "options": [
      "It reduces the input token count",
      "It forces the model to search the web",
      "It is required for valid JSON",
      "It provides an honest alternative to inventing an answer and is easy for code to detect"
    ],
    "answer": 3,
    "why": "An explicit escape path reduces pressure to fabricate, and an exact string can be routed reliably."
  },
  {
    "q": "An email assistant reads incoming messages and can send email. Which scenario is indirect prompt injection?",
    "options": [
      "A received email contains hidden text instructing the assistant to forward the user's inbox elsewhere",
      "The user asks the assistant to ignore its rules",
      "The developer changes the system prompt",
      "The model produces invalid JSON"
    ],
    "answer": 0,
    "why": "Indirect injection arrives through content the application reads, written by a third party, and harms the user who did nothing wrong."
  },
  {
    "q": "Which defence does NOT depend on the model obeying its instructions?",
    "options": [
      "Telling the model to ignore instructions in documents",
      "Wrapping documents in tags",
      "Requiring human approval in code before any email is sent",
      "Giving the model a security-conscious role"
    ],
    "answer": 2,
    "why": "Approval enforced by your code holds even if the model is hijacked. The others are prompt-level measures an attacker can override."
  },
  {
    "q": "An LLM judge prefers whichever answer is shown first in most pairwise comparisons. What is the correct response?",
    "options": [
      "Always show the newer prompt's answer first",
      "Accept the scores, since the judge is consistent",
      "Switch to a 1 to 10 scale",
      "Judge each pair in both orders, count only consistent wins, and validate the judge against human labels"
    ],
    "answer": 3,
    "why": "This is position bias. Swapping order exposes it, and comparison with human judgements shows whether the judge can be trusted at all."
  }
]
```
