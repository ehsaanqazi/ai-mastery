This quiz covers the whole of the Hugging Face Ecosystem module: `transformers` models, heads and generation, tokenizers, `datasets`, the Model Hub and its security, pipelines, fine-tuning with the `Trainer`, PEFT, LoRA, QLoRA and quantisation. Aim for at least 13 out of 16 before moving on to RAG Fundamentals.

```quiz
[
  {
    "q": "Which file does `AutoModelForCausalLM.from_pretrained` read to decide which architecture class to build?",
    "options": ["`tokenizer.json`", "`config.json`", "`generation_config.json`", "`README.md`"],
    "answer": 1,
    "why": "The Auto classes read `model_type` and the hyperparameters from `config.json`, build the layers, then load the weights into them."
  },
  {
    "q": "A sequence-classification model returns `logits` of shape `(4, 3)`. What does this mean?",
    "options": [
      "4 tokens, 3 layers",
      "4 input texts, 3 possible labels",
      "4 labels, 3 input texts",
      "Batch size 3, sequence length 4"
    ],
    "answer": 1,
    "why": "Sequence classification produces one row of label scores per input text."
  },
  {
    "q": "What does this code print the reply of?\n\n`ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=\"pt\")`\n`out = model.generate(ids, max_new_tokens=50)`\n`print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))`",
    "options": [
      "The whole conversation including the prompt",
      "Only the newly generated assistant tokens",
      "The system message",
      "The last 50 tokens of the prompt"
    ],
    "answer": 1,
    "why": "`generate` returns the prompt followed by new tokens; slicing from `ids.shape[1]` keeps only the generated reply."
  },
  {
    "q": "You batch prompts of different lengths into a decoder-only model for generation. Which tokenizer setting do you need?",
    "options": [
      "`padding_side = \"right\"`",
      "`padding_side = \"left\"`",
      "`truncation = False`",
      "`add_special_tokens = False`"
    ],
    "answer": 1,
    "why": "Left padding makes every prompt end at the final position, which is where generation continues from."
  },
  {
    "q": "Which capability comes only from fast (Rust-backed) tokenizers?",
    "options": [
      "Decoding ids to text",
      "Character offset mapping and `word_ids()`",
      "Adding special tokens",
      "Padding a batch"
    ],
    "answer": 1,
    "why": "Offset mappings back to the original string, and word alignment, are provided by the `tokenizers` backend."
  },
  {
    "q": "A batched `datasets.map` function receives 2 rows and returns 5. What must you also pass to `map`?",
    "options": [
      "`num_proc=5`",
      "`remove_columns` for the original columns",
      "`streaming=True`",
      "`batch_size=5`"
    ],
    "answer": 1,
    "why": "Old columns have 2 values and new columns 5, so the old ones must be removed for the table to stay rectangular."
  },
  {
    "q": "Why does a 100 GB dataset loaded with `datasets` not need 100 GB of RAM?",
    "options": [
      "It is compressed in memory",
      "The Arrow cache file is memory-mapped and pages are read on demand",
      "Only the first split is loaded",
      "Rows are fetched from the Hub on every access"
    ],
    "answer": 1,
    "why": "Memory mapping gives the process a view of the file; the OS reads only the pages that are touched."
  },
  {
    "q": "Which option makes a production service load exactly the same model files every time?",
    "options": [
      "`revision=\"main\"`",
      "Setting `HF_HOME`",
      "Pinning `revision` to a full commit hash",
      "Passing `trust_remote_code=True`"
    ],
    "answer": 2,
    "why": "A commit hash is immutable; branches and tags can move."
  },
  {
    "q": "Why are safetensors files safer than pickle-based `.bin` checkpoints?",
    "options": [
      "They are encrypted",
      "They contain only a JSON header and raw tensor bytes, so loading cannot execute code",
      "They are smaller",
      "They are signed by Hugging Face"
    ],
    "answer": 1,
    "why": "Unpickling can call arbitrary functions; safetensors is a pure data format with nothing to execute."
  },
  {
    "q": "A zero-shot classification pipeline is called with 8 candidate labels. Roughly how many forward passes does each input need?",
    "options": ["1", "2", "8", "64"],
    "answer": 2,
    "why": "It runs one NLI premise-hypothesis pass per candidate label."
  },
  {
    "q": "With `per_device_train_batch_size=8`, `gradient_accumulation_steps=4` on one GPU, and 3,200 training examples, how many optimiser steps are in one epoch?",
    "options": ["100", "400", "800", "32"],
    "answer": 0,
    "why": "Effective batch = 8 x 4 = 32 examples per update, and 3,200 / 32 = 100 steps."
  },
  {
    "q": "What does `compute_metrics` receive from the `Trainer`?",
    "options": [
      "The model and the evaluation dataset",
      "Logits and labels for the whole evaluation set",
      "One batch at a time during training",
      "Only the validation loss"
    ],
    "answer": 1,
    "why": "The Trainer collects predictions over the evaluation set and passes them, with the labels, as NumPy arrays."
  },
  {
    "q": "LoRA with rank 16 is applied to one 2048 x 2048 matrix. How many parameters are trained for it?",
    "options": ["32,768", "65,536", "4,194,304", "4,096"],
    "answer": 1,
    "why": "r (d + k) = 16 x (2048 + 2048) = 65,536, compared with 4,194,304 for the full matrix."
  },
  {
    "q": "Why does LoRA initialise B to zero?",
    "options": [
      "To save memory",
      "So that BA = 0 and the adapted model starts identical to the pretrained model",
      "So that A does not need gradients",
      "To make the rank exactly r"
    ],
    "answer": 1,
    "why": "Starting with a zero update means training begins from the pretrained behaviour rather than a random perturbation of it."
  },
  {
    "q": "Which combination describes QLoRA?",
    "options": [
      "4-bit trainable adapters on a 16-bit base",
      "A frozen 4-bit NF4 base model with 16-bit LoRA adapters, double quantisation and paged optimisers",
      "Full fine-tuning in int8",
      "Quantisation-aware training of every weight"
    ],
    "answer": 1,
    "why": "QLoRA quantises only the frozen base; adapters and computation stay in 16-bit."
  },
  {
    "q": "Symmetric int8 quantisation of a tensor whose largest absolute value is 6.35 uses which scale?",
    "options": ["0.025", "0.05", "6.35", "0.0249"],
    "answer": 1,
    "why": "Scale = max|x| / 127 = 6.35 / 127 = 0.05."
  }
]
```
