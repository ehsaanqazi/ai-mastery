This quiz covers the whole Advanced NLP module: sequence-to-sequence modelling and decoding, attention and cross-attention, the encoder-decoder and Transformer architectures, BERT, GPT and T5, and the summarisation, translation and question-answering tasks with their metrics. Aim for at least 13 out of 16 before moving on to Generative AI Fundamentals.

```quiz
[
  {
    "q": "A seq2seq model factorises P(y | x) as a product over target positions. What does each factor condition on?",
    "options": [
      "Only the source sequence",
      "Only the previous target token",
      "The source sequence and all previous target tokens",
      "The full target sequence, including future tokens"
    ],
    "answer": 2,
    "why": "Each step predicts the next token given the source and the target prefix, by the chain rule of probability."
  },
  {
    "q": "Greedy decoding returns a sequence with probability 0.18 while beam search returns one with probability 0.36 from the same model. What explains this?",
    "options": [
      "Beam search kept a locally weaker first token alive, and its continuation had a higher total probability",
      "Beam search uses a different model",
      "Greedy decoding has a bug",
      "Beam search always produces longer sequences"
    ],
    "answer": 0,
    "why": "Greedy commits to the best token at each step; beam search compares whole partial sequences."
  },
  {
    "q": "What is exposure bias?",
    "options": [
      "Bias in the training data against certain groups",
      "The tendency of attention to focus on the first token",
      "A model favouring short outputs in beam search",
      "The mismatch between training on reference prefixes (teacher forcing) and generating from the model's own predictions"
    ],
    "answer": 3,
    "why": "The decoder never sees its own mistakes during teacher-forced training, so errors can compound at inference."
  },
  {
    "q": "In an alignment matrix from an English-to-French model, the row for `noir` puts most weight on the column for `black`, which comes before `cat` in the source. What does this show?",
    "options": [
      "The model is broken, since alignments should be diagonal",
      "Attention has learned the reordering between the two languages",
      "The source was padded incorrectly",
      "The decoder is attending to future target tokens"
    ],
    "answer": 1,
    "why": "Off-diagonal weight reflects word-order differences, such as adjective placement in French."
  },
  {
    "q": "In scaled dot-product attention, why are scores divided by the square root of d_k?",
    "options": [
      "To keep scores from growing with dimension and saturating the softmax",
      "To make the weights sum to 1",
      "To reduce the number of parameters",
      "To implement the causal mask"
    ],
    "answer": 0,
    "why": "Large dot products make the softmax nearly one-hot, with tiny gradients. Scaling keeps them in a reasonable range."
  },
  {
    "q": "During generation with an encoder-decoder model, which statement is true?",
    "options": [
      "The encoder and decoder both run once per output token",
      "The decoder runs once and the encoder runs per token",
      "The encoder runs once; the decoder runs once per output token, reusing the encoder memory",
      "Neither runs more than once"
    ],
    "answer": 2,
    "why": "The source is fixed, so its memory is computed once and attended to at every decoding step."
  },
  {
    "q": "What does this NumPy expression build? `np.tril(np.ones((T, T), dtype=bool))`",
    "options": [
      "A padding mask",
      "An identity matrix",
      "A matrix of attention weights",
      "A causal mask: position i may attend to positions 0 to i"
    ],
    "answer": 3,
    "why": "The lower triangle (including the diagonal) is True, so each query sees itself and earlier positions only."
  },
  {
    "q": "In an original-style Transformer layer with d_ff = 4 times d_model, roughly what share of an encoder layer's parameters are in the feed-forward network?",
    "options": [
      "About two thirds",
      "About one tenth",
      "About half",
      "Almost none"
    ],
    "answer": 0,
    "why": "The FFN has about 8 d^2 parameters and self-attention about 4 d^2, so the FFN holds about two thirds."
  },
  {
    "q": "Which change did most Transformers after the original adopt for more stable training of deep stacks?",
    "options": [
      "Removing residual connections",
      "Applying layer normalisation before each sub-layer (pre-LN)",
      "Replacing attention with recurrence",
      "Using batch normalisation"
    ],
    "answer": 1,
    "why": "Pre-LN, x + Sublayer(LayerNorm(x)), keeps a clean residual path and trains more stably than post-LN."
  },
  {
    "q": "Why did BERT replace only 80% of selected tokens with `[MASK]`, leaving 10% random and 10% unchanged?",
    "options": [
      "To save computation",
      "Because `[MASK]` is not in the vocabulary",
      "Because `[MASK]` never appears in fine-tuning, so the model must build good representations of ordinary tokens too",
      "To implement next sentence prediction"
    ],
    "answer": 2,
    "why": "If only `[MASK]` positions were ever tested, representations of normal tokens would matter less, hurting transfer."
  },
  {
    "q": "You fine-tune BERT for sentiment classification. Where does the classification head usually attach?",
    "options": [
      "The final vector of the `[CLS]` token",
      "Every token's vector, one label each",
      "The `[SEP]` token's embedding",
      "The attention weights of the last layer"
    ],
    "answer": 0,
    "why": "Sequence-level tasks use the `[CLS]` vector; token-level tasks such as NER use every token's vector."
  },
  {
    "q": "What is the training objective of GPT-style models?",
    "options": [
      "Masked language modelling",
      "Span corruption",
      "Next sentence prediction",
      "Causal language modelling: predict each token from all previous tokens"
    ],
    "answer": 3,
    "why": "Decoder-only models are trained to predict the next token, with a causal mask so all positions train in parallel."
  },
  {
    "q": "In T5's span corruption, the input is `Thank you <extra_id_0> me to your party <extra_id_1> week .`. What is the target?",
    "options": [
      "`Thank you for inviting me to your party last week .`",
      "`<extra_id_0> for inviting <extra_id_1> last <extra_id_2>`",
      "`for inviting last`",
      "`<extra_id_0> <extra_id_1>`"
    ],
    "answer": 1,
    "why": "The target lists each sentinel followed by the tokens it replaced, then a final sentinel."
  },
  {
    "q": "A summariser's output reverses the key fact of the reference but reuses almost all its words. What happens to ROUGE?",
    "options": [
      "ROUGE drops to zero",
      "ROUGE flags it as a contradiction",
      "ROUGE stays high, because it measures word overlap rather than correctness",
      "ROUGE cannot be computed"
    ],
    "answer": 2,
    "why": "One changed word barely affects n-gram overlap, which is why faithfulness needs separate checks."
  },
  {
    "q": "A correct paraphrase translation shares no 4-gram with the single reference. What is its BLEU score (without smoothing)?",
    "options": [
      "100, because it is correct",
      "About 50",
      "Equal to its unigram precision",
      "0, because the geometric mean includes a zero 4-gram precision"
    ],
    "answer": 3,
    "why": "BLEU multiplies precisions through a geometric mean, so any zero precision makes the score zero."
  },
  {
    "q": "In a retriever-reader QA system, which statement is correct?",
    "options": [
      "The reader can only answer from passages the retriever returns, so retrieval recall limits accuracy",
      "The reader searches the whole collection itself",
      "The retriever generates the final answer",
      "Retrieval is only needed for extractive readers"
    ],
    "answer": 0,
    "why": "If the right passage is not retrieved, no reader can recover it. Replacing the reader with a grounded LLM gives RAG."
  }
]
```
