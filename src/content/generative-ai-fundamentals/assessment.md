This quiz covers the whole of Generative AI Fundamentals: generative versus discriminative modelling, the modern decoder-only Transformer, attention and positional information, tokenisation and embeddings, context windows, pretraining, fine-tuning, inference and sampling. Aim for at least 13 out of 16 before moving on to Working with LLMs.

```quiz
[
  {
    "q": "Which statement best describes what a generative model learns?",
    "options": [
      "A decision boundary between classes",
      "The distribution of the data, so it can sample new examples and score how likely an input is",
      "A lookup table of question and answer pairs",
      "Only the conditional p(y | x)"
    ],
    "answer": 1,
    "why": "Generative models learn p(x) or p(x, y). Discriminative models learn p(y | x) or a boundary."
  },
  {
    "q": "An autoregressive language model computes the probability of a sequence as:",
    "options": [
      "The product of p(x_t | all previous tokens) over every position t",
      "The sum of the probabilities of each token on its own",
      "The probability of the last token only",
      "The average of the word embeddings"
    ],
    "answer": 0,
    "why": "This is the chain rule of probability, which turns sequence modelling into repeated next-token prediction."
  },
  {
    "q": "What is the purpose of the causal mask in a decoder-only Transformer?",
    "options": [
      "To hide padding tokens from the loss",
      "To randomly drop attention weights during training",
      "To stop each position from attending to later positions, so the model cannot see the token it must predict",
      "To reduce the size of the vocabulary"
    ],
    "answer": 2,
    "why": "Masking future positions makes training on all positions at once equivalent to left-to-right prediction."
  },
  {
    "q": "In scaled dot-product attention, softmax(QK^T / sqrt(d_k)) V, why divide by sqrt(d_k)?",
    "options": [
      "To make the attention weights sum to one",
      "To keep the dot products from growing with d_k, which would push softmax into a saturated region with tiny gradients",
      "To convert the scores into probabilities",
      "To apply the causal mask"
    ],
    "answer": 1,
    "why": "Dot products of d_k-dimensional vectors have variance that grows with d_k. Scaling keeps the softmax in a well-behaved range."
  },
  {
    "q": "A model has d_model = 512 and 8 attention heads. What is the dimension of each head?",
    "options": [
      "8",
      "512",
      "4096",
      "64"
    ],
    "answer": 3,
    "why": "Heads split the model width: d_head = d_model / h = 512 / 8 = 64."
  },
  {
    "q": "What does grouped-query attention (GQA) change compared with standard multi-head attention?",
    "options": [
      "Several query heads share one set of key and value heads, shrinking the KV cache",
      "It removes the softmax",
      "Each head attends only to its own group of tokens",
      "It replaces attention with a convolution"
    ],
    "answer": 0,
    "why": "Sharing K and V heads across groups of query heads reduces the memory needed to cache keys and values during generation."
  },
  {
    "q": "Why does a Transformer need positional information at all?",
    "options": [
      "Without it the embeddings would be too small",
      "Attention on its own treats the input as an unordered set, so token order would be invisible",
      "To make the vocabulary larger",
      "Positional information is only needed for images"
    ],
    "answer": 1,
    "why": "Self-attention is permutation-equivariant. Positional encodings or rotary embeddings inject order."
  },
  {
    "q": "What is the key idea of rotary position embeddings (RoPE)?",
    "options": [
      "Add a learned vector for each absolute position to the token embedding",
      "Subtract a fixed penalty proportional to distance from each attention score",
      "Rotate query and key vectors by position-dependent angles so their dot product depends on relative position",
      "Sort the tokens before attention"
    ],
    "answer": 2,
    "why": "Rotating q and k by angles proportional to position makes q.k depend on the offset between positions. ALiBi is the method that adds a distance penalty."
  },
  {
    "q": "Why does the same sentence often cost more tokens in some languages than in English?",
    "options": [
      "Non-English text is encrypted before tokenisation",
      "Tokenisers always split non-English text into single bytes",
      "Those languages have longer words",
      "BPE merges are learned from training data that is often dominated by English, so other languages get fewer long merged tokens"
    ],
    "answer": 3,
    "why": "Frequent strings in the tokeniser's training data become single tokens. Under-represented languages are split into more, shorter pieces."
  },
  {
    "q": "What is the main difference between a token embedding and a sentence embedding?",
    "options": [
      "A token embedding is a row of a learned lookup table for one token; a sentence embedding is one vector summarising a whole text, for example by pooling contextual vectors",
      "They are the same thing with different names",
      "Sentence embeddings are always one-hot vectors",
      "Token embeddings are only used in retrieval systems"
    ],
    "answer": 0,
    "why": "The embedding table maps token IDs to vectors at the input. Sentence embedding models produce a single vector per text for similarity search and RAG."
  },
  {
    "q": "If the context length doubles, roughly how does the cost of computing the full attention score matrix change?",
    "options": [
      "It stays the same",
      "It doubles",
      "It roughly quadruples",
      "It halves"
    ],
    "answer": 2,
    "why": "The score matrix has T x T entries, so its cost grows with the square of the sequence length."
  },
  {
    "q": "The 'lost in the middle' effect refers to:",
    "options": [
      "Tokens in the middle of the vocabulary being sampled less often",
      "Models using information at the start or end of a long context more reliably than information placed in the middle",
      "The middle layers of a Transformer being unused",
      "Training data being lost during pretraining"
    ],
    "answer": 1,
    "why": "Studies of long-context retrieval found accuracy is often lower when the relevant passage sits in the middle of the context."
  },
  {
    "q": "During supervised fine-tuning of a chat model, why is the loss usually computed only on the response tokens?",
    "options": [
      "Prompt tokens cannot be tokenised",
      "It makes the model forget its pretraining",
      "The model should learn to produce good responses, not to predict the user's prompt",
      "It doubles the context window"
    ],
    "answer": 2,
    "why": "Masking prompt tokens from the loss focuses learning on the assistant's behaviour."
  },
  {
    "q": "Which phase of LLM inference processes the whole prompt in parallel and produces the first output token?",
    "options": [
      "Decode",
      "Pretraining",
      "Fine-tuning",
      "Prefill"
    ],
    "answer": 3,
    "why": "Prefill runs the prompt through the model in one parallel pass and fills the KV cache. Decode then generates one token per step."
  },
  {
    "q": "Logits are [2.0, 1.0, 0.0]. What happens to the resulting distribution as the temperature approaches 0?",
    "options": [
      "It becomes uniform",
      "It approaches putting all probability on the first token, like greedy decoding",
      "It is unchanged",
      "It puts all probability on the last token"
    ],
    "answer": 1,
    "why": "Dividing logits by a small temperature exaggerates their differences, so softmax concentrates on the largest logit."
  },
  {
    "q": "Top-p (nucleus) sampling with p = 0.9 keeps:",
    "options": [
      "The smallest set of most likely tokens whose cumulative probability reaches at least 0.9",
      "Exactly the 9 most likely tokens",
      "Every token with probability above 0.9",
      "A random 90% of the vocabulary"
    ],
    "answer": 0,
    "why": "Top-p adapts the number of candidates to the shape of the distribution, unlike top-k, which always keeps a fixed number."
  }
]
```
