This quiz covers the whole Advanced RAG module: advanced chunking, query transformation and expansion, hybrid retrieval and fusion, reranking models, multi-query and parent-child retrieval, contextual retrieval, Graph RAG, multimodal RAG and evaluation. Aim for at least 13 out of 16 before moving on to AI Agent Fundamentals.

```quiz
[
  {
    "q": "Semantic chunking computes the cosine similarity between each sentence and the next. Given gaps `[0.51, 0.12, 0.0, 0.39, 0.15]` and a threshold of 0.1, how many chunks are produced from the six sentences?",
    "options": [
      "1",
      "2",
      "3",
      "6"
    ],
    "answer": 1,
    "why": "Only the gap of 0.0 (between sentences 3 and 4) is below 0.1, so there is one boundary and therefore two chunks."
  },
  {
    "q": "Which technique embeds a whole document with a long-context model first and pools token vectors into chunk vectors afterwards?",
    "options": [
      "Proposition indexing",
      "Contextual retrieval",
      "Late chunking",
      "Sentence-window retrieval"
    ],
    "answer": 2,
    "why": "Late chunking lets every token vector absorb document-wide context before chunk boundaries are applied."
  },
  {
    "q": "In a chat, the user's latest message is \"How long does its battery last?\". Which transformation is essential before retrieval?",
    "options": [
      "Conversational query condensation",
      "Step-back prompting",
      "HyDE",
      "Pseudo-relevance feedback"
    ],
    "answer": 0,
    "why": "Condensation rewrites the follow-up as a standalone question that names the product the pronoun refers to."
  },
  {
    "q": "Why can HyDE improve retrieval even when the hypothetical answer it generates is factually wrong?",
    "options": [
      "The hypothetical answer is shown to the user, who corrects it",
      "The retriever ignores the hypothetical answer's content",
      "Wrong answers have lower embedding norms",
      "It is used only as a search query, and answers resemble other answers in vocabulary and form"
    ],
    "answer": 3,
    "why": "HyDE closes the question/answer vocabulary gap; the generated text is never presented as the answer."
  },
  {
    "q": "Pseudo-relevance feedback updates the query as `q' = alpha * q + beta * centroid(top results)`. What happens when the first retrieval's top result is off-topic?",
    "options": [
      "The feedback is automatically ignored",
      "The query moves towards the off-topic document, causing query drift",
      "Recall always improves anyway",
      "The query becomes empty"
    ],
    "answer": 1,
    "why": "PRF assumes the top results are relevant; when they are not, their terms pull the query further from the user's intent."
  },
  {
    "q": "In a convex combination `alpha * dense + (1 - alpha) * sparse`, why must the two score lists be normalised first?",
    "options": [
      "BM25 scores are unbounded while cosine similarities lie in [-1, 1], so one would dominate",
      "Normalisation removes irrelevant documents",
      "Vector databases require scores between 0 and 1",
      "It makes the fusion run faster"
    ],
    "answer": 0,
    "why": "Without a common scale, the retriever with larger score magnitudes decides almost every ranking."
  },
  {
    "q": "Using reciprocal rank fusion with k = 60, document X is ranked 1st by one retriever and absent from the other. Document Y is ranked 3rd by both. Which ranks higher?",
    "options": [
      "X, because rank 1 always wins",
      "It depends on the raw similarity scores",
      "They tie",
      "Y, because 2/63 is larger than 1/61"
    ],
    "answer": 3,
    "why": "Y scores 1/63 + 1/63, about 0.032, while X scores 1/61, about 0.016. RRF rewards agreement across lists."
  },
  {
    "q": "A query router sees the query `INV-20931 status`. What is the most sensible retrieval setting?",
    "options": [
      "Dense-heavy, alpha close to 1",
      "HyDE followed by dense search",
      "Keyword-heavy, alpha close to 0",
      "Graph global search"
    ],
    "answer": 2,
    "why": "Identifiers are matched exactly by keyword search, and embedding models represent them poorly."
  },
  {
    "q": "What gives a cross-encoder its accuracy advantage over a bi-encoder, and what is its main cost?",
    "options": [
      "It uses larger vectors; it needs more storage",
      "Query and document are encoded jointly with full attention between them; nothing can be precomputed, so cost grows with the number of candidates",
      "It is trained on more languages; it is slower to train",
      "It uses BM25 internally; it cannot handle synonyms"
    ],
    "answer": 1,
    "why": "Joint encoding lets the model judge this specific pair, but every query requires a forward pass per candidate."
  },
  {
    "q": "In ColBERT-style late interaction, how is a query-document score computed?",
    "options": [
      "Cosine similarity of mean-pooled vectors",
      "Run the pair through a cross-encoder",
      "Count the shared tokens",
      "For each query token, take its maximum similarity to any document token, then sum over query tokens"
    ],
    "answer": 3,
    "why": "This is MaxSim: token-level matching that avoids diluting specific terms into one averaged vector."
  },
  {
    "q": "You want to answer \"I don't know\" when no retrieved passage is relevant enough. What should you do with reranker scores?",
    "options": [
      "Calibrate them on labelled pairs from your domain (for example with Platt scaling) and choose a threshold from precision and recall",
      "Apply a sigmoid and use 0.5 as the threshold",
      "Use the raw score of the top passage without any threshold",
      "Average the scores of all candidates"
    ],
    "answer": 0,
    "why": "Raw reranker scores are not calibrated probabilities; their meaning must be learned from labelled data."
  },
  {
    "q": "What is the defining feature of parent-child retrieval?",
    "options": [
      "Every chunk is summarised by an LLM",
      "Parents and children are fused with RRF",
      "Small child chunks are embedded and searched, and their larger parent chunks are returned to the generator",
      "Queries are split into parent and child sub-questions"
    ],
    "answer": 2,
    "why": "It separates the precise unit used for search from the context-rich unit used for reading."
  },
  {
    "q": "Why does contextual retrieval place the full document before the chunk in the context-generation prompt?",
    "options": [
      "Models read the end of a prompt more carefully",
      "So that all calls for one document share an identical prefix that prompt caching can reuse cheaply",
      "To keep the output short",
      "Because the chunk must be the last line for BM25"
    ],
    "answer": 1,
    "why": "One call per chunk would otherwise pay full price to process the same document many times."
  },
  {
    "q": "Which question most clearly calls for Graph RAG global search over community summaries?",
    "options": [
      "What are the main recurring causes across all 3,000 incident reports?",
      "Which company supplies Helix Motors with battery cells?",
      "What is the refund window for online orders?",
      "What does error code E1043 mean?"
    ],
    "answer": 0,
    "why": "A corpus-wide theme question has no single relevant chunk; community summaries cover the whole corpus."
  },
  {
    "q": "In multimodal RAG over annual reports, what is a good way to make a financial table retrievable for questions about specific cells?",
    "options": [
      "Embed a screenshot of the table only",
      "Store the table as one long chunk of numbers",
      "Drop tables, since text usually repeats them",
      "Index one sentence per row that repeats the caption and column names, and return the full table to the generator"
    ],
    "answer": 3,
    "why": "Row sentences give precise matches, while the full table preserves structure for answering."
  },
  {
    "q": "In error analysis, the gold chunk was not among the retriever's candidates at all. Which category is this, and which fix is irrelevant?",
    "options": [
      "Ranking miss; a reranker is irrelevant",
      "Unfaithful generation; hybrid retrieval is irrelevant",
      "Retrieval miss; improving the generation prompt is irrelevant",
      "Missing content; chunking changes are irrelevant"
    ],
    "answer": 2,
    "why": "If the evidence never reached the candidate set, no prompt can use it; the fix lies in chunking, indexing or retrieval."
  }
]
```
