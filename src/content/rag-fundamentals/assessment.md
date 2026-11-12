This quiz covers the whole RAG pipeline from this module: why RAG is used, ingestion and parsing, chunking, embeddings and similarity, retrieval, context construction, answer generation and evaluation. Aim for at least 13 out of 16 before moving on to Vector Databases.

```quiz
[
  {
    "q": "Which pair of stages forms the query-time path of a RAG system?",
    "options": [
      "Parsing documents and chunking them",
      "Embedding the question and retrieving similar chunks",
      "Hashing documents and updating the registry",
      "Re-embedding the corpus and rebuilding the index"
    ],
    "answer": 1,
    "why": "The query path runs per question: embed the question, retrieve, build the prompt and generate. The other options are ingestion-time work."
  },
  {
    "q": "A team needs an assistant that answers from 50,000 frequently updated support articles, with citations and per-customer access control. Which approach fits best?",
    "options": [
      "Fine-tune a model on the articles every night",
      "Paste all the articles into a long context window",
      "RAG over an index of the articles, filtered by customer permissions",
      "Use a larger model with no external data"
    ],
    "answer": 2,
    "why": "RAG handles large, changing corpora, supports citations, and allows permission filtering at retrieval time. Fine-tuning and long context fail on several of these requirements."
  },
  {
    "q": "During incremental ingestion, a document's content hash differs from the stored one. What is the correct sequence?",
    "options": [
      "Delete all chunks with its `doc_id`, then parse, chunk, embed and insert the new version",
      "Insert the new chunks alongside the old ones",
      "Skip it until the next full rebuild",
      "Change its `doc_id` to the new hash"
    ],
    "answer": 0,
    "why": "Deleting first guarantees no stale chunks survive if the new version produces fewer chunks. The `doc_id` must stay stable."
  },
  {
    "q": "Text extraction from a PDF returns almost no characters, although the file has 40 pages. What is the most likely explanation?",
    "options": [
      "The PDF uses a rare font",
      "The pages are mostly tables",
      "The embedding model's input limit was exceeded",
      "It is a scanned document without a text layer and needs OCR"
    ],
    "answer": 3,
    "why": "Scanned PDFs contain images of pages. Without a text layer, extraction finds nothing until OCR is applied."
  },
  {
    "q": "What does this code print?\n\n```python\nwords = list(range(10))\nstep = 4 - 1\nprint([words[s:s + 4] for s in range(0, 7, step)])\n```",
    "options": [
      "[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]",
      "[[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]",
      "[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]",
      "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]"
    ],
    "answer": 1,
    "why": "Size 4 with overlap 1 gives a step of 3, so windows start at 0, 3 and 6, each sharing one item with the previous window."
  },
  {
    "q": "Why does structure-aware chunking on headings usually beat fixed-size chunking for a well-formatted handbook?",
    "options": [
      "It produces chunks of exactly equal size",
      "It avoids the need for embeddings",
      "Chunks follow the author's topic boundaries and carry a heading path for context and citation",
      "It always produces fewer tokens"
    ],
    "answer": 2,
    "why": "Headings mark topic boundaries. Splitting on them keeps each chunk on one topic and gives it section metadata."
  },
  {
    "q": "An embedding cache is keyed only on the SHA-256 of the chunk text. What goes wrong when the team changes embedding models?",
    "options": [
      "The cache serves vectors from the old model, which are not comparable with new query vectors",
      "Hashes change when the model changes",
      "The cache becomes too large to store",
      "Nothing, because text hashes are model independent"
    ],
    "answer": 0,
    "why": "Each model has its own vector space. The cache key must include the model name so a model switch causes a clean miss."
  },
  {
    "q": "Two unit-length vectors have cosine similarity 0.8. What is the Euclidean distance between them?",
    "options": [
      "0.2",
      "0.4",
      "About 0.894",
      "About 0.632"
    ],
    "answer": 3,
    "why": "For unit vectors, distance squared = 2 - 2cos = 0.4, so the distance is the square root of 0.4, about 0.632."
  },
  {
    "q": "Embeddings are not normalised, and the index uses raw dot product although the model was trained for cosine similarity. What is the likely effect?",
    "options": [
      "An error when the index is created",
      "Long vectors are systematically over-ranked, silently lowering retrieval quality",
      "Identical results to cosine",
      "All scores become negative"
    ],
    "answer": 1,
    "why": "The dot product grows with vector length, so without normalisation it favours long vectors regardless of direction. Nothing raises an error."
  },
  {
    "q": "Which query is the strongest argument for adding BM25 alongside dense retrieval?",
    "options": [
      "\"What's our policy on working from home?\"",
      "\"I feel unwell, who do I tell?\"",
      "\"What does error code E-4012 mean?\"",
      "\"How much holiday do I get?\""
    ],
    "answer": 2,
    "why": "Exact identifiers are rare tokens that lexical matching handles well, while dense embeddings often represent them weakly."
  },
  {
    "q": "What problem does maximal marginal relevance (MMR) address?",
    "options": [
      "Retrieved chunks being near-duplicates of each other",
      "Queries in different languages",
      "Chunks that exceed the embedding model's input limit",
      "Slow vector search"
    ],
    "answer": 0,
    "why": "MMR trades off relevance against similarity to already-selected chunks, so the final set covers more distinct information."
  },
  {
    "q": "What is the purpose of a similarity score threshold in retrieval?",
    "options": [
      "To speed up the matrix multiplication",
      "To guarantee exactly k results",
      "To normalise the vectors",
      "To return nothing when no chunk is relevant enough, so the system can say it does not know"
    ],
    "answer": 3,
    "why": "Top-k always returns k results. A threshold, tuned on labelled data for the specific model, filters out weak matches."
  },
  {
    "q": "Which instruction in a RAG system prompt most directly reduces hallucinated answers when the sources lack the information?",
    "options": [
      "\"Be concise.\"",
      "\"If the sources do not contain the answer, reply exactly: I couldn't find this in the handbook.\"",
      "\"Use a friendly tone.\"",
      "\"Answer in British English.\""
    ],
    "answer": 1,
    "why": "Giving the model an explicit, permitted refusal reduces its tendency to invent a plausible answer, and a fixed sentence is easy to detect in code."
  },
  {
    "q": "A model's answer cites `[4]`, but only three sources were sent in the prompt. What should the application do?",
    "options": [
      "Display the answer unchanged",
      "Fetch a fourth chunk and add it to the citations",
      "Treat it as a failed citation check: flag, strip or regenerate the answer",
      "Increase k for future questions"
    ],
    "answer": 2,
    "why": "A citation to a non-existent source signals possibly invented content and should fail validation."
  },
  {
    "q": "A question has relevance labels `{\"a\": 1}` and the retriever returns `[\"x\", \"y\", \"a\"]`. What are recall@3 and the reciprocal rank?",
    "options": [
      "Recall@3 = 1.0, reciprocal rank = 1/3",
      "Recall@3 = 1/3, reciprocal rank = 1.0",
      "Recall@3 = 1.0, reciprocal rank = 1.0",
      "Recall@3 = 1/3, reciprocal rank = 1/3"
    ],
    "answer": 0,
    "why": "The only relevant chunk is in the top 3, so recall@3 is 1. It first appears at rank 3, so the reciprocal rank is 1/3."
  },
  {
    "q": "An answer is fully correct in the real world but includes one claim not present in the retrieved context. Which metric penalises this?",
    "options": [
      "Recall@k",
      "Answer relevance",
      "MRR",
      "Faithfulness"
    ],
    "answer": 3,
    "why": "Faithfulness measures whether each claim is supported by the retrieved context, regardless of whether it happens to be true."
  }
]
```
