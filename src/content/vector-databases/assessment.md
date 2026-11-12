This quiz covers the whole Vector Databases module: what vector databases add over arrays, index types and their trade-offs, exact and approximate search, filtering, hybrid search and reranking, and the five tools (FAISS, ChromaDB, Pinecone, Qdrant and Weaviate), with a comparison table at the end of the [Weaviate lesson](/learn/vector-databases/weaviate). Aim for at least 13 out of 16 before moving on to Advanced RAG.

```quiz
[
  {
    "q": "Which capability is the main reason to move from a NumPy matrix of embeddings to a vector database?",
    "options": [
      "Embeddings become more accurate",
      "Persistence, efficient updates, metadata filtering, ANN indexes and concurrent access",
      "Cosine similarity becomes available",
      "The embedding dimension can be changed freely"
    ],
    "answer": 1,
    "why": "A vector database adds database properties (durability, CRUD, filters, scale) plus fast approximate indexes; the embeddings themselves are unchanged."
  },
  {
    "q": "For unit-length vectors a and b, which identity holds?",
    "options": [
      "`||a - b||^2 = 2 - 2 a·b`",
      "`||a - b||^2 = a·b`",
      "`||a - b|| = 1 - a·b`",
      "`a·b = 0`"
    ],
    "answer": 0,
    "why": "Expanding the square gives ||a||^2 + ||b||^2 - 2a·b = 2 - 2a·b, so L2, cosine and dot product rank identically."
  },
  {
    "q": "About how much raw float32 memory do 10 million vectors of dimension 1,024 need?",
    "options": ["About 4 GiB", "About 10 GiB", "About 38 GiB", "About 380 GiB"],
    "answer": 2,
    "why": "10,000,000 x 1,024 x 4 bytes = 40.96 billion bytes, about 38 GiB."
  },
  {
    "q": "Which index type compresses each vector into a short sequence of one-byte codes?",
    "options": ["Flat", "HNSW", "IVF-Flat", "Product quantisation (PQ)"],
    "answer": 3,
    "why": "PQ splits a vector into sub-vectors and replaces each with the one-byte ID of its nearest sub-space centroid."
  },
  {
    "q": "What is the main memory overhead of HNSW compared with a flat index?",
    "options": [
      "The k-means centroids",
      "The neighbour links stored for every vector",
      "PQ codebooks",
      "A copy of every vector per layer"
    ],
    "answer": 1,
    "why": "HNSW stores the full vectors plus graph links, roughly 2M neighbour IDs per node on the bottom layer."
  },
  {
    "q": "What does `top = np.argpartition(-scores, k - 1)[:k]` followed by `top = top[np.argsort(-scores[top])]` compute?",
    "options": [
      "The k lowest scores in random order",
      "All indices sorted by score",
      "The indices of the k highest scores, best first",
      "The k-th highest score only"
    ],
    "answer": 2,
    "why": "argpartition isolates the k largest scores in linear time; argsort then orders just those k."
  },
  {
    "q": "An IVF index returns recall@10 of 0.80 with `nprobe = 4`. What is the cheapest first step to raise recall?",
    "options": [
      "Increase `nprobe`",
      "Retrain with more lists",
      "Switch to PQ",
      "Reduce k to 5"
    ],
    "answer": 0,
    "why": "nprobe is a query-time setting; scanning more lists raises recall without rebuilding anything."
  },
  {
    "q": "How is recall@k of an ANN index measured?",
    "options": [
      "By asking users to rate results",
      "By the average overlap between the index's top k and the exact top k from brute force, divided by k",
      "By the fraction of queries that return any result",
      "By the mean similarity score of returned vectors"
    ],
    "answer": 1,
    "why": "Exact brute-force search provides the ground truth; recall is the fraction of true neighbours the index finds."
  },
  {
    "q": "A filter matches 0.5% of a collection. Post-filtering the top 10 results usually returns:",
    "options": [
      "Exactly 10 results",
      "More than 10 results",
      "An error",
      "Few or no results"
    ],
    "answer": 3,
    "why": "About 99.5% of the unfiltered top 10 fail the filter and are discarded."
  },
  {
    "q": "Why is filtered search difficult for HNSW?",
    "options": [
      "Skipping non-matching nodes breaks the graph's connectivity, so matches can become unreachable",
      "HNSW cannot store IDs",
      "Filters require L2 distance",
      "HNSW only supports one query at a time"
    ],
    "answer": 0,
    "why": "The greedy walk depends on links through all nodes; heavy filtering fragments the navigable graph."
  },
  {
    "q": "With reciprocal rank fusion (c = 60), a document ranked 1st by BM25 and 4th by dense search scores:",
    "options": [
      "1/61 x 1/64",
      "1/61 + 1/64",
      "1/5",
      "(1 + 4) / 60"
    ],
    "answer": 1,
    "why": "RRF sums 1 / (c + rank) over the rankers: 1/61 + 1/64."
  },
  {
    "q": "Why does hybrid search usually beat dense search alone?",
    "options": [
      "BM25 is always more accurate",
      "Keyword and dense retrieval fail on different queries, such as exact codes versus paraphrases",
      "It uses larger embeddings",
      "It removes the need for an index"
    ],
    "answer": 1,
    "why": "Combining complementary retrievers recovers documents that either one alone would miss."
  },
  {
    "q": "Why is a cross-encoder used only to rerank a short candidate list?",
    "options": [
      "It cannot score more than 10 documents",
      "It produces vectors that do not fit in an index",
      "It needs a full forward pass per query-document pair, so nothing can be precomputed",
      "It only works with BM25 candidates"
    ],
    "answer": 2,
    "why": "Joint encoding makes it accurate but expensive per candidate, so it is applied after a cheap first stage."
  },
  {
    "q": "Which FAISS setup gives cosine similarity search?",
    "options": [
      "`faiss.IndexFlatL2` on raw vectors",
      "`faiss.normalize_L2` on the vectors, then `faiss.IndexFlatIP`",
      "`faiss.IndexIVFFlat` without training",
      "`faiss.IndexHNSWFlat` with `M = 0`"
    ],
    "answer": 1,
    "why": "Inner product on unit vectors equals cosine similarity."
  },
  {
    "q": "A Chroma cosine collection and a Qdrant cosine collection hold the same vectors. Chroma reports 0.0098 for the top result. What will Qdrant report as its score?",
    "options": [
      "0.0098",
      "About 0.99",
      "About 0.0001",
      "The same rank but no score"
    ],
    "answer": 1,
    "why": "Chroma reports cosine distance (1 - cos); Qdrant reports cosine similarity, so 1 - 0.0098 = 0.9902."
  },
  {
    "q": "A team needs a self-hosted database with rich payload filtering, local in-process mode for unit tests, and quantisation options. Which tool fits best?",
    "options": ["FAISS", "Pinecone", "Qdrant", "A NumPy array"],
    "answer": 2,
    "why": "Qdrant is open source and self-hostable, filters on indexed payloads, offers quantisation, and its Python client has a local mode for tests."
  }
]
```
