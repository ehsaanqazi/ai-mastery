This quiz covers the whole NLP Fundamentals module: preprocessing, tokenization, stemming and lemmatisation, stop words, bag of words, TF-IDF, n-grams, word embeddings, Word2Vec, GloVe, named entity recognition, sentiment analysis and text classification. Aim for at least 13 out of 16 before moving on to Advanced NLP.

```quiz
[
  {
    "q": "You are preparing text for two systems: a BM25 keyword index and a transformer embedding model. Which preprocessing plan is most appropriate?",
    "options": ["Stem and remove stop words for both", "Leave both completely raw, including HTML", "Stem and remove stop words for BM25; only strip markup and junk for the transformer", "Lowercase and stem for the transformer only"],
    "answer": 2,
    "why": "Keyword search benefits from aggressive normalisation applied to documents and queries; transformers expect natural text and have their own subword tokenizer."
  },
  {
    "q": "In BPE training, the current symbol-pair counts are (e, s) = 9, (l, o) = 7 and (w, e) = 8. What happens next?",
    "options": ["(e, s) is merged into the new symbol `es` everywhere", "(l, o) is merged because it comes first alphabetically", "All three pairs are merged", "(w, e) is merged because it is in the middle"],
    "answer": 0,
    "why": "BPE merges the most frequent adjacent pair, then recounts with the new symbol."
  },
  {
    "q": "Why do subword tokenizers avoid the out-of-vocabulary problem?",
    "options": ["Their vocabulary contains every English word", "They ignore rare words", "They replace unknown words with `UNK`", "Any unseen word can be split into smaller known pieces, down to characters or bytes"],
    "answer": 3,
    "why": "Rare words decompose into known subwords, so nothing is ever unrepresentable."
  },
  {
    "q": "Which pair shows the key difference between stemming and lemmatisation?",
    "options": ["Both turn `studies` into `studi`", "`ran` stays `ran` with a stemmer but becomes `run` with a lemmatiser", "Stemmers need a POS tagger, lemmatisers do not", "Lemmatisers are always faster"],
    "answer": 1,
    "why": "Stemmers strip suffixes by rule and cannot handle irregular forms; lemmatisers use a dictionary and part of speech."
  },
  {
    "q": "What does this print?\n\n`CountVectorizer(stop_words=\"english\").fit([\"The battery is not good\", \"The battery is good\"]).get_feature_names_out()`",
    "options": ["`['battery' 'good']`", "`['battery' 'good' 'not']`", "`['the' 'battery' 'is' 'not' 'good']`", "`['not' 'good']`"],
    "answer": 0,
    "why": "scikit-learn's English stop-word list contains `the`, `is` and `not`, so the two reviews end up with identical features."
  },
  {
    "q": "A fitted `CountVectorizer` transforms a document whose words were all unseen during fitting. What does the model receive?",
    "options": ["An error", "A vector with an UNK count", "An all-zero vector, silently", "A refitted vocabulary"],
    "answer": 2,
    "why": "Unknown words have no column and are ignored, so the vector is all zeros and no warning is raised."
  },
  {
    "q": "With scikit-learn's default smoothed idf, idf(t) = ln((1 + n) / (1 + df(t))) + 1. For n = 3 documents and a term in 1 document, what is idf?",
    "options": ["ln(3) + 1", "ln(2) + 1", "ln(4/3) + 1", "1"],
    "answer": 1,
    "why": "ln((1 + 3) / (1 + 1)) + 1 = ln(2) + 1, about 1.693."
  },
  {
    "q": "Why does `TfidfVectorizer` L2-normalise rows by default?",
    "options": ["To make long documents score higher", "To make the matrix dense", "To remove stop words", "So document length does not inflate scores and dot products equal cosine similarities"],
    "answer": 3,
    "why": "Unit-length rows compare proportions, and the dot product of unit vectors is their cosine similarity."
  },
  {
    "q": "Which feature setting lets a linear sentiment model distinguish 'not good' from 'good'?",
    "options": ["`binary=True`", "`stop_words=\"english\"`", "`ngram_range=(1, 2)`", "`max_df=0.5`"],
    "answer": 2,
    "why": "Bigrams make `not good` its own feature with its own weight."
  },
  {
    "q": "A bigram language model has seen `the cat` 3 times and `the` 12 times. What is the maximum likelihood estimate of P(cat | the)?",
    "options": ["0.25", "0.75", "3", "1/12"],
    "answer": 0,
    "why": "count(the, cat) / count(the) = 3 / 12 = 0.25."
  },
  {
    "q": "What is the main idea behind all classical word embedding methods?",
    "options": ["Words with similar spellings have similar meanings", "Every word should be orthogonal to every other", "Rare words carry the most meaning", "Words that occur in similar contexts have similar meanings"],
    "answer": 3,
    "why": "The distributional hypothesis: meaning is inferred from the company a word keeps."
  },
  {
    "q": "In Word2Vec's negative sampling, what is each training example?",
    "options": ["A softmax over the whole vocabulary", "A real (centre, context) pair to score high and k random pairs to score low", "A labelled sentence", "A co-occurrence count to regress"],
    "answer": 1,
    "why": "Negative sampling replaces the full softmax with k + 1 binary classifications, making updates cheap."
  },
  {
    "q": "What does GloVe fit with weighted least squares?",
    "options": ["The dot product of word and context vectors plus biases to the log co-occurrence count", "The probability that a pair is real", "The raw counts", "Cosine similarities between all pairs"],
    "answer": 0,
    "why": "GloVe minimises the weighted squared error between w_i·w̃_j + b_i + b̃_j and log X_ij over non-zero entries."
  },
  {
    "q": "Tag the sentence 'Ada met Charles Babbage' (Ada and Charles Babbage are people) in BIO format.",
    "options": ["I-PER O B-PER B-PER", "B-PER O I-PER I-PER", "B-PER O B-PER I-PER", "B-PER O B-PER B-PER"],
    "answer": 2,
    "why": "Each entity starts with B-; continuation tokens of the same entity use I-."
  },
  {
    "q": "A support-ticket classifier reports 96 percent accuracy, but only 4 percent of tickets are urgent, and urgent recall is 0.05. What is happening?",
    "options": ["The model is excellent", "It mostly predicts the majority class; use class weights or threshold tuning, and report per-class recall and macro-F1", "The test set is too large", "Accuracy and recall are unrelated"],
    "answer": 1,
    "why": "On imbalanced data a majority-class predictor scores high accuracy while missing the class that matters."
  },
  {
    "q": "You have 300 texts per day to classify, no labelled data, and categories that change monthly. What is the most sensible starting point?",
    "options": ["Fine-tune a transformer immediately", "A sentiment lexicon", "Train naive Bayes on the label names", "An LLM prompt with the current label list and constrained output, evaluated on a small human-labelled sample"],
    "answer": 3,
    "why": "LLMs need no training data and adapt to changing labels by editing the prompt; at low volume the cost is manageable, but evaluation is still required."
  }
]
```
