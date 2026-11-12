This quiz covers the whole Unsupervised Learning module: k-means, hierarchical clustering, DBSCAN, PCA, dimensionality reduction for visualisation, and anomaly detection. Aim for at least 13 out of 16 before moving on to Model Evaluation.

```quiz
[
  {
    "q": "In each iteration of k-means, which two steps alternate?",
    "options": [
      "Merge the closest clusters, then recompute distances",
      "Assign each point to its nearest centroid, then move each centroid to the mean of its points",
      "Mark core points, then expand clusters from them",
      "Compute eigenvectors, then project the data"
    ],
    "answer": 1,
    "why": "Lloyd's algorithm alternates the assign and update steps, and neither can increase inertia."
  },
  {
    "q": "A silhouette score close to -1 for a point means:",
    "options": [
      "The point is probably in the wrong cluster",
      "The point is at the exact centre of its cluster",
      "The point is a DBSCAN core point",
      "The clustering has converged"
    ],
    "answer": 0,
    "why": "s(i) = (b - a) / max(a, b). A negative value means the point is on average closer to another cluster than to its own."
  },
  {
    "q": "`X` holds 300 points drawn uniformly at random in a square. What does `len(set(KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(X)))` return?",
    "options": [
      "1, because uniform data has no clusters",
      "A random number between 1 and 5",
      "5",
      "It raises an error"
    ],
    "answer": 2,
    "why": "K-means always returns exactly k non-empty clusters, even for structureless data. That is why its output must be validated."
  },
  {
    "q": "Why should you standardise features before k-means, hierarchical clustering or DBSCAN?",
    "options": [
      "These algorithms only accept values between 0 and 1",
      "Standardising makes them run faster",
      "It guarantees the correct number of clusters",
      "They rely on distances, which are dominated by features with large numeric ranges"
    ],
    "answer": 3,
    "why": "Without scaling, a feature measured in thousands swamps one measured in units, regardless of which carries the structure."
  },
  {
    "q": "Which linkage merges the pair of clusters that least increases the total within-cluster sum of squares?",
    "options": [
      "Single",
      "Complete",
      "Ward",
      "Average"
    ],
    "answer": 2,
    "why": "Ward linkage optimises the same quantity as k-means, which is why it requires Euclidean distance."
  },
  {
    "q": "SciPy's `linkage` returns a row `[4, 7, 0.604, 3]`. What does it mean?",
    "options": [
      "Clusters 4 and 7 merged at distance 0.604, forming a cluster of 3 points",
      "Point 4 has 7 neighbours within distance 0.604",
      "There are 3 clusters after cutting at height 0.604",
      "Clusters 4 and 7 are 3 units apart"
    ],
    "answer": 0,
    "why": "Each row records the two cluster IDs merged, the linkage distance, and the size of the new cluster."
  },
  {
    "q": "On two interleaving crescent shapes, which methods are most likely to recover the crescents?",
    "options": [
      "K-means and Ward linkage",
      "DBSCAN and single linkage",
      "PCA and k-means",
      "Complete linkage and One-Class SVM"
    ],
    "answer": 1,
    "why": "Both follow chains of nearby points and can trace non-convex shapes. K-means and Ward favour compact, round clusters."
  },
  {
    "q": "In DBSCAN with `eps=1.5` and `min_samples=3`, a point has only itself and one other point within 1.5, but that other point is a core point. The first point is:",
    "options": [
      "A core point",
      "Noise",
      "A border point",
      "A centroid"
    ],
    "answer": 2,
    "why": "It is not dense enough to be core, but it lies within eps of a core point, so it joins that cluster as a border point."
  },
  {
    "q": "You plot sorted distances to each point's 5th nearest neighbour. The curve is flat around 0.2 and shoots up after 0.35. A sensible starting `eps` for DBSCAN with `min_samples=5` is:",
    "options": [
      "0.01",
      "Around 0.3",
      "The maximum distance on the curve",
      "5"
    ],
    "answer": 1,
    "why": "The knee of the k-distance curve separates points inside dense regions from sparse noise."
  },
  {
    "q": "The principal components of standardised data are the eigenvectors of which matrix?",
    "options": [
      "The data matrix itself",
      "The identity matrix",
      "The distance matrix between points",
      "The correlation matrix of the features"
    ],
    "answer": 3,
    "why": "Standardised features have unit variance, so their covariance matrix is the correlation matrix. PCA diagonalises it."
  },
  {
    "q": "Eigenvalues of a covariance matrix are 4, 3, 2 and 1. How many components do you need to keep at least 70% of the variance?",
    "options": [
      "1",
      "2",
      "3",
      "4"
    ],
    "answer": 1,
    "why": "Cumulative ratios are 0.4, 0.7, 0.9 and 1.0, so two components reach 70%."
  },
  {
    "q": "PCA with 3 components keeps 85% of the variance of standardised data. What is the mean squared reconstruction error per feature after `inverse_transform`?",
    "options": [
      "About 0.15",
      "About 0.85",
      "Exactly 0",
      "It cannot be determined"
    ],
    "answer": 0,
    "why": "For standardised data the reconstruction error equals the fraction of variance discarded: 1 - 0.85."
  },
  {
    "q": "Why should you not use t-SNE coordinates as input features for a production classifier?",
    "options": [
      "t-SNE output always has negative values",
      "t-SNE is stochastic and scikit-learn's version cannot transform new points consistently",
      "t-SNE removes the labels",
      "t-SNE only works on images"
    ],
    "answer": 1,
    "why": "Each run gives a different layout and there is no `transform` for new data, so features would not be stable."
  },
  {
    "q": "To isolate a point, Isolation Forest needs on average 3 random splits for point A and 11 for point B. Which is more likely anomalous?",
    "options": [
      "B, because it needs more splits",
      "Neither: path length is unrelated to anomaly",
      "A, because it is isolated quickly",
      "Both equally"
    ],
    "answer": 2,
    "why": "Anomalies are few and different, so random splits separate them from everything else after few cuts."
  },
  {
    "q": "Your anomaly detector flags nothing and reports 99.7% accuracy on data where 0.3% of records are fraud. What is wrong?",
    "options": [
      "Accuracy is meaningless under this imbalance; use precision, recall, average precision or precision at k",
      "Nothing, 99.7% is excellent",
      "The detector needs more trees",
      "Fraud detection cannot be evaluated"
    ],
    "answer": 0,
    "why": "Predicting 'normal' for everything achieves the base rate. Evaluate the top of the ranking against confirmed anomalies."
  },
  {
    "q": "You want to remove near-duplicate chunks from a RAG corpus using embeddings, keeping unique chunks untouched. Which approach fits best?",
    "options": [
      "K-means with k equal to the number of chunks divided by 2",
      "PCA to 2 dimensions, then keep the chunks with the largest PC1",
      "Isolation Forest, removing the inliers",
      "DBSCAN with cosine distance, a small `eps` and `min_samples=2`, keeping one chunk per cluster and every noise point"
    ],
    "answer": 3,
    "why": "Tight groups of near-identical chunks become clusters, while unique chunks are labelled noise and kept as they are."
  }
]
```
