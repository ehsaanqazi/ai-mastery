This quiz covers the whole Linear Algebra module: vectors and matrices, the core operations, transpose and inverse, the identity, the dot product, eigenvectors, vector spaces and norms, and where each appears in AI. Aim for at least 13 out of 16 before moving on to Calculus.

```quiz
[
  {
    "q": "A dataset has 500 examples and 12 features. Using the machine learning convention, what is the shape of its data matrix `X`?",
    "options": ["(12, 500)", "(500, 12)", "(500,)", "(6000,)"],
    "answer": 1,
    "why": "Rows are examples and columns are features, so X is 500 by 12."
  },
  {
    "q": "In NumPy, which expression is the matrix product of `A` and `B`?",
    "options": ["`A * B`", "`A + B`", "`A @ B`", "`A ** B`"],
    "answer": 2,
    "why": "`@` is matrix multiplication. `*` is the element-wise product, which is a different operation."
  },
  {
    "q": "What is the shape of the product of a (64, 512) matrix and a (512, 10) matrix?",
    "options": ["(64, 10)", "(512, 512)", "(10, 64)", "Undefined"],
    "answer": 0,
    "why": "The inner dimensions match (512) and drop out, leaving the outer dimensions 64 by 10."
  },
  {
    "q": "Compute the matrix product `[[1, 2], [3, 4]] @ [[1], [1]]`.",
    "options": ["`[[1, 2], [3, 4]]`", "`[[4], [6]]`", "`[[10]]`", "`[[3], [7]]`"],
    "answer": 3,
    "why": "Row 1: 1 + 2 = 3. Row 2: 3 + 4 = 7. Multiplying by a vector of ones adds up each row."
  },
  {
    "q": "Which property does matrix multiplication have in general?",
    "options": ["Commutative: `AB = BA`", "Associative: `(AB)C = A(BC)`", "It requires both matrices to be square", "It multiplies matching entries"],
    "answer": 1,
    "why": "Matrix multiplication is associative but not commutative. Order matters because it composes transformations."
  },
  {
    "q": "Which equals the transpose of `AB`?",
    "options": ["`B^T A^T`", "`A^T B^T`", "`BA`", "`A^-1 B^-1`"],
    "answer": 0,
    "why": "The transpose of a product reverses the order of the factors."
  },
  {
    "q": "The matrix `[[2, 4], [1, 2]]` has determinant 0. What follows?",
    "options": ["Its inverse is `[[2, -4], [-1, 2]]`", "It is the identity", "It has no inverse, because it flattens the plane onto a line", "It is symmetric"],
    "answer": 2,
    "why": "Determinant zero means areas are squashed to zero; information is lost, so the transformation cannot be undone."
  },
  {
    "q": "Why is `np.linalg.solve(A, b)` preferred to `np.linalg.inv(A) @ b`?",
    "options": ["`inv` only works on 2 by 2 matrices", "`solve` returns a matrix rather than a vector", "They give different answers even in exact arithmetic", "`solve` is faster and more numerically stable"],
    "answer": 3,
    "why": "Forming the inverse explicitly costs more and amplifies rounding error, especially for nearly singular matrices."
  },
  {
    "q": "A one-hot vector for token 3 is multiplied by an embedding matrix `E` with one row per token. The result is:",
    "options": ["Row 3 of `E`", "The sum of all rows of `E`", "The identity matrix", "Column 3 of `E`"],
    "answer": 0,
    "why": "A one-hot vector is a row of the identity, so the product selects exactly one row of E: an embedding lookup."
  },
  {
    "q": "Ridge regression solves `(X^T X + lambda I) w = X^T y`. What does adding `lambda I` achieve?",
    "options": ["It transposes the data", "It makes the matrix invertible and penalises large weights", "It removes the intercept", "It sets some weights exactly to zero"],
    "answer": 1,
    "why": "Adding a positive amount to the diagonal keeps the matrix away from singular and corresponds to an L2 penalty. Exact zeros come from L1."
  },
  {
    "q": "Two embeddings have cosine similarity close to -1. They:",
    "options": ["Are nearly identical in meaning", "Are unrelated", "Point in nearly opposite directions", "Have very different lengths"],
    "answer": 2,
    "why": "Cosine similarity is the cosine of the angle between them; -1 means 180 degrees. It says nothing about length."
  },
  {
    "q": "What does `sum(x * y for x, y in zip([1, 2], [3, -1]))` return?",
    "options": ["`[3, -2]`", "5", "1", "7"],
    "answer": 2,
    "why": "It is the dot product: 1 times 3 + 2 times -1 = 3 - 2 = 1."
  },
  {
    "q": "In PCA, the first principal component is:",
    "options": ["The eigenvector of the covariance matrix with the largest eigenvalue", "The feature with the largest mean", "The first column of the data matrix", "The eigenvector with the smallest eigenvalue"],
    "answer": 0,
    "why": "Each eigenvalue of the covariance matrix is the variance along its eigenvector, so the largest marks the direction of greatest spread."
  },
  {
    "q": "A recurrent network repeatedly multiplies its state by a matrix whose eigenvalues all have magnitude below 1. Over many steps the signal tends to:",
    "options": ["Explode", "Stay exactly constant", "Rotate without changing size", "Vanish towards zero"],
    "answer": 3,
    "why": "Each step scales the component along each eigenvector by its eigenvalue, so magnitudes below 1 shrink exponentially."
  },
  {
    "q": "What is the rank of `[[1, 2, 3], [2, 4, 6], [1, 0, 0]]`?",
    "options": ["1", "2", "3", "0"],
    "answer": 1,
    "why": "The second row is twice the first, so only two rows are independent and the rank is 2."
  },
  {
    "q": "Which regularisation penalty encourages sparse weights, with many exactly zero?",
    "options": ["The squared L2 norm", "The L-infinity norm", "The L1 norm", "No penalty can do this"],
    "answer": 2,
    "why": "The L1 penalty (lasso) has corners on the axes, so solutions often have some coordinates exactly zero. L2 shrinks weights smoothly."
  }
]
```
