This quiz covers the whole NumPy module: array shapes and dtypes, indexing and views, broadcasting, vectorisation, matrix operations, numerically stable functions and random number generation. Aim for at least 12 out of 15 before moving on to Pandas.

```quiz
[
  {
    "q": "An array has shape `(32, 3, 224, 224)`. What are its `ndim` and the length of axis 1?",
    "options": ["`ndim` 4, axis 1 has length 3", "`ndim` 4, axis 1 has length 32", "`ndim` 3, axis 1 has length 224", "`ndim` 224, axis 1 has length 3"],
    "answer": 0,
    "why": "`ndim` is the number of entries in the shape tuple. Axes are numbered from 0, so axis 1 is the second entry."
  },
  {
    "q": "For `x` of shape `(4, 5, 6)`, what is the shape of `x.sum(axis=(0, 2))`?",
    "options": ["`(4, 6)`", "`(5, 6)`", "`(5,)`", "`(4, 1, 6)`"],
    "answer": 2,
    "why": "Reducing over axes 0 and 2 removes both of them, leaving only axis 1 of length 5."
  },
  {
    "q": "How many bytes does a float32 array of shape `(1000, 512)` use?",
    "options": ["512,000", "4,096,000", "1,024,000", "2,048,000"],
    "answer": 3,
    "why": "1000 × 512 = 512,000 elements at 4 bytes each is 2,048,000 bytes, about 2 MB."
  },
  {
    "q": "After `a = np.arange(6)`, `b = a[2:5]` and `b[0] = 99`, what is `a[2]`?",
    "options": ["`2`", "`99`", "`5`", "An error"],
    "answer": 1,
    "why": "A basic slice is a view of the same memory, so writing to `b[0]` changes `a[2]`."
  },
  {
    "q": "Which selection returns a copy rather than a view?",
    "options": ["`x[1:4]`", "`x[:, 0]`", "`x[x > 0]`", "`x[..., ::2]`"],
    "answer": 2,
    "why": "Boolean masks and integer-array indexing always return copies. Basic slices return views."
  },
  {
    "q": "`probs` has shape `(4, 10)` and `labels = np.array([3, 0, 9, 1])`. Which expression gives each row's probability for its label?",
    "options": ["`probs[:, labels]`", "`probs[labels]`", "`probs[labels, :]`", "`probs[np.arange(4), labels]`"],
    "answer": 3,
    "why": "Two integer arrays are paired element by element, selecting `(0, 3)`, `(1, 0)`, `(2, 9)` and `(3, 1)`. `probs[:, labels]` would give a `(4, 4)` array."
  },
  {
    "q": "What is the broadcast shape of `(6, 1, 4)` and `(5, 1)`?",
    "options": ["`(6, 5, 4)`", "Error", "`(6, 5, 1)`", "`(5, 6, 4)`"],
    "answer": 0,
    "why": "Right-align and pad: `(6, 1, 4)` with `(1, 5, 1)`. Each pair is equal or contains a 1, and the result takes the larger length on each axis."
  },
  {
    "q": "Which pair of shapes fails to broadcast?",
    "options": ["`(8, 3)` and `(3,)`", "`(8, 3)` and `(8,)`", "`(8, 3)` and `(8, 1)`", "`(8, 3)` and `(1, 3)`"],
    "answer": 1,
    "why": "`(8,)` aligns with the last axis, length 3. 8 and 3 differ and neither is 1. Reshaping it to `(8, 1)` would work."
  },
  {
    "q": "A model returns predictions with shape `(100, 1)` and the labels have shape `(100,)`. What does `((pred - y) ** 2).mean()` compute?",
    "options": [
      "The correct mean squared error",
      "A `ValueError` is raised",
      "The mean of a `(100, 100)` array of every prediction minus every label",
      "The mean of a `(100, 1)` array"
    ],
    "answer": 2,
    "why": "`(100, 1)` and `(100,)` broadcast to `(100, 100)`. The result is a valid-looking number that is not the MSE."
  },
  {
    "q": "Which loop can be fully replaced by array operations?",
    "options": [
      "Applying a ReLU to each element of an activation array",
      "Updating weights over 500 gradient descent steps",
      "Generating 50 tokens one after another from a language model",
      "Running a recurrent network along a sequence"
    ],
    "answer": 0,
    "why": "Each ReLU output is independent of the others. The other loops have steps that depend on the result of the previous step."
  },
  {
    "q": "For two `(3, 3)` matrices `A` and `B`, which statement is true?",
    "options": [
      "`A * B` and `A @ B` give the same result",
      "`A * B` raises an error because it needs `@`",
      "`A @ B` is elementwise and `A * B` is the matrix product",
      "`A * B` is elementwise and `A @ B` is the matrix product"
    ],
    "answer": 3,
    "why": "`*` is always elementwise in NumPy; `@` is matrix multiplication."
  },
  {
    "q": "`Q` has shape `(8, 12, 64)` and `K` has shape `(8, 12, 64)`. What is the shape of `Q @ K.transpose(0, 2, 1)`?",
    "options": ["`(8, 64, 64)`", "`(8, 12, 12)`", "`(12, 12)`", "Error"],
    "answer": 1,
    "why": "The batch axis 8 is kept and each `(12, 64) @ (64, 12)` product gives `(12, 12)`: a score for every pair of tokens."
  },
  {
    "q": "Why is `np.linalg.solve(A, b)` preferred over `np.linalg.inv(A) @ b`?",
    "options": [
      "`inv` only works on 2 × 2 matrices",
      "`solve` works on singular matrices",
      "`solve` is faster and more numerically accurate",
      "`inv` returns a view, not a copy"
    ],
    "answer": 2,
    "why": "Solving factorises the matrix once and avoids the extra work and rounding error of forming the full inverse."
  },
  {
    "q": "With `z = np.array([1000.0, 1000.0])` and `e = np.exp(z - z.max())`, what is `e / e.sum()`?",
    "options": ["`[nan nan]`", "`[inf inf]`", "`[1. 0.]`", "`[0.5 0.5]`"],
    "answer": 3,
    "why": "Subtracting the maximum gives `[0, 0]`, whose exponentials are `[1, 1]`. Equal logits give equal probabilities, with no overflow."
  },
  {
    "q": "You need to shuffle features `X` and labels `y` for a new epoch. Which code is correct?",
    "options": [
      "`perm = rng.permutation(len(X)); X, y = X[perm], y[perm]`",
      "`rng.shuffle(X); rng.shuffle(y)`",
      "`np.random.seed(0); np.random.shuffle(X); np.random.shuffle(y)`",
      "`X = rng.permutation(X); y = rng.permutation(y)`"
    ],
    "answer": 0,
    "why": "One permutation applied to both arrays keeps every example paired with its label. The other options shuffle the two arrays in different orders."
  }
]
```
