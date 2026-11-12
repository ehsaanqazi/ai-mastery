This quiz covers the whole Sequence Models module: recurrent networks and backpropagation through time, LSTM and GRU cells and their PyTorch shapes, sequence modelling patterns, time-series forecasting practice and attention. Aim for at least 13 out of 16 before moving on to NLP Fundamentals.

```quiz
[
  {
    "q": "In the RNN update `h_t = tanh(W_xh x_t + W_hh h_t-1 + b_h)`, why does the number of parameters not depend on sequence length?",
    "options": [
      "The sequence is truncated to a fixed length",
      "The same weights are reused at every time step",
      "Only the last step has trainable weights",
      "tanh has no parameters"
    ],
    "answer": 1,
    "why": "Weight sharing across time is the defining feature of a recurrent network."
  },
  {
    "q": "During backpropagation through time, the gradient from step T to step k involves a product of how many Jacobians `dh_t/dh_t-1`?",
    "options": [
      "One",
      "Two",
      "T - k",
      "T times k"
    ],
    "answer": 2,
    "why": "The chain rule passes through every intermediate step, giving a product of T - k Jacobians. This repeated product causes vanishing and exploding gradients."
  },
  {
    "q": "Your RNN's loss is stable, then suddenly becomes `nan`. Which change addresses the most likely cause directly?",
    "options": [
      "Add `torch.nn.utils.clip_grad_norm_` between `backward()` and `step()`",
      "Increase the hidden size",
      "Remove the tanh activation",
      "Shuffle the time steps within each sequence"
    ],
    "answer": 0,
    "why": "A sudden blow-up is the signature of exploding gradients. Clipping the gradient norm caps the size of each update."
  },
  {
    "q": "What does truncated backpropagation through time do?",
    "options": [
      "Drops the second half of every sequence",
      "Stops gradients at chunk boundaries while still carrying the hidden state forward",
      "Resets the hidden state to zero at every step",
      "Clips gradient values to the range -1 to 1"
    ],
    "answer": 1,
    "why": "Calling `h.detach()` between chunks bounds memory and compute; the state still flows forward, but gradients do not flow back past the boundary."
  },
  {
    "q": "Which LSTM gate decides how much of the previous cell state is kept?",
    "options": [
      "Input gate",
      "Output gate",
      "Candidate",
      "Forget gate"
    ],
    "answer": 3,
    "why": "c_t = f_t * c_t-1 + i_t * g_t, so the forget gate scales the old cell state."
  },
  {
    "q": "How many parameters does a single-layer PyTorch `nn.LSTM(input_size=8, hidden_size=16)` have?",
    "options": [
      "1,664",
      "416",
      "1,536",
      "1,248"
    ],
    "answer": 0,
    "why": "4 x (16 x 8 + 16 x 16 + 2 x 16) = 4 x (128 + 256 + 32) = 1,664."
  },
  {
    "q": "For `nn.LSTM(10, 32, num_layers=3, batch_first=True)` and input of shape `(16, 50, 10)`, what is the shape of `h_n`?",
    "options": [
      "`(16, 3, 32)`",
      "`(16, 50, 32)`",
      "`(3, 16, 32)`",
      "`(16, 32)`"
    ],
    "answer": 2,
    "why": "`h_n` is `(num_layers * num_directions, batch, hidden)` and is not affected by `batch_first`."
  },
  {
    "q": "Read this code: `output, (h_n, c_n) = lstm(x)` with `batch_first=True`, then `features = output[-1]`. What does `features` contain?",
    "options": [
      "The last time step of every sequence",
      "All time steps of the last sequence in the batch",
      "The final cell state",
      "The first layer's hidden state"
    ],
    "answer": 1,
    "why": "With batch_first, dim 0 is the batch. The intended expression is `output[:, -1, :]` or `h_n[-1]`."
  },
  {
    "q": "In PyTorch's GRU, `h_t = (1 - z_t) * n_t + z_t * h_t-1`. What does the reset gate `r_t` affect?",
    "options": [
      "How much the previous state contributes to the candidate `n_t`",
      "Whether the output is exposed",
      "The learning rate",
      "The final interpolation between old state and candidate"
    ],
    "answer": 0,
    "why": "The reset gate multiplies the recurrent term inside the candidate computation. The interpolation is controlled by the update gate z."
  },
  {
    "q": "You replace `nn.LSTM` with a 2-layer `nn.GRU` but keep `output, (h_n, c_n) = rnn(x)`. What happens?",
    "options": [
      "A shape error is raised immediately",
      "PyTorch creates a zero cell state",
      "It runs silently, with `h_n` and `c_n` holding layer 1 and layer 2 final states",
      "The GRU is converted to an LSTM"
    ],
    "answer": 2,
    "why": "The GRU returns one tensor of shape (2, batch, hidden), which Python unpacks along its first dimension."
  },
  {
    "q": "Which task is aligned many-to-many?",
    "options": [
      "Translating English to German",
      "Classifying a review as positive or negative",
      "Generating a caption for an image",
      "Tagging each word in a sentence with its part of speech"
    ],
    "answer": 3,
    "why": "Part-of-speech tagging produces exactly one output per input token. Translation has different input and output lengths, so it needs seq2seq."
  },
  {
    "q": "Why use `pack_padded_sequence` before an RNN on a padded batch?",
    "options": [
      "So the RNN skips padding steps and `h_n` is each sequence's state at its true last token",
      "So all sequences are truncated to the shortest length",
      "To convert token ids into embeddings",
      "To make the batch fit on a GPU"
    ],
    "answer": 0,
    "why": "Without packing, short sequences keep being updated with padding inputs, and their final states are corrupted."
  },
  {
    "q": "What is teacher forcing?",
    "options": [
      "Training a small model to imitate a large one",
      "Feeding the true previous target token into the decoder during training instead of its own prediction",
      "Freezing the encoder while training the decoder",
      "Clipping the decoder's gradients"
    ],
    "answer": 1,
    "why": "It stabilises and speeds up training but creates exposure bias, because at inference the decoder sees its own outputs."
  },
  {
    "q": "A forecasting model scores MAE 4.1 on the test period. Seasonal naive scores 3.6 on the same period. What should you conclude?",
    "options": [
      "The model is good because MAE is low",
      "The test set is too small to judge",
      "The model is worse than a trivial baseline and should not ship as is",
      "MAE is the wrong metric for forecasting"
    ],
    "answer": 2,
    "why": "A model must beat simple baselines on the same data. Here, repeating last week's value is more accurate."
  },
  {
    "q": "Which preprocessing step leaks future information into a forecasting model?",
    "options": [
      "Fitting a `StandardScaler` on the full series before splitting into train and test",
      "Computing a lag-7 feature with `series.shift(7)`",
      "Using `TimeSeriesSplit` for validation",
      "Computing a rolling mean with `series.shift(1).rolling(7).mean()`"
    ],
    "answer": 0,
    "why": "The scaler's mean and variance would include test-period values. Fit preprocessing on the training period only."
  },
  {
    "q": "Compute scaled dot-product attention weights for one query `q = [2, 0]` and keys `k1 = [1, 0]`, `k2 = [0, 1]` (so `d_k = 2`). Which is closest?",
    "options": [
      "`[0.5, 0.5]`",
      "`[1.0, 0.0]`",
      "`[0.88, 0.12]`",
      "`[0.80, 0.20]`"
    ],
    "answer": 3,
    "why": "Scores are 2/sqrt(2) = 1.414 and 0. softmax gives exp(1.414) / (exp(1.414) + 1) = 4.11 / 5.11 = 0.80 for k1 and 0.20 for k2."
  }
]
```
