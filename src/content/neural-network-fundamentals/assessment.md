This quiz covers the whole Neural Network Fundamentals module: neurons and perceptrons, architecture and layers, activation functions, forward and backward propagation, losses and optimisers, learning rates, batch size and epochs, and regularisation with dropout and batch normalisation. Answer every question to see your score. Your best result is saved. Aim for at least 16 out of 20 before moving on to PyTorch.

```quiz
[
  {
    "q": "A neuron has weights `(1, -2)`, bias `0.5` and a ReLU activation. What does it output for the input `(3, 1)`?",
    "options": ["0", "1.5", "2.5", "3.5"],
    "answer": 1,
    "why": "$z = 1 \\cdot 3 + (-2) \\cdot 1 + 0.5 = 1.5$, and ReLU leaves positive values unchanged."
  },
  {
    "q": "Why can a single perceptron never learn XOR, however long it trains?",
    "options": [
      "The step function has no derivative",
      "XOR needs a learning rate below 1",
      "XOR has too few training examples",
      "No single straight line separates the inputs labelled 1 from those labelled 0"
    ],
    "answer": 3,
    "why": "A perceptron's decision boundary is one hyperplane. XOR's positive points sit on opposite corners of the square, so it is not linearly separable."
  },
  {
    "q": "A perceptron predicts 1 for an example whose label is 0. With learning rate 1, what is the update?",
    "options": ["`w = w - x`, `b = b - 1`", "`w = w + x`, `b = b + 1`", "`w = w - x`, `b` unchanged", "No update is made"],
    "answer": 0,
    "why": "The factor $y - \\hat{y} = 0 - 1 = -1$, so the input is subtracted from the weights and 1 from the bias, lowering the pre-activation for that input."
  },
  {
    "q": "How many learnable parameters does an MLP with layer sizes `[10, 32, 1]` have?",
    "options": ["352", "385", "353", "320"],
    "answer": 1,
    "why": "First layer: 10 × 32 + 32 = 352. Output layer: 32 × 1 + 1 = 33. Total 385."
  },
  {
    "q": "What is a network of five dense layers with no activation functions equivalent to?",
    "options": [
      "A five-times more expressive model",
      "A network that cannot be trained",
      "A single linear layer",
      "A decision tree of depth five"
    ],
    "answer": 2,
    "why": "Composing linear maps gives a linear map, so the whole stack collapses to $xW' + b'$."
  },
  {
    "q": "In a deep MLP, which hidden-layer activation is least prone to vanishing gradients caused by the activation itself?",
    "options": ["Sigmoid", "Tanh", "ReLU", "Softmax"],
    "answer": 2,
    "why": "ReLU's derivative is exactly 1 for every active unit, while sigmoid's is at most 0.25 and tanh's falls below 1 everywhere except at zero."
  },
  {
    "q": "What weight initialisation variance is designed for a ReLU layer with fan-in $n$?",
    "options": ["$2/n$ (He)", "$1$", "$0$", "$2/(n + n_{\\text{out}})$ only, for every activation"],
    "answer": 0,
    "why": "He initialisation uses variance $2/n$ to compensate for ReLU zeroing about half its inputs, keeping activation scale stable through depth."
  },
  {
    "q": "A batch `X` has shape `(64, 20)` and the first layer's weights `W1` have shape `(20, 50)`. What is the shape of `X @ W1 + b1`?",
    "options": ["`(20, 50)`", "`(64, 20)`", "`(50, 64)`", "`(64, 50)`"],
    "answer": 3,
    "why": "Matrix multiplication of `(64, 20)` by `(20, 50)` gives `(64, 50)`, and the bias of shape `(50,)` broadcasts across the 64 rows."
  },
  {
    "q": "In a backward pass, a layer's input `A_in` has shape `(64, 50)` and its error signal `delta` has shape `(64, 10)`. What does `A_in.T @ delta` compute?",
    "options": [
      "The gradient for the layer's bias, shape `(10,)`",
      "The gradient for the layer's weights, shape `(50, 10)`, summed over the batch",
      "The error signal for the previous layer, shape `(64, 50)`",
      "The layer's output, shape `(64, 10)`"
    ],
    "answer": 1,
    "why": "The weight gradient is the layer input transposed times the error signal; the matrix product sums the per-example contributions."
  },
  {
    "q": "For a sigmoid output unit trained with binary cross-entropy, what is the gradient of the loss with respect to the output's pre-activation?",
    "options": ["$a - y$", "$a(1 - a)$", "$(a - y)\\,a(1 - a)$", "$-y / a$"],
    "answer": 0,
    "why": "The sigmoid's derivative cancels the denominator of the cross-entropy derivative, leaving prediction minus target."
  },
  {
    "q": "A gradient check on your hand-written backward pass reports relative errors of about `1e-9` for every layer. What should you conclude?",
    "options": [
      "There is a bug in the output layer",
      "The step size `h` is too large",
      "The analytic gradients are correct",
      "The network is overfitting"
    ],
    "answer": 2,
    "why": "In float64, correct gradients typically agree with central differences to 1e-7 or better. Errors around 1e-3 or larger indicate a bug."
  },
  {
    "q": "A freshly initialised 10-class classifier logs a first-batch cross-entropy of about 2.3. What does that tell you?",
    "options": [
      "The model is already trained",
      "This is the expected loss for near-uniform predictions, since ln(10) ≈ 2.3",
      "The labels must be wrong",
      "The learning rate is too high"
    ],
    "answer": 1,
    "why": "Uniform predictions over k classes give a cross-entropy of ln k, a useful sanity check at the start of training."
  },
  {
    "q": "A model ends with a softmax layer, and its probabilities are passed to a loss function that expects raw logits. What happens?",
    "options": [
      "An error is raised immediately",
      "Training is unaffected",
      "The loss becomes negative",
      "Softmax is effectively applied twice, flattening the outputs and weakening the gradients, without any error"
    ],
    "answer": 3,
    "why": "Losses such as PyTorch's `CrossEntropyLoss` apply log-softmax internally. Feeding them probabilities is a silent bug: training runs but learns poorly."
  },
  {
    "q": "Why is AdamW preferred to adding an L2 penalty to the loss when training with Adam?",
    "options": [
      "AdamW decouples weight decay from the adaptive scaling, so every weight is decayed as intended",
      "AdamW needs no learning rate",
      "L2 penalties cannot be differentiated",
      "AdamW stores fewer moment estimates"
    ],
    "answer": 0,
    "why": "Inside Adam, an L2 gradient term is divided by each parameter's running gradient scale, under-regularising weights with large gradients. AdamW applies the decay separately."
  },
  {
    "q": "In a learning-rate range test, the smoothed loss falls steadily, reaches its lowest point at lr = 0.5, then shoots up. Which learning rate is a sensible choice?",
    "options": ["About 5", "About 0.5", "About 0.05", "About 0.00005"],
    "answer": 2,
    "why": "At the minimum the loss is already close to diverging. A value roughly ten times smaller sits in the steadily decreasing region."
  },
  {
    "q": "A dataset has 1,000 training examples and the batch size is 64. How many iterations are in one epoch if the final partial batch is kept?",
    "options": ["15", "16", "64", "1,000"],
    "answer": 1,
    "why": "1,000 / 64 = 15.6, so there are 15 full batches and one partial batch: 16 iterations."
  },
  {
    "q": "You increase the batch size from 32 to 256 and keep everything else the same. Which of these is the most likely change?",
    "options": [
      "More optimiser updates per epoch",
      "Higher gradient noise per step",
      "Lower activation memory",
      "Fewer updates per epoch, so the learning rate usually needs to be raised to compensate"
    ],
    "answer": 3,
    "why": "Updates per epoch fall by a factor of 8 and each gradient is less noisy, so the usable learning rate rises. Activation memory grows with the batch."
  },
  {
    "q": "Training loss keeps falling, but validation loss reached its minimum at epoch 18 and has risen since. What should you do?",
    "options": [
      "Keep training until training loss reaches zero",
      "Raise the learning rate",
      "Use the epoch-18 checkpoint and add regularisation such as weight decay, augmentation or dropout",
      "Report the training loss as the final result"
    ],
    "answer": 2,
    "why": "Falling training loss with rising validation loss is overfitting. Early stopping keeps the best checkpoint, and regularisation can push the validation minimum lower."
  },
  {
    "q": "With inverted dropout at drop probability p = 0.25, a unit that is kept has activation 3.0. What value does it pass on during training, and what during evaluation?",
    "options": [
      "4.0 during training, 3.0 during evaluation",
      "3.0 during training, 2.25 during evaluation",
      "2.25 during training, 3.0 during evaluation",
      "4.0 in both modes"
    ],
    "answer": 0,
    "why": "Kept units are divided by 1 - p = 0.75 during training, giving 4.0. In evaluation dropout is the identity, so the activation stays 3.0."
  },
  {
    "q": "Which statement about batch normalisation and layer normalisation is correct?",
    "options": [
      "Both normalise each feature over the batch",
      "Batch norm uses running statistics in evaluation mode, while layer norm normalises each example over its own features and behaves the same in both modes",
      "Layer norm needs large batches to work well",
      "Batch norm gives the same output for an example regardless of its batch, even in training mode"
    ],
    "answer": 1,
    "why": "Batch norm's statistics come from the batch during training and from running averages at evaluation. Layer norm uses only the example itself, which is why transformers use it."
  }
]
```
