This quiz covers the whole TensorFlow and Keras module: tensors, gradient tapes and `tf.function`, building models with the Sequential and Functional APIs, compiling and training, callbacks, evaluation and deployment. Aim for at least 12 out of 16 before moving on to Computer Vision.

```quiz
[
  {
    "q": "Which TensorFlow object holds a value that an optimiser can update in place?",
    "options": ["`tf.constant`", "`tf.Variable`", "`tf.Tensor`", "`tf.data.Dataset`"],
    "answer": 1,
    "why": "Tensors are immutable; trainable state such as layer weights lives in `tf.Variable`s."
  },
  {
    "q": "What does this print?\n\n`x = tf.Variable(4.0)`\n`with tf.GradientTape() as tape: y = x * x + 2 * x`\n`print(tape.gradient(y, x).numpy())`",
    "options": ["`24.0`", "`8.0`", "`10.0`", "`18.0`"],
    "answer": 2,
    "why": "The derivative of x² + 2x is 2x + 2, which is 10 at x = 4."
  },
  {
    "q": "`tf.constant([1, 2]) + tf.constant([0.5, 0.5])` raises an error. Why?",
    "options": [
      "The shapes do not match",
      "TensorFlow does not automatically promote `int32` to `float32` in mixed operations",
      "Constants cannot be added",
      "Addition requires `tf.add`"
    ],
    "answer": 1,
    "why": "TensorFlow is strict about dtypes; cast explicitly with `tf.cast`."
  },
  {
    "q": "Inside a `@tf.function`, when does a plain Python `print` run?",
    "options": [
      "On every call",
      "Only when the function is traced, for example on the first call or when input shapes change",
      "Never",
      "Only in eager mode after the first call"
    ],
    "answer": 1,
    "why": "Python code executes during tracing; the cached graph does not contain it. Use `tf.print` for per-call output."
  },
  {
    "q": "In a `tf.data` pipeline, what does `.prefetch(tf.data.AUTOTUNE)` do?",
    "options": [
      "Loads the entire dataset into GPU memory",
      "Prepares upcoming batches while the model trains on the current one",
      "Shuffles the data",
      "Caches results to disk"
    ],
    "answer": 1,
    "why": "Prefetching overlaps data preparation with computation, so the accelerator waits less."
  },
  {
    "q": "How do you select the PyTorch backend for Keras 3?",
    "options": [
      "`import keras.torch`",
      "Set `KERAS_BACKEND=torch` before importing Keras",
      "`model.compile(backend=\"torch\")`",
      "Keras cannot run on PyTorch"
    ],
    "answer": 1,
    "why": "Keras 3 reads the backend from the `KERAS_BACKEND` environment variable (or its config file) at import time."
  },
  {
    "q": "How many parameters does this model have?\n\n`keras.Sequential([keras.Input(shape=(8,)), layers.Dense(4), layers.Dense(2)])`",
    "options": ["42", "46", "36", "48"],
    "answer": 1,
    "why": "8 × 4 + 4 = 36 for the first layer and 4 × 2 + 2 = 10 for the second: 46."
  },
  {
    "q": "Which requirement forces you to use the Functional API rather than Sequential?",
    "options": [
      "Using dropout",
      "Using more than three layers",
      "A model that takes an image and a table of metadata as two separate inputs",
      "Using a softmax output"
    ],
    "answer": 2,
    "why": "Sequential models have exactly one input and one output and no branches."
  },
  {
    "q": "What is `keras.Input(shape=(4,))` in a functional model?",
    "options": [
      "A layer with 4 trainable weights",
      "A symbolic tensor that declares the per-example shape; no data and no weights",
      "A batch of 4 examples",
      "A normalisation layer"
    ],
    "answer": 1,
    "why": "Calling layers on it records the graph; `keras.Model(inputs, outputs)` then traces that graph."
  },
  {
    "q": "The last layer is `Dense(5)` with no activation and the labels are integers 0 to 4. Which loss is correct?",
    "options": [
      "`\"categorical_crossentropy\"`",
      "`keras.losses.SparseCategoricalCrossentropy(from_logits=True)`",
      "`\"sparse_categorical_crossentropy\"` with the default settings",
      "`\"binary_crossentropy\"`"
    ],
    "answer": 1,
    "why": "Integer labels need the sparse loss, and raw logit outputs need `from_logits=True`."
  },
  {
    "q": "In a custom Keras training loop, which call applies the computed gradients?",
    "options": [
      "`loss.backward()`",
      "`optimizer.step()`",
      "`optimizer.apply_gradients(zip(grads, model.trainable_variables))`",
      "`model.update(grads)`"
    ],
    "answer": 2,
    "why": "Gradients come from `tape.gradient`; the optimiser applies them with `apply_gradients`."
  },
  {
    "q": "Why should you call `model(x, training=True)` in a hand-written training step?",
    "options": [
      "To enable gradient computation",
      "So that layers like Dropout and BatchNormalization use their training behaviour",
      "To make the weights trainable",
      "It is required by `tf.function`"
    ],
    "answer": 1,
    "why": "The `training` argument switches layer behaviour, as `model.train()` does in PyTorch; calling the model directly defaults to inference behaviour."
  },
  {
    "q": "Which callback configuration stops training when validation loss stalls and keeps the best weights in the model?",
    "options": [
      "`EarlyStopping(monitor=\"loss\", patience=5)`",
      "`EarlyStopping(monitor=\"val_loss\", patience=5, restore_best_weights=True)`",
      "`ReduceLROnPlateau(monitor=\"val_loss\")`",
      "`ModelCheckpoint(\"m.keras\")`"
    ],
    "answer": 1,
    "why": "Monitoring `val_loss` detects overfitting, and `restore_best_weights=True` rolls back to the best epoch."
  },
  {
    "q": "Why does Keras compute precision over a whole evaluation set using stateful metrics instead of averaging per-batch precision?",
    "options": [
      "It is faster",
      "The average of per-batch precisions is generally not equal to the precision over all examples",
      "Batches must be the same size",
      "Per-batch precision cannot be computed"
    ],
    "answer": 1,
    "why": "Metrics accumulate true and false positive counts across batches, so the final value is exact for the whole set."
  },
  {
    "q": "Which artefact would you give to TensorFlow Serving?",
    "options": [
      "A `.keras` file",
      "A `.weights.h5` file",
      "A SavedModel directory written by `model.export`, inside a numbered version folder",
      "A `.tflite` file"
    ],
    "answer": 2,
    "why": "TensorFlow Serving loads SavedModels and serves the highest numbered version in the model directory by default."
  },
  {
    "q": "You convert a model to TensorFlow Lite with `Optimize.DEFAULT`. What should you do before shipping it?",
    "options": [
      "Nothing; conversion never changes predictions",
      "Retrain it from scratch",
      "Compare its predictions and accuracy with the original model on validation data",
      "Convert it to TensorFlow.js as well"
    ],
    "answer": 2,
    "why": "Quantisation usually costs little accuracy, but you must measure it rather than assume it."
  }
]
```
