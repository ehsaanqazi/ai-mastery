The [perceptrons](/learn/neural-network-fundamentals/perceptrons) lesson ended with a network of three neurons in two layers solving a problem no single neuron could. A network's **architecture** is the design decision behind that: how many neurons, how they are grouped into layers, how the layers connect, and which activation each uses. Architecture fixes *what functions the network can possibly represent* and how many parameters it must learn; training then picks one function from that family. This lesson covers the vocabulary, the parameter arithmetic, why depth matters, and how to initialise the weights so that a deep network can train at all.

## The multilayer perceptron

The simplest and most general architecture is the **multilayer perceptron (MLP)**, also called a **fully connected** or **feed-forward** network. Neurons are arranged in layers; every neuron in one layer feeds every neuron in the next; information flows strictly forwards with no loops.

```mermaid
flowchart LR
    I["Input layer<br/>4 features"] -- "W1: 4×8" --> H1["Hidden layer 1<br/>8 units, ReLU"]
    H1 -- "W2: 8×8" --> H2["Hidden layer 2<br/>8 units, ReLU"]
    H2 -- "W3: 8×3" --> O["Output layer<br/>3 units, softmax"]
```

The vocabulary you will see everywhere:

| Term | Meaning |
| --- | --- |
| Layer | A group of neurons that all receive the same inputs and are computed together |
| Dense / fully connected / linear layer | A layer where every input connects to every output: $\mathbf{h} = f(W\mathbf{x} + \mathbf{b})$ |
| Width | Number of neurons in a layer |
| Depth | Number of layers with weights. The network above has depth 3: the input layer holds data, not parameters, so it is not counted |
| Hidden layer | Any layer between input and output; its values are not observed in the data |
| Feed-forward | No cycles: each layer depends only on earlier layers |
| Parameters | All the weights and biases, the numbers training adjusts |
| Hyperparameters | Choices you make before training: depth, widths, activations, learning rate |

"Deep" learning simply means networks with many layers. There is no official threshold, but the term usually implies more than two or three.

## Counting parameters

A dense layer from $m$ inputs to $n$ outputs has an $m \times n$ weight matrix and $n$ biases, so $mn + n$ parameters. A network's size is the sum over its layers:

```python
import numpy as np

def count_params(sizes):
    """sizes = [inputs, hidden1, ..., outputs]"""
    return sum(m * n + n for m, n in zip(sizes[:-1], sizes[1:]))

print("784-128-64-10:      ", count_params([784, 128, 64, 10]))
print("784-512-10:         ", count_params([784, 512, 10]))
print("784-64-64-64-64-10: ", count_params([784, 64, 64, 64, 64, 10]))
```

```output
784-128-64-10:       109386
784-512-10:          407050
784-64-64-64-64-10:  63370
```

These are typical shapes for classifying 28 by 28 pixel digit images (784 inputs, 10 classes). Two things stand out. The first layer usually dominates, because the input is wide. And a deeper, narrower network can have far fewer parameters than a shallow, wide one. Parameter count drives memory use and compute cost, so it is the first number to check when you design or read about a model.

## What can a network represent?

The **universal approximation theorem** (Cybenko 1989 for sigmoid units, extended by Hornik and others) states that a network with a *single* hidden layer and a suitable nonlinear activation can approximate any continuous function on a bounded region as closely as you like, given enough hidden units.

It is an important result, and an easily misread one. It says such a network *exists*. It does not say:

- how many units you need (for some functions, exponentially many),
- that gradient descent will find the right weights,
- that the network will generalise from limited data.

So why go deep instead of wide? Because depth lets a network reuse computation. Each layer builds features out of the previous layer's features, and composing simple functions can create complicated ones very cheaply.

## Depth in action: folding with ReLUs

Here is a concrete demonstration. The ReLU activation is $\text{relu}(z) = \max(0, z)$. Two ReLU units make a "hat" function on $[0, 1]$: it rises from 0 to 1 and falls back to 0.

$$
\text{hat}(x) = 2\,\text{relu}(x) - 4\,\text{relu}(x - 0.5)
$$

Each copy of the hat is a tiny layer of two neurons. Feed the hat's output into another hat and the input range is folded twice. Count the straight-line pieces of the resulting function after each extra layer:

```python
relu = lambda z: np.maximum(0, z)
hat = lambda x: 2 * relu(x) - 4 * relu(x - 0.5)

x = np.linspace(0, 1, 100_001)
y = x.copy()
for layer in range(1, 6):
    y = hat(y)
    slopes = np.round(np.diff(y) / np.diff(x), 3)
    pieces = 1 + np.count_nonzero(np.diff(slopes))
    print(f"{layer} layer(s), {2 * layer} ReLUs: {pieces} linear pieces")
```

```output
1 layer(s), 2 ReLUs: 2 linear pieces
2 layer(s), 4 ReLUs: 4 linear pieces
3 layer(s), 6 ReLUs: 8 linear pieces
4 layer(s), 8 ReLUs: 16 linear pieces
5 layer(s), 10 ReLUs: 32 linear pieces
```

Each layer of just two neurons doubles the number of pieces, so $k$ layers produce a zig-zag with $2^k$ pieces using only $2k$ ReLUs. A single hidden layer of ReLUs can only create one kink per unit, so it would need roughly $2^k$ units to draw the same shape. This exponential gap is one precise sense in which depth is more efficient than width.

> [!KEY]
> Width adds capacity linearly; depth composes functions and can add expressive power exponentially. That is why practical networks are deep rather than one enormous hidden layer, provided you can train them, which is where initialisation and normalisation come in.

## Initialising the weights

Before training starts, every weight needs a value. Two rules matter.

**Break the symmetry.** If every weight in a layer starts at the same value, every neuron in that layer computes the same output, receives the same gradient and stays identical forever. A layer of 100 identical neurons is really one neuron. Random initial weights make each unit different. (Biases can safely start at zero.)

```python
X = np.array([[0.5, -1.0, 2.0]])
W_same = np.full((3, 4), 0.1)          # every weight equal
print(relu(X @ W_same))                 # all four hidden units identical
```

```output
[[0.15 0.15 0.15 0.15]]
```

**Keep the signal scale steady.** Each layer multiplies the signal by a random matrix. If the weights are too small, activations shrink layer after layer towards zero; too large, and they explode. Gradients, which flow back through the same matrices, suffer the same fate. The standard fix is to scale the random weights by the layer's **fan-in** (number of inputs, $n_{\text{in}}$) and fan-out ($n_{\text{out}}$):

| Scheme | Weight variance | Designed for |
| --- | --- | --- |
| Xavier / Glorot (2010) | $\dfrac{2}{n_{\text{in}} + n_{\text{out}}}$ | tanh and sigmoid |
| He / Kaiming (2015) | $\dfrac{2}{n_{\text{in}}}$ | ReLU and its variants |

He initialisation has the extra factor of 2 because ReLU zeroes about half its inputs, halving the signal's second moment at each layer. Watch the standard deviation of the activations through ten ReLU layers of width 256 under each scheme:

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 256))   # a batch of 1000 standardised inputs

schemes = {"small (0.01)": 0.01,
           "Xavier": np.sqrt(2 / (256 + 256)),
           "He": np.sqrt(2 / 256)}

for name, std in schemes.items():
    h, stds = X, []
    for layer in range(10):
        W = rng.standard_normal((256, 256)) * std
        h = relu(h @ W)
        stds.append(h.std())
    print(f"{name:13s} layer 1: {stds[0]:.2e}  layer 5: {stds[4]:.2e}  layer 10: {stds[9]:.2e}")
```

```output
small (0.01)  layer 1: 9.40e-02  layer 5: 1.59e-05  layer 10: 3.04e-10
Xavier        layer 1: 5.87e-01  layer 5: 1.47e-01  layer 10: 2.41e-02
He            layer 1: 8.26e-01  layer 5: 9.92e-01  layer 10: 9.85e-01
```

With small weights the signal is billions of times smaller by layer 10: effectively zero, and the gradients will be too. Xavier, designed for activations symmetric around zero, still shrinks the standard deviation by a factor of about $\sqrt{2}$ per layer with ReLU. He initialisation keeps the scale near 1 all the way down.

> [!TIP]
> Frameworks pick sensible defaults, but they differ between layer types and libraries. When a deep network refuses to train, checking the activation statistics per layer, exactly as above, is one of the first diagnostics to run.

## Beyond fully connected layers

The MLP makes no assumptions about its input: every feature talks to every neuron. Specialised architectures build in assumptions, called **inductive biases**, that match the structure of the data. That saves parameters and helps generalisation.

| Architecture | Built-in assumption | Typical data | Where you meet it |
| --- | --- | --- | --- |
| MLP | None; any feature can interact with any other | Tabular features, small fixed-size inputs, the final layers of larger models | This module |
| Convolutional network (CNN) | Nearby pixels matter together; the same pattern can appear anywhere | Images, audio spectrograms | [CNNs](/learn/computer-vision/convolutional-neural-networks-cnns) |
| Recurrent network (RNN, LSTM) | Data arrives in order; a hidden state summarises the past | Sequences, time series | [RNNs](/learn/sequence-models/recurrent-neural-networks) |
| Transformer | Every position can attend to every other, with learned weighting | Text, and increasingly images and audio | [Transformer architecture](/learn/generative-ai-fundamentals/transformer-architecture) |

All of them are still built from the same ingredients as this module: linear maps, nonlinear activations, a loss, backpropagation and an optimiser. Even a transformer contains an MLP block in every layer.

## Choosing a starting architecture

For a new problem with an MLP:

1. Match the **input layer** to your feature count and the **output layer** to the task, as described in [input, hidden and output layers](/learn/neural-network-fundamentals/input-hidden-and-output-layers).
2. Start with two hidden layers of moderate width (64 to 256 units) and ReLU activations.
3. Check you can overfit a small subset. If not, there is a bug or the model is too small.
4. Then scale width and depth up while validation performance improves, and add [regularisation](/learn/neural-network-fundamentals/regularization) when it starts to overfit.

## Exercises

**1.** How many parameters does a network with layer sizes `[20, 50, 50, 1]` have? Compute it by hand and check with `count_params`.

<details>
<summary>Show solution</summary>

$20 \cdot 50 + 50 = 1050$, $50 \cdot 50 + 50 = 2550$, $50 \cdot 1 + 1 = 51$. Total $3651$.

```python
print(count_params([20, 50, 50, 1]))   # 3651
```

</details>

**2.** In the initialisation experiment, add a fourth scheme with standard deviation `0.1` and predict, before running it, whether the signal will grow or shrink. Explain using the He formula.

<details>
<summary>Show solution</summary>

He's standard deviation for fan-in 256 is $\sqrt{2/256} \approx 0.088$. A standard deviation of 0.1 is slightly larger, so each layer multiplies the scale by about $0.1 / 0.088 \approx 1.13$, and the signal grows by roughly $1.13^{10} \approx 3.4$ over ten layers. Over a hundred layers that would be an explosion.

</details>

**3.** Why is the input layer usually not counted when stating a network's depth?

<details>
<summary>Show solution</summary>

The input layer has no weights, biases or activation; it simply holds the feature vector. Depth counts the layers that transform data, each of which has parameters.

</details>

```quiz
[
  {
    "q": "A dense layer maps 300 inputs to 100 outputs. How many parameters does it have?",
    "options": ["30,000", "30,100", "30,300", "400"],
    "answer": 1,
    "why": "The weight matrix has 300 × 100 = 30,000 entries, plus one bias per output: 30,100."
  },
  {
    "q": "What does the universal approximation theorem NOT guarantee?",
    "options": [
      "That a one-hidden-layer network with enough units can approximate a continuous function on a bounded region",
      "That suitable weights exist",
      "That gradient descent will find good weights with a reasonable number of units",
      "That the approximation can be made arbitrarily accurate"
    ],
    "answer": 2,
    "why": "The theorem is an existence result. It says nothing about the number of units needed or whether training will find the weights."
  },
  {
    "q": "Why must the weights not all be initialised to the same value?",
    "options": [
      "It causes a division by zero",
      "Every neuron in a layer would compute the same thing and receive the same update, so they would never differ",
      "The loss would be undefined",
      "It makes training too fast"
    ],
    "answer": 1,
    "why": "Identical weights mean identical outputs and identical gradients, so the symmetry is never broken and the layer acts like a single neuron."
  },
  {
    "q": "Which initialisation is designed for ReLU networks?",
    "options": ["Xavier/Glorot", "All zeros", "Standard deviation 1", "He/Kaiming, variance 2/fan-in"],
    "answer": 3,
    "why": "He initialisation doubles the variance to compensate for ReLU zeroing roughly half of its inputs, keeping activation scale stable through depth."
  }
]
```
