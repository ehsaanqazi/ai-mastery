The [loss function](/learn/essential-mathematics-for-machine-learning/loss-functions) tells you how wrong a model is. The **optimiser** decides how to change millions or billions of parameters to make it less wrong, one step at a time. You met plain [gradient descent](/learn/calculus/gradient-descent) in the calculus module. Real training adds three ideas on top of it: estimate the gradient from small random batches, remember past gradients to build up speed, and give each parameter its own step size. Put together, those ideas give you SGD, momentum, RMSProp and Adam, the optimisers behind nearly every model you will train. This lesson writes out their update rules, runs them side by side, and covers the learning-rate schedules that go with them.

## The training loop and stochastic gradients

Every optimiser in this lesson fits into the same loop. Only the "update" box changes.

```mermaid
flowchart LR
    A["Sample a mini-batch"] --> B["Forward pass:<br/>predictions and loss"]
    B --> C["Backward pass:<br/>gradient g"]
    C --> D["Optimiser update:<br/>θ ← θ − step"]
    D --> E["Learning-rate schedule:<br/>adjust η"]
    E --> A
```

Computing the exact gradient of the cost needs a pass over the whole dataset, which is far too slow when there are millions of examples. **Stochastic gradient descent (SGD)** instead estimates it from a random **mini-batch** $B$ of, say, 32 to a few thousand examples:

$$
g_t = \frac{1}{|B|}\sum_{i \in B} \nabla_\theta\, \ell\big(f_\theta(x_i), y_i\big)
\qquad
\theta_{t+1} = \theta_t - \eta\, g_t
$$

Here $\theta_t$ are the parameters at step $t$, $g_t$ is the mini-batch gradient and $\eta$ (eta) is the **learning rate**. Because the batch is random, $g_t$ is a noisy but unbiased estimate of the full gradient: on average it points the right way. The noise is not only a cost. It helps the optimiser leave sharp, narrow minima, and many cheap noisy steps beat a few expensive exact ones.

## The problem: ravines

Loss surfaces are rarely round bowls. Typically the loss is very steep in some directions and nearly flat in others, like a long narrow ravine. In the language of [second derivatives](/learn/calculus/optimization) from Module 4, the curvature in different directions can differ by orders of magnitude.

This causes a conflict for a single learning rate. To stay stable in the steep direction, $\eta$ must be smaller than $2/L$, where $L$ is the largest curvature. But that small $\eta$ makes progress along the flat direction painfully slow. The result is the classic zig-zag: bouncing between the ravine walls while crawling along the floor. Momentum and adaptive methods are two different fixes for this one problem.

## Momentum

**Momentum** keeps a running "velocity" $v_t$ that accumulates gradients:

$$
v_t = \beta\, v_{t-1} + g_t
\qquad
\theta_{t+1} = \theta_t - \eta\, v_t
$$

with $\beta$ typically 0.9. Think of a heavy ball rolling down the ravine. Gradient components that keep pointing the same way (along the floor) add up, so the ball speeds up. Components that flip sign each step (across the walls) cancel out. With $\beta = 0.9$, a steady gradient is effectively amplified up to $1/(1 - \beta) = 10$ times.

## RMSProp: a step size per parameter

**RMSProp** takes the other approach. It tracks a running average of each parameter's *squared* gradient and divides by its square root:

$$
s_t = \rho\, s_{t-1} + (1 - \rho)\, g_t^2
\qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t
$$

All operations are element-wise: each parameter has its own $s_t$. $\rho$ is a decay rate around 0.9 and $\epsilon$ is a tiny constant such as $10^{-8}$ to avoid dividing by zero. A parameter with consistently large gradients gets divided by a large number and takes smaller steps; one with tiny gradients gets boosted. The ravine is rescaled so that every direction looks roughly equally steep. The step for each parameter becomes roughly $\eta$ in size whatever the gradient's scale, which also makes the learning rate much easier to choose.

## Adam: momentum plus adaptive scaling

**Adam** (adaptive moment estimation) combines both ideas. It keeps a running mean of gradients (the first moment, like momentum) and a running mean of squared gradients (the second moment, like RMSProp):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

The defaults from the original paper are $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$. The lines with hats are **bias correction**. Because $m$ and $v$ start at zero, their early values are far too small; dividing by $1 - \beta^t$ fixes this, and the correction fades to nothing as $t$ grows. Step through the first three updates of Adam on $f(w) = w^2$ to see it.

```trace
{
  "title": "Three Adam steps on f(w) = w², lr = 0.1",
  "code": [
    "w, m, v = 1.0, 0.0, 0.0",
    "for t in range(1, 4):",
    "    g = 2 * w",
    "    m = 0.9 * m + 0.1 * g",
    "    v = 0.999 * v + 0.001 * g * g",
    "    m_hat = m / (1 - 0.9 ** t)",
    "    v_hat = v / (1 - 0.999 ** t)",
    "    w = w - 0.1 * m_hat / (v_hat ** 0.5 + 1e-8)",
    "    print(f\"t={t} w={w:.4f}\")"
  ],
  "steps": [
    { "line": 1, "vars": { "w": "1.0", "m": "0.0", "v": "0.0" }, "note": "Start at w = 1. Both moment estimates start at zero." },
    { "line": 3, "vars": { "t": "1", "w": "1.0", "g": "2.0", "m": "0.0", "v": "0.0" }, "note": "The gradient of w² is 2w = 2." },
    { "line": 5, "vars": { "t": "1", "w": "1.0", "g": "2.0", "m": "0.2", "v": "0.004" }, "note": "m is only 0.2 and v only 0.004: both badly underestimate the true gradient statistics because they started at zero." },
    { "line": 7, "vars": { "t": "1", "w": "1.0", "g": "2.0", "m_hat": "2.0", "v_hat": "4.0" }, "note": "Bias correction divides by 0.1 and 0.001, recovering exactly g = 2 and g² = 4." },
    { "line": 9, "vars": { "t": "1", "w": "0.9000" }, "out": ["t=1 w=0.9000"], "note": "Step = 0.1 × 2 / √4 = 0.1. Adam's first step is almost exactly the learning rate, whatever the gradient's size." },
    { "line": 9, "vars": { "t": "2", "w": "0.8004", "m_hat": "1.8947", "v_hat": "3.6198" }, "out": ["t=1 w=0.9000", "t=2 w=0.8004"], "note": "Again a step of about 0.1. The ratio m_hat / √v_hat stays close to 1 while the gradient keeps the same sign." },
    { "line": 9, "vars": { "t": "3", "w": "0.7016", "m_hat": "1.7863", "v_hat": "3.2671" }, "out": ["t=1 w=0.9000", "t=2 w=0.8004", "t=3 w=0.7016"], "note": "Plain gradient descent with lr 0.1 would shrink w by 20% per step. Adam instead moves at a roughly constant speed set by lr." }
  ]
}
```

### AdamW

Classic L2 regularisation adds $\lambda\theta$ to the gradient. Inside Adam, that term gets divided by $\sqrt{\hat{v}_t}$ like everything else, which weakens it for exactly the parameters with large gradients. **AdamW** decouples weight decay by applying it directly to the parameters:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_t\right)
$$

AdamW is the standard choice for training transformers, and `torch.optim.AdamW` is what you will usually reach for.

## A side-by-side comparison

Here is every optimiser on an elongated bowl, $f(w_1, w_2) = \tfrac{1}{2}(w_1^2 + 50\, w_2^2)$, which is 50 times more curved in $w_2$ than in $w_1$. The gradient is exact here (no mini-batch noise), so the differences come purely from the update rules.

```python
import numpy as np

# An elongated bowl: 50 times steeper along w2 than w1. Minimum at (0, 0).
def loss(w):
    return 0.5 * (w[0] ** 2 + 50 * w[1] ** 2)

def grad(w):
    return np.array([w[0], 50 * w[1]])

def sgd(w, g, s, lr):
    return w - lr * g

def momentum(w, g, s, lr, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g
    return w - lr * s["v"]

def rmsprop(w, g, s, lr, beta=0.9, eps=1e-8):
    s["s"] = beta * s.get("s", 0) + (1 - beta) * g ** 2
    return w - lr * g / (np.sqrt(s["s"]) + eps)

def adam(w, g, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] = s.get("t", 0) + 1
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat = s["m"] / (1 - b1 ** t)
    v_hat = s["v"] / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)

runs = [("SGD", sgd, 0.035), ("SGD", sgd, 0.05), ("Momentum", momentum, 0.01),
        ("RMSProp", rmsprop, 0.1), ("Adam", adam, 0.3)]
for name, step, lr in runs:
    w, state, record = np.array([10.0, 1.0]), {}, []
    for t in range(1, 101):
        w = step(w, grad(w), state, lr)
        if t in (10, 50, 100):
            record.append(f"{loss(w):10.4g}")
    print(f"{name:8s} lr={lr:<5}", " ".join(record))
```

The three columns are the loss after 10, 50 and 100 steps, starting from a loss of 75:

```output
SGD      lr=0.035       24.6      1.418    0.04022
SGD      lr=0.05   8.315e+04  1.016e+19  4.132e+36
Momentum lr=0.01       19.48     0.2955  0.0005556
RMSProp  lr=0.1        34.64      10.55     0.3976
Adam     lr=0.3        27.92     0.0731   0.001384
```

What this shows:

- **SGD has a hard ceiling.** The steep direction has curvature 50, so the stability limit is $\eta < 2/50 = 0.04$. At 0.035 it converges; at 0.05 it explodes. Yet even 0.035 is tiny for the flat direction.
- **Momentum wins on this smooth quadratic**, even with a smaller learning rate, because velocity builds up along the ravine floor.
- **RMSProp** moves fast at first, then stalls near 0.4: with a constant learning rate its per-parameter steps stay roughly $\eta$ in size, so it keeps hopping around the minimum. A decaying learning rate fixes this.
- **Adam** tolerates a learning rate several times SGD's limit. Rerun it with any rate from 0.2 to 1.0 and it still converges, which is the property that matters in practice.

Do not over-read a toy. On real deep networks, with noisy gradients and parameters on wildly different scales, Adam-style methods are usually far easier to tune, while well-tuned SGD with momentum remains competitive on some vision tasks.

> [!KEY]
> Momentum fixes the ravine by averaging gradients over time. RMSProp fixes it by rescaling each parameter. Adam does both, which is why it is the default starting point for deep learning and why its learning rate transfers between problems better than SGD's.

## Learning-rate schedules

A fixed learning rate is a compromise: large is fast early but noisy late, small is precise but slow. **Schedules** change $\eta$ over training.

| Schedule | Rule | Typical use |
| --- | --- | --- |
| Step decay | Multiply by a factor such as 0.1 every N epochs | Older CNN recipes |
| Exponential decay | $\eta_t = \eta_0\, \gamma^t$ | Simple, smooth decay |
| Cosine decay | Follows half a cosine from $\eta_{\max}$ to $\eta_{\min}$ | Transformers, most modern training |
| Linear warm-up | Rise from near 0 to $\eta_{\max}$ over the first steps | Combined with cosine or linear decay |
| Reduce on plateau | Cut $\eta$ when validation loss stops improving | When you cannot predict training length |

**Warm-up** matters for Adam in particular: early on, the second-moment estimates are based on very few gradients and are unreliable, and large random-initialised networks are fragile. Starting small and ramping up avoids early blow-ups. The common pairing is linear warm-up followed by cosine decay:

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi\,(t - t_w)}{T - t_w}\right) \quad \text{for } t \ge t_w
$$

where $t_w$ is the number of warm-up steps and $T$ the total.

```python
import math

def lr_at(step, peak=3e-4, warmup=100, total=1000, floor=3e-5):
    if step < warmup:
        return peak * (step + 1) / warmup                    # linear warm-up
    progress = (step - warmup) / (total - warmup)            # 0 -> 1
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))

for step in [0, 49, 99, 100, 325, 550, 775, 999]:
    print(f"step {step:4d}: lr = {lr_at(step):.2e}")
```

```output
step    0: lr = 3.00e-06
step   49: lr = 1.50e-04
step   99: lr = 3.00e-04
step  100: lr = 3.00e-04
step  325: lr = 2.60e-04
step  550: lr = 1.65e-04
step  775: lr = 6.95e-05
step  999: lr = 3.00e-05
```

> [!TIP]
> If training diverges (loss jumps to `nan` or explodes), lower the learning rate or lengthen the warm-up before changing anything else. If loss falls but plateaus early, the learning rate is often too low or decays too soon. The learning rate is the single most important hyperparameter in this lesson.

You will use these through framework APIs in [optimisers in neural networks](/learn/neural-network-fundamentals/optimizers) and [PyTorch optimisers](/learn/pytorch/optimizers); the equations here are what those classes compute.

## Exercises

**1.** Plain gradient descent on $f(w) = \tfrac{a}{2}w^2$ updates $w \leftarrow w - \eta a w = (1 - \eta a)w$. For which $\eta$ does it converge? Use this to explain why SGD at 0.05 exploded in the comparison above.

<details>
<summary>Show solution</summary>

It converges when $|1 - \eta a| < 1$, that is $0 < \eta < 2/a$. In the comparison, the $w_2$ direction has $a = 50$, so $\eta$ must be below 0.04. At 0.05, $|1 - 0.05 \times 50| = 1.5$, so $w_2$ grows by 50% each step and flips sign, which is the explosion you saw.

</details>

**2.** With momentum $\beta = 0.9$ and a constant gradient $g$, what does the velocity $v_t = \beta v_{t-1} + g$ converge to? What is the effective learning rate?

<details>
<summary>Show solution</summary>

At the fixed point $v = 0.9v + g$, so $v = 10g$. The update becomes $\eta \cdot 10g$: momentum multiplies the effective learning rate by $1/(1-\beta) = 10$ in directions where the gradient is consistent. This is why momentum often needs a smaller $\eta$ than plain SGD.

</details>

**3.** With a constant learning rate of 0.3, RMSProp on the bowl above ends at a loss of about 0.61 after 100 steps: it hops around the minimum. Run it again with a cosine schedule (peak 0.3, no warm-up, floor 0.001, 100 steps) and compare.

<details>
<summary>Show solution</summary>

```python
import math

def lr_cos(t, peak=0.3, floor=0.001, total=100):
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t / total))

w, state = np.array([10.0, 1.0]), {}
for t in range(100):
    w = rmsprop(w, grad(w), state, lr_cos(t))
print(f"{loss(w):.4g}")
```

```output
3.192e-07
```

The large early steps cover the distance quickly, and the shrinking late steps let RMSProp settle into the minimum instead of bouncing around it. The loss improves by about six orders of magnitude with the same optimiser and the same peak rate.

</details>

```quiz
[
  {
    "q": "Why does SGD use mini-batches instead of the full dataset to compute each gradient?",
    "options": [
      "Mini-batch gradients are more accurate",
      "Full-dataset gradients cannot be computed for neural networks",
      "Mini-batch gradients are cheap, unbiased estimates, so you can take many more steps for the same compute",
      "Mini-batches remove the need for a learning rate"
    ],
    "answer": 2,
    "why": "A mini-batch gives a noisy but unbiased estimate of the full gradient at a fraction of the cost, and the noise can even help generalisation."
  },
  {
    "q": "In Adam, what is the purpose of dividing m_t by (1 - β1^t)?",
    "options": [
      "To correct the bias towards zero caused by initialising m at zero",
      "To apply weight decay",
      "To make the learning rate decay over time",
      "To prevent division by zero"
    ],
    "answer": 0,
    "why": "m and v start at zero, so early estimates are too small. Bias correction rescales them; its effect disappears as t grows."
  },
  {
    "q": "What does RMSProp divide each parameter's gradient by?",
    "options": [
      "The learning rate",
      "The square root of a running average of that parameter's squared gradients",
      "The number of parameters",
      "The norm of the full gradient vector"
    ],
    "answer": 1,
    "why": "Each parameter gets its own scale, so steep directions take smaller steps and flat directions larger ones."
  },
  {
    "q": "What does AdamW change compared with Adam plus L2 regularisation?",
    "options": [
      "It removes the second-moment estimate",
      "It uses a cosine learning-rate schedule by default",
      "It removes bias correction",
      "It applies weight decay directly to the parameters instead of through the adaptive gradient scaling"
    ],
    "answer": 3,
    "why": "With L2 inside Adam, the decay term is divided by √v̂ and weakened for some parameters. AdamW decouples it."
  }
]
```
