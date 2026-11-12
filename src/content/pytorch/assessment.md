This quiz covers the whole PyTorch module: tensors and their operations, autograd, building models with `nn.Module`, the training loop, data loading, losses, optimisers, GPUs, saving models and transfer learning. Aim for at least 13 out of 16 before moving on to TensorFlow and Keras.

```quiz
[
  {
    "q": "What dtype does `torch.tensor([0, 1, 2])` have, and is it suitable as targets for `nn.CrossEntropyLoss`?",
    "options": [
      "`torch.float32`; no, targets must be integers",
      "`torch.int32`; no, targets must be int64",
      "`torch.int64`; yes, class-index targets must be int64",
      "`torch.float64`; yes"
    ],
    "answer": 2,
    "why": "Integer literals default to int64 (`torch.long`), which is what class-index targets require."
  },
  {
    "q": "Which conversion produces a tensor that shares memory with a NumPy array `arr`?",
    "options": ["`torch.tensor(arr)`", "`torch.from_numpy(arr)`", "`torch.as_tensor(arr.tolist())`", "`torch.tensor(arr).clone()`"],
    "answer": 1,
    "why": "`torch.from_numpy` wraps the existing buffer; changes to one are visible in the other."
  },
  {
    "q": "`x` has shape `(32, 1, 28, 28)`. What is the shape of `x.flatten(start_dim=1)`?",
    "options": ["`(32, 784)`", "`(25088,)`", "`(32, 28, 28)`", "`(32, 1, 784)`"],
    "answer": 0,
    "why": "All dimensions from 1 onwards are merged: 1 * 28 * 28 = 784, keeping the batch dimension."
  },
  {
    "q": "A model outputs `pred` of shape `(64, 1)` and the targets `y` have shape `(64,)`. What happens with `nn.MSELoss()(pred, y)`?",
    "options": [
      "It raises an error immediately",
      "It computes the correct loss",
      "It broadcasts to a `(64, 64)` difference matrix and returns a wrong loss, with only a warning",
      "It uses only the first prediction"
    ],
    "answer": 2,
    "why": "Broadcasting silently pairs every prediction with every target. Make the shapes identical."
  },
  {
    "q": "What does this print?\n\n```python\nw = torch.tensor(1.0, requires_grad=True)\nfor _ in range(2):\n    (3 * w).backward()\nprint(w.grad)\n```",
    "options": ["`tensor(3.)`", "`tensor(0.)`", "`None`", "`tensor(6.)`"],
    "answer": 3,
    "why": "Each backward pass adds 3 to `.grad`, because gradients accumulate until you zero them."
  },
  {
    "q": "Why is a manual parameter update written inside `with torch.no_grad():`?",
    "options": [
      "So the update is not recorded in the computation graph, and in-place changes to leaf tensors are allowed",
      "To make the update run faster on the GPU",
      "Because `no_grad` zeroes the gradients",
      "To switch dropout off during the update"
    ],
    "answer": 0,
    "why": "Recording the update would make it part of the graph, and PyTorch forbids in-place edits of leaves that require grad."
  },
  {
    "q": "A custom `nn.Module` stores its layers in a plain Python list. What goes wrong?",
    "options": [
      "The forward pass raises an error",
      "The layers are not registered, so `parameters()` misses them and the optimiser never updates them",
      "The layers run on the CPU only",
      "Nothing; lists work the same as `nn.ModuleList`"
    ],
    "answer": 1,
    "why": "Only `nn.Module`, `nn.Parameter` and container attributes like `nn.ModuleList` are registered."
  },
  {
    "q": "What shape is the weight of `nn.Linear(128, 10)`?",
    "options": ["`(128, 10)`", "`(10,)`", "`(10, 128)`", "`(1280,)`"],
    "answer": 2,
    "why": "PyTorch stores linear weights as (out_features, in_features)."
  },
  {
    "q": "Which statement about evaluation is correct?",
    "options": [
      "`model.eval()` alone stops gradient tracking",
      "`torch.no_grad()` alone switches dropout off",
      "Validation needs neither",
      "Validation should use both `model.eval()` and `torch.no_grad()`, because they do different things"
    ],
    "answer": 3,
    "why": "`eval()` changes layer behaviour (dropout, batch norm); `no_grad()` stops graph recording."
  },
  {
    "q": "A classifier ends with `nn.Softmax(dim=1)` and is trained with `nn.CrossEntropyLoss`. What is the effect?",
    "options": [
      "Softmax is applied twice, so the loss cannot approach zero and learning is weakened, with no error raised",
      "Training is more stable",
      "An error is raised because the inputs are probabilities",
      "Nothing; the loss detects probabilities and skips its own softmax"
    ],
    "answer": 0,
    "why": "`CrossEntropyLoss` expects logits and applies log-softmax internally."
  },
  {
    "q": "For a multi-label problem with 6 tags per example, which setup is correct?",
    "options": [
      "6 logits, `CrossEntropyLoss`, long targets of shape `(N,)`",
      "6 logits, `BCEWithLogitsLoss`, float targets of shape `(N, 6)`",
      "1 logit, `BCEWithLogitsLoss`, float targets of shape `(N, 1)`",
      "6 softmax probabilities, `MSELoss`"
    ],
    "answer": 1,
    "why": "Each tag is an independent yes-or-no decision, so use one logit and a binary loss per tag."
  },
  {
    "q": "What does a `DataLoader`'s `collate_fn` do?",
    "options": [
      "Shuffles the dataset each epoch",
      "Loads a single example from disk",
      "Combines a list of examples into one batch, for example by stacking or padding",
      "Moves the batch to the GPU"
    ],
    "answer": 2,
    "why": "The default collate stacks tensors; a custom one handles variable-length examples."
  },
  {
    "q": "In which order should these run in a training step?",
    "options": [
      "`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `scheduler.step()`",
      "`scheduler.step()`, `optimizer.step()`, `loss.backward()`, `optimizer.zero_grad()`",
      "`loss.backward()`, `optimizer.zero_grad()`, `optimizer.step()`, `scheduler.step()`",
      "`optimizer.step()`, `loss.backward()`, `optimizer.zero_grad()`, `scheduler.step()`"
    ],
    "answer": 0,
    "why": "Clear gradients, compute new ones, apply them, then advance the learning-rate schedule."
  },
  {
    "q": "On a CUDA machine, what is wrong with this loop body?\n\n```python\nfor xb, yb in loader:\n    xb.to(device)\n    yb = yb.to(device)\n    loss = loss_fn(model(xb), yb)\n```",
    "options": [
      "`yb` should not be moved",
      "`loss_fn` must be moved to the device first",
      "Nothing is wrong",
      "`xb.to(device)` discards its result, so `xb` stays on the CPU"
    ],
    "answer": 3,
    "why": "`tensor.to` returns a new tensor; it must be assigned. Only modules move in place."
  },
  {
    "q": "Why should you pass `weights_only=True` to `torch.load` for files you did not create?",
    "options": [
      "It loads faster",
      "It prevents a malicious pickled file from executing arbitrary code",
      "It converts the weights to float16",
      "It skips the optimiser state"
    ],
    "answer": 1,
    "why": "`torch.save` uses pickle; restricting loading to tensors and simple types blocks code execution."
  },
  {
    "q": "You fine-tune a pretrained ResNet by unfreezing `layer4` and training a new `fc`. Which optimiser setup follows good practice?",
    "options": [
      "One group, `lr=1e-1`, for every parameter including frozen ones",
      "`layer4` at `lr=1e-3` and `fc` at `lr=1e-5`",
      "`layer4` at `lr=1e-4` and `fc` at `lr=1e-3`",
      "Train `layer4` only and leave `fc` frozen"
    ],
    "answer": 2,
    "why": "Pretrained layers get smaller steps so their features are adjusted, not overwritten; the new head learns faster."
  }
]
```
