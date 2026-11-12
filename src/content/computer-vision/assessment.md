This quiz covers the whole Computer Vision module: images as arrays, classification, CNNs, augmentation, transfer learning, detection, segmentation, YOLO and Vision Transformers. Aim for at least 13 out of 16 before moving on to Sequence Models.

```quiz
[
  {
    "q": "A Pillow image has `size == (800, 600)`. What is the shape of `np.asarray(img)` for an RGB image?",
    "options": ["`(800, 600, 3)`", "`(600, 800, 3)`", "`(3, 600, 800)`", "`(3, 800, 600)`"],
    "answer": 1,
    "why": "Pillow reports (width, height); NumPy arrays are (height, width, channels)."
  },
  {
    "q": "What does this print?\n\n```python\nimport numpy as np\nimg = np.zeros((32, 32, 3))\nprint(img.transpose(2, 0, 1)[np.newaxis].shape)\n```",
    "options": ["`(32, 32, 3, 1)`", "`(1, 32, 32, 3)`", "`(1, 3, 32, 32)`", "`(3, 1, 32, 32)`"],
    "answer": 2,
    "why": "The transpose moves channels first (3, 32, 32) and `np.newaxis` adds a leading batch axis: NCHW."
  },
  {
    "q": "A 3x3 kernel has weights that sum to 1 and are all positive. What kind of filter is it most likely to be?",
    "options": ["An edge detector", "A blur", "A sharpening filter", "An identity that inverts colours"],
    "answer": 1,
    "why": "A positive kernel summing to 1 is a weighted average of the neighbourhood, which smooths the image while preserving brightness."
  },
  {
    "q": "What is the output height of `nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2)` on a 100-pixel-high input?",
    "options": ["48", "100", "51", "50"],
    "answer": 3,
    "why": "floor((100 + 4 - 5) / 2) + 1 = floor(49.5) + 1 = 50."
  },
  {
    "q": "How many parameters does `nn.Conv2d(16, 32, kernel_size=3)` have?",
    "options": ["4,640", "4,608", "512", "1,568"],
    "answer": 0,
    "why": "(3 x 3 x 16 + 1) x 32 = 145 x 32 = 4,640, including one bias per output channel."
  },
  {
    "q": "Three stacked 3x3 convolutions with stride 1 have what receptive field?",
    "options": ["3 x 3", "9 x 9", "7 x 7", "5 x 5"],
    "answer": 2,
    "why": "Each extra 3x3 layer adds 2 pixels: 3, then 5, then 7."
  },
  {
    "q": "You split one dataset with `random_split` and then set `train_ds.dataset.transform = train_tfm`. What happens to the validation subset?",
    "options": [
      "It keeps no transform",
      "It also uses `train_tfm`, because both subsets share the same underlying dataset",
      "It raises an error",
      "It automatically uses an evaluation transform"
    ],
    "answer": 1,
    "why": "`Subset` objects reference one dataset. Create separate dataset objects with different transforms and split their indices instead."
  },
  {
    "q": "Which augmentation is most likely to change the label in a digit-recognition task?",
    "options": ["Small brightness change", "Slight Gaussian blur", "Rotation by 180 degrees", "Small random crop with padding"],
    "answer": 2,
    "why": "Rotating a 6 by 180 degrees produces a 9. Augmentations must preserve the label."
  },
  {
    "q": "You fine-tune a pretrained ResNet on a few hundred images with the backbone frozen. Why put frozen BatchNorm layers in eval mode?",
    "options": [
      "To stop their running statistics drifting towards your small, unusual batches",
      "To freeze their weights, which `requires_grad = False` does not do",
      "To enable dropout",
      "To make the model accept grayscale images"
    ],
    "answer": 0,
    "why": "In train mode BatchNorm updates running mean and variance regardless of `requires_grad`, which can damage pretrained features."
  },
  {
    "q": "Box A is (0, 0, 4, 4) and box B is (2, 0, 6, 4) in xyxy format. What is their IoU?",
    "options": ["0.5", "0.25", "0.4", "0.333"],
    "answer": 3,
    "why": "Each area is 16; the overlap is 2 x 4 = 8; union = 16 + 16 - 8 = 24; IoU = 8 / 24 = 0.333."
  },
  {
    "q": "During NMS with threshold 0.5, the top box overlaps a lower-scoring box of the same class with IoU 0.7. What happens to the lower-scoring box?",
    "options": ["It is kept", "It is merged into the top box by averaging", "It is suppressed", "Its score is doubled"],
    "answer": 2,
    "why": "Standard NMS removes any remaining box whose IoU with a kept box exceeds the threshold."
  },
  {
    "q": "Why is COCO-style mAP@0.5:0.95 stricter than mAP@0.5?",
    "options": [
      "It averages over IoU thresholds up to 0.95, so loosely placed boxes stop counting as correct",
      "It ignores small objects",
      "It only counts the top-scoring detection per image",
      "It uses accuracy instead of precision"
    ],
    "answer": 0,
    "why": "At high IoU thresholds only tightly aligned boxes are true positives, so localisation quality matters."
  },
  {
    "q": "A segmentation model outputs logits of shape `(4, 6, 128, 128)`. What shape and dtype should the target for `nn.CrossEntropyLoss` have?",
    "options": [
      "`(4, 6, 128, 128)`, float",
      "`(4, 128, 128)`, int64 class IDs",
      "`(4, 6)`, int64",
      "`(128, 128)`, float"
    ],
    "answer": 1,
    "why": "For per-pixel classification the target holds one class index per pixel: (N, H, W) of type long."
  },
  {
    "q": "A predicted mask has 40 pixels, the true mask 60, and they share 30. What is the Dice coefficient?",
    "options": ["0.43", "0.5", "0.75", "0.6"],
    "answer": 3,
    "why": "Dice = 2 x 30 / (40 + 60) = 0.6. (IoU would be 30 / 70, about 0.43.)"
  },
  {
    "q": "In a YOLO label file for a 1000 x 500 image, an object spans x from 100 to 300 and y from 100 to 200 in pixels. Which line is correct for class 1?",
    "options": [
      "`1 0.2 0.3 0.2 0.2`",
      "`1 100 100 300 200`",
      "`1 0.1 0.2 0.3 0.4`",
      "`1 200 150 200 100`"
    ],
    "answer": 0,
    "why": "Centre x = 200 / 1000 = 0.2, centre y = 150 / 500 = 0.3, width = 200 / 1000 = 0.2, height = 100 / 500 = 0.2."
  },
  {
    "q": "Which statement about Vision Transformers is correct?",
    "options": [
      "They use max pooling between attention layers to grow the receptive field",
      "Their attention cost grows linearly with the number of patches",
      "Every patch can attend to every other patch from the first layer, giving a global receptive field",
      "They need no position information because attention is order-aware"
    ],
    "answer": 2,
    "why": "Self-attention relates all tokens at once, so the receptive field is global immediately; the cost is quadratic in tokens and position embeddings are required."
  }
]
```
