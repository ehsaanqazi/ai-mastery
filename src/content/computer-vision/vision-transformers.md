For years, computer vision meant convolutions. The **Vision Transformer** (ViT) showed that a nearly unmodified Transformer, the architecture behind modern language models, can classify images just as well when trained on enough data. The trick is to turn an image into a sequence: cut it into small square patches, treat each patch like a word token, and let self-attention relate every patch to every other patch. ViTs and their descendants are now standard backbones for classification, detection and segmentation, and they are the image encoders inside most multimodal models that take pictures as input. This lesson builds a ViT from its parts and compares it with the CNNs from earlier in the module.

If you need a refresher on self-attention, multi-head attention and the encoder block, read [Transformer architecture](/learn/advanced-nlp/transformer-architecture) first. Here we focus on what changes when the input is an image.

## From image to sequence

A Transformer takes a sequence of vectors. An image of shape $C \times H \times W$ is not one, so ViT makes one:

1. Split the image into non-overlapping $P \times P$ patches. There are $N = \frac{H W}{P^2}$ of them.
2. Flatten each patch into a vector of length $P^2 C$.
3. Multiply each vector by a learned matrix $E$ to get a $D$-dimensional **patch embedding**.

For the standard setting, a $224 \times 224$ RGB image with $P = 16$: there are $(224/16)^2 = 14^2 = 196$ patches, each flattened to $16 \cdot 16 \cdot 3 = 768$ numbers.

```python
import torch
from torch import nn

torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)
P = 16

# (1, 3, 224, 224) -> (1, 3, 14, 16, 14, 16) -> (1, 14, 14, 3, 16, 16) -> (1, 196, 768)
patches = img.reshape(1, 3, 14, P, 14, P).permute(0, 2, 4, 1, 3, 5).reshape(1, 196, 3 * P * P)
print(patches.shape)
```

```output
torch.Size([1, 196, 768])
```

### Patch embedding is a strided convolution

Applying the same linear map to every non-overlapping patch is exactly what a convolution does when its kernel size and stride both equal $P$. Real implementations use that, because it is simpler and faster:

```python
D = 192
linear = nn.Linear(3 * P * P, D)
conv = nn.Conv2d(3, D, kernel_size=P, stride=P)

with torch.no_grad():   # give the conv the same weights as the linear layer
    conv.weight.copy_(linear.weight.reshape(D, 3, P, P))
    conv.bias.copy_(linear.bias)

a = linear(patches)                               # (1, 196, D)
b = conv(img).flatten(2).transpose(1, 2)          # (1, D, 14, 14) -> (1, 196, D)
print(a.shape, b.shape, torch.allclose(a, b, atol=1e-5))
```

```output
torch.Size([1, 196, 192]) torch.Size([1, 196, 192]) True
```

So a ViT does begin with one convolution, just a large-stride one with no overlap.

## Class token and position embeddings

Two more ingredients complete the input sequence.

- **Class token.** A learned vector $x_{\text{class}}$ is prepended to the sequence. It belongs to no patch; through attention it gathers information from all of them, and its final state is fed to the classification head. (Averaging all patch outputs instead is a common, equally valid alternative.)
- **Position embeddings.** Self-attention treats its input as a set: shuffle the patches and the outputs shuffle with them. To tell the model where each patch came from, a learned vector per position is added to the embeddings.

The input to the Transformer is therefore

$$
z_0 = \left[\, x_{\text{class}};\; x_p^1 E;\; x_p^2 E;\; \dots;\; x_p^N E \,\right] + E_{\text{pos}}
$$

where $x_p^i$ is the $i$-th flattened patch, $E$ is the $(P^2 C) \times D$ patch-embedding matrix, $E_{\text{pos}}$ is an $(N+1) \times D$ matrix of position embeddings, and the semicolons mean stacking along the sequence. The sequence has $N + 1$ tokens of dimension $D$.

```mermaid
flowchart LR
    A["Image<br/>3 x 224 x 224"] --> B["Split into 16 x 16 patches<br/>196 patches"]
    B --> C["Linear patch embedding<br/>196 x D"]
    C --> D["Prepend class token<br/>197 x D"]
    D --> E["Add position embeddings"]
    E --> F["Transformer encoder<br/>L blocks of attention and MLP"]
    F --> G["Take class token output<br/>D"]
    G --> H["MLP head<br/>class logits"]
```

## The encoder and the head

The rest is a standard Transformer encoder: $L$ identical blocks, each with layer normalisation, multi-head self-attention, a residual connection, then layer normalisation, an MLP and another residual connection. ViT puts the normalisation **before** each sub-layer (pre-norm):

$$
z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell
$$

where MSA is multi-head self-attention, LN is layer normalisation and $\ell = 1, \dots, L$ indexes the blocks. PyTorch provides this block as `nn.TransformerEncoderLayer` with `norm_first=True`. Putting everything together:

```python
class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=64, depth=4, heads=4, num_classes=10):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)    # (B, N, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)       # (B, 1, dim)
        x = torch.cat([cls, x], dim=1) + self.pos_embed       # (B, N+1, dim)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))                  # class token only

vit = TinyViT()
x = torch.randn(8, 3, 32, 32)
print("tokens:", vit.patch_embed(x).flatten(2).shape[-1] + 1)
print("logits:", tuple(vit(x).shape))
print("parameters:", sum(p.numel() for p in vit.parameters()))
```

```output
tokens: 65
logits: (8, 10)
parameters: 208074
```

A $32 \times 32$ image with $4 \times 4$ patches gives 64 patch tokens plus the class token. Swap this model into the training loop from [CNNs](/learn/computer-vision/convolutional-neural-networks-cnns) and it trains the same way. torchvision ships full-size versions, such as `torchvision.models.vit_b_16` with pretrained `ViT_B_16_Weights`, which you would normally fine-tune rather than train from scratch.

## Patch size and cost

Self-attention compares every token with every other, so its cost grows with the square of the number of tokens. Patch size is therefore the main dial between detail and compute:

```python
for p in [32, 16, 14, 8]:
    n = (224 // p) ** 2
    print(f"patch {p:2d}: {n:4d} tokens, {n * n:>9,} attention pairs per head per layer")
```

```output
patch 32:   49 tokens,     2,401 attention pairs per head per layer
patch 16:  196 tokens,    38,416 attention pairs per head per layer
patch 14:  256 tokens,    65,536 attention pairs per head per layer
patch  8:  784 tokens,   614,656 attention pairs per head per layer
```

Halving the patch size from 16 to 8 quadruples the tokens and multiplies the attention cost by sixteen. This is why ViTs work on modest resolutions and why dense tasks on large images use variants that restrict attention, such as attention within local windows arranged in a CNN-like hierarchy (the Swin Transformer is the best-known example).

> [!WARNING]
> Position embeddings are learned for one grid size. Fine-tuning at a higher resolution with the same patch size produces more patches than there are embeddings; the standard fix is to interpolate the position embeddings to the new grid. Library implementations usually handle this, but a hand-rolled model will fail with a shape error.

## ViTs versus CNNs

| | CNN | Vision Transformer |
| --- | --- | --- |
| Built-in assumptions (inductive bias) | strong: locality and translation equivariance | weak: only the patch grid; relationships are learned |
| Receptive field | grows gradually with depth | global from the first layer |
| Data needs from scratch | works well on small and medium datasets | needs large datasets or strong augmentation and regularisation |
| Scaling with data and model size | good | very good, which is why the largest vision models are mostly Transformers |
| Cost versus resolution | roughly linear in pixels | quadratic in the number of patches |
| Shares architecture with language models | no | yes, which makes multimodal models natural |

The weak inductive bias cuts both ways. With little data, a ViT has to learn from scratch what a CNN gets for free, such as "nearby pixels are related", so it tends to underperform. With a lot of data, the same flexibility lets it learn relationships a CNN's fixed local structure would miss. In practice you rarely train either from scratch: you fine-tune a pretrained backbone, and then both families are strong choices, so measure on your own data. Hybrid designs that use convolutions in early stages and attention later, and modern CNNs that borrow Transformer design choices, blur the line further.

> [!KEY]
> A Vision Transformer is a Transformer whose tokens are image patches. Everything specific to images happens in the first few lines: patching, embedding, the class token and position embeddings. After that it is the same encoder used for text.

## Exercises

**1.** How many tokens (including the class token) does a ViT see for a $384 \times 384$ image with $16 \times 16$ patches, and what is the length of each flattened patch for an RGB image?

<details>
<summary>Show solution</summary>

$384 / 16 = 24$, so there are $24^2 = 576$ patches plus 1 class token: 577 tokens. Each flattened patch has $16 \cdot 16 \cdot 3 = 768$ values.

```python
print((384 // 16) ** 2 + 1, 16 * 16 * 3)
```

```output
577 768
```

</details>

**2.** Change `TinyViT` to classify from the mean of the patch outputs instead of the class token. Which lines change?

<details>
<summary>Show solution</summary>

Remove the class token, size the position embeddings for $N$ tokens, and average over the sequence axis:

```python
class MeanPoolViT(TinyViT):
    def __init__(self, **kw):
        super().__init__(**kw)
        n = self.pos_embed.shape[1] - 1
        self.pos_embed = nn.Parameter(torch.randn(1, n, self.pos_embed.shape[2]) * 0.02)
        del self.cls_token

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x.mean(dim=1)))

print(tuple(MeanPoolViT()(torch.randn(2, 3, 32, 32)).shape))
```

```output
(2, 10)
```

</details>

**3.** A team has 2,000 labelled images and limited compute. They are deciding between training a ViT from scratch and fine-tuning a pretrained CNN or ViT. What would you advise, and why?

<details>
<summary>Show solution</summary>

Do not train a ViT from scratch on 2,000 images: with its weak inductive bias it needs far more data to learn basic spatial structure, and it will probably underperform a CNN. Fine-tune a pretrained backbone instead. Try a small pretrained CNN and a small pretrained ViT with the same fine-tuning recipe and pick the better one on a held-out validation set, taking inference cost into account.

</details>

```quiz
[
  {
    "q": "How many patch tokens does a ViT produce from a 224 x 224 image with 16 x 16 patches?",
    "options": ["196", "224", "256", "14"],
    "answer": 0,
    "why": "224 / 16 = 14 patches per side, and 14 x 14 = 196. Adding the class token gives 197 tokens."
  },
  {
    "q": "Why does a ViT need position embeddings?",
    "options": [
      "To normalise the pixel values",
      "To reduce the number of parameters",
      "Self-attention is permutation-equivariant, so without them the model cannot tell where each patch came from",
      "To replace the class token"
    ],
    "answer": 2,
    "why": "Attention treats its input as a set. Adding a learned vector per position injects spatial order."
  },
  {
    "q": "Which PyTorch layer computes the patch embedding of a ViT in one step?",
    "options": [
      "`nn.MaxPool2d(kernel_size=P)`",
      "`nn.Conv2d(C, D, kernel_size=P, stride=P)`",
      "`nn.Embedding(N, D)`",
      "`nn.ConvTranspose2d(C, D, kernel_size=P)`"
    ],
    "answer": 1,
    "why": "A convolution whose kernel and stride equal the patch size applies one shared linear map to each non-overlapping patch."
  },
  {
    "q": "Compared with a CNN trained from scratch on a small dataset, a ViT trained from scratch typically:",
    "options": [
      "Performs better, because attention is global",
      "Needs no augmentation",
      "Trains faster, because it has no convolutions",
      "Performs worse, because it lacks the CNN's built-in locality and translation assumptions"
    ],
    "answer": 3,
    "why": "Weak inductive bias means the ViT must learn spatial structure from data, which small datasets do not supply."
  }
]
```
