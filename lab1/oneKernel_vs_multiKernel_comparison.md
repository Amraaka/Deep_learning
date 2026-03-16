# oneKernel.ipynb vs multiKernel.ipynb — Code Comparison

This document summarizes the **specific code differences** between the single-kernel and multi-kernel ConvNet implementations.

---

## 1. Data Loading & Dependencies

| Aspect | oneKernel.ipynb | multiKernel.ipynb |
|--------|-----------------|--------------------|
| **Framework** | TensorFlow | PyTorch |
| **Imports** | `tensorflow`, `numpy` | `torch`, `torchvision`, `matplotlib`, `seaborn`, `numpy` |
| **MNIST** | `tf.keras.datasets.mnist` | `torchvision.datasets.MNIST` with transforms |
| **Data format** | NumPy arrays from TF | NumPy from `dataset.data.numpy()` |

```python
# oneKernel
import tensorflow as tf
Mnist = tf.keras.datasets.mnist
(Xtr, Ytr), (Xte, Yte) = Mnist.load_data()

# multiKernel
import torch
from torchvision import datasets, transforms
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
Xtr = train_dataset.data.numpy().astype(np.float32) / 255.0
```

---

## 2. ConvLayer — Main Difference

### oneKernel: Single Kernel (2D filter)

- **Filter shape:** `(filter_dim, filter_dim)` — one 2D kernel
- **Bias:** single scalar per layer
- **Input:** `(batch, height, width)` — 3D
- **Output:** `(batch, out_h, out_w)` — 3D

```python
# oneKernel ConvLayer
def __init__(self, filter_dim=3, stride=1, pad=1, alpha=0.01):
    self.filter = np.random.randn(self.filter_dim, self.filter_dim)
    self.filter = self.filter / self.filter.sum()
    self.bias = np.random.rand() / 10

def forward_pass(self, X):
    (d, p, t) = self.X.shape
    self.z = np.zeros((d, dimen_x, dimen_y))
    for i in range(d):
        self.z[i] = self.convolving(self.X[i], self.filter, dimen_x, dimen_y) + self.bias
```

### multiKernel: Multiple Kernels (4D filters)

- **Filter shape:** `(num_kernels, in_channels, filter_dim, filter_dim)` — many 3D kernels
- **Bias:** one per kernel `(num_kernels,)`
- **Input:** `(N, in_ch, H, W)` — 4D (batch, channels, height, width)
- **Output:** `(N, num_kernels, out_h, out_w)` — 4D

```python
# multiKernel ConvLayer
def __init__(self, filter_dim=3, stride=1, pad=1, alpha=0.01, num_kernels=8, in_channels=None):
    self.num_kernels = num_kernels
    self.in_channels = in_channels  # inferred from input if None

def _init_weights(self, in_ch):
    self.filter = np.random.randn(self.num_kernels, in_ch, self.filter_dim, self.filter_dim) / fac
    self.bias = np.random.randn(self.num_kernels) / fac

def forward_pass(self, X):
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]  # (N,H,W) -> (N,1,H,W)
    self.z = np.zeros((N, self.num_kernels, dimen_x, dimen_y))
    for n in range(N):
        for k in range(self.num_kernels):
            self.z[n, k] = self._convolve_single(self.X[n], self.filter[k], ...) + self.bias[k]
```

---

## 3. Convolution Helper

| oneKernel | multiKernel |
|-----------|-------------|
| `convolving(X, fil, dimen_x, dimen_y)` — 2D patch × 2D filter | `_convolve_single(X, fil, dimen_x, dimen_y)` — 3D patch × 3D filter |
| `X` is `(height, width)` | `X` is `(in_ch, height, width)` |
| `patch = X[x0:x0+F, y0:y0+F]` | `patch = X[:, x0:x0+F, y0:y0+F]` |
| `z[i,ii] = (patch * fil).sum()` | `z[i,ii] = np.sum(patch * fil)` |

---

## 4. Backpropagation

### oneKernel

- `grad_z` shape: `(batch, out_h, out_w)`
- `grad_filter`: `(filter_dim, filter_dim)`
- `grad_bias`: scalar

### multiKernel

- `grad_z` shape: `(N, K, out_h, out_w)` where K = num_kernels
- `grad_filter`: `(num_kernels, in_ch, filter_dim, filter_dim)`
- `grad_bias`: `(num_kernels,)`
- Loops over batch, output positions, filter positions, channels, and kernels

---

## 5. Pooling

| Aspect | oneKernel | multiKernel |
|--------|-----------|-------------|
| **Input** | 3D `(batch, H, W)` | 3D or 4D; 3D is treated as `(batch, 1, H, W)` |
| **Output** | 3D | 4D `(batch, n_channels, out_h, out_w)` |
| **Logic** | One channel only | Loops over `batch` and `n_channels` |

```python
# oneKernel — single channel
(q, p, t) = data.shape
after_pool = np.zeros((q, z_x, z_y))

# multiKernel — multi-channel
if data.ndim == 3:
    data = data[:, np.newaxis, :, :]
(q, n_channels, p, t) = data.shape
after_pool = np.zeros((q, n_channels, z_x, z_y))
for batch in range(q):
    for ch in range(n_channels):
        # ... max pooling per channel
```

---

## 6. Padding

| oneKernel | multiKernel |
|-----------|-------------|
| 3D only: `np.pad(data, ((0,0), (p,p), (p,p)))` | Supports 3D and 4D |
| `backprop`: `y[:, 1:-1, 1:-1]` | `backprop`: `y[:, :, p:-p, p:-p]` for 4D |

```python
# multiKernel padding
def forward_pass(self, data):
    if data.ndim == 3:
        return np.pad(data, ((0, 0), (p, p), (p, p)), ...)
    elif data.ndim == 4:
        return np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)), ...)
```

---

## 7. ReLU

| oneKernel | multiKernel |
|-----------|-------------|
| Element-wise loops, `derivative()` helper | Vectorized: `np.maximum(0, z)` and `grad_previous * (self.z > 0)` |
| Handles 2D and 3D manually | Works for any shape |

---

## 8. Reshaping

| oneKernel | multiKernel |
|-----------|-------------|
| `a.reshape(shape_a[0], shape_a[1]*shape_a[2])` | `a.reshape(a.shape[0], -1)` |
| Assumes 3D input | Works for any shape (flattens all but batch) |

---

## 9. Network Architecture & Linear Layer Input Size

### oneKernel

- Single channel throughout → final conv output `(batch, 5, 5)`
- `Linear_Layer(5*5, 24, ...)` → `Linear_Layer(24, 10, ...)`

```python
complete_NN = ConvNet([
    padding(), ConvLayer(), Pooling(), relu(),
    padding(), ConvLayer(), Pooling(), relu(),
    ConvLayer(), relu(),
    reshaping(),
    Linear_Layer(5*5, 24, alpha=al),
    relu(),
    Linear_Layer(24, 10, alpha=al),
    softmax()
])
```

### multiKernel

- Multiple channels: 8 → 8 → 16
- Final conv output `(batch, 16, 5, 5)` → flattened `16*5*5`
- `Linear_Layer(16*5*5, 24, ...)` → `Linear_Layer(24, 10, ...)`

```python
complete_NN = ConvNet([
    padding(),
    ConvLayer(num_kernels=8, alpha=al),
    Pooling(), relu(),
    padding(),
    ConvLayer(num_kernels=8, in_channels=8, alpha=al),
    Pooling(), relu(),
    ConvLayer(num_kernels=16, in_channels=8, alpha=al),
    relu(),
    reshaping(),
    Linear_Layer(16*5*5, 24, alpha=al),
    relu(),
    Linear_Layer(24, 10, alpha=al),
    softmax()
])
```

---

## 10. Summary Table

| Component | oneKernel | multiKernel |
|-----------|-----------|-------------|
| **Conv filter** | 1 kernel `(F,F)` | K kernels `(K,C,F,F)` |
| **Conv bias** | 1 scalar | K scalars |
| **Data format** | 3D `(N,H,W)` | 4D `(N,C,H,W)` |
| **Pooling** | 3D in/out | 4D in/out |
| **Padding** | 3D only | 3D and 4D |
| **Reshape** | `5*5` | `16*5*5` |
| **Linear in_dim** | 25 | 400 |
| **Dependencies** | TensorFlow | PyTorch |

---

## 11. Why Multi-Kernel Improves Performance

- **More feature maps:** Each kernel learns different patterns (edges, textures, etc.).
- **Richer representation:** 16 channels vs 1 channel before the FC layer.
- **More parameters:** More capacity to fit the data.
- **Typical accuracy:** multiKernel reaches ~95%+; oneKernel stays lower with a single kernel.
