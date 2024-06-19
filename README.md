# The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric

Daniel Severo, Lucas Theis, Johannes Ballé

An **unofficial** Pytorch implementation of LASI (The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric).

The officail JAX implementation is [here](https://github.com/dsevero/Linear-Autoregressive-Similarity-Index).

Details can be found in the following paper:
```
@inproceedings{severo2024the,
    title={The Unreasonable Effectiveness of Linear Prediction as a Perceptual Metric},
    author={Daniel Severo and Lucas Theis and Johannes Ball{\'e}},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=e4FG5PJ9uC}
}
```

## Requirements
- Python 3.10+
- Pytorch 2.0+


## Differences from the official implementation

### Vectorized alternatives for `vmap`
Methods in `LASI` will usually come in pairs: a method that takes a tensor
element as input and another that takes the entire tensor itself. The latter
is a helper that vmaps the former over the tensor **if vmap is available and specified clearly, 
otherwise a hand-written vectorized version is provided**.

### JIT
My unofficial implementation does not utilize JIT compilation.

### Numberical Errors

Please note that the implemented `LASI.compute_distance` is 
*NOT NUMBERICALLY SAME* with the JAX version due to the following reasons:
1. The numerical accuracy of JAX seems to be lower, which I'm not pretty sure about.

    1.1 `80 * jnp.eye(3) / 127.5 != 80 / 127.5 * jnp.eye(3)`.
        (The right term is more accurate and is equal to `80 * torch.eye(3) / 127.5`)

    1.2 Accumulated error invovled by `sum(axis=0)`.
2. `pinv` is not numberically stable, but the error is ignorably small(`1e-16`).

For reference, I provide the following code snippet and the results from the JAX and Pytorch implementations.

```python
from PIL import Image
from lasi_pytorch import LASI
import torch
import numpy as np

# load images
img_megg = Image.open('assets/megg.png').convert('RGB')
img_megg = torch.tensor(np.array(img_megg))
img_dark_megg = Image.open('assets/dark_megg.png').convert('RGB')
img_dark_megg = torch.tensor(np.array(img_dark_megg))
assert img_dark_megg.shape == img_megg.shape

# Compute the distance between img_megg and img_dark_megg.
lasi = LASI(img_megg.shape, neighborhood_size=10)
distance = lasi.compute_distance(img_megg, img_dark_megg)
print(f'd(img_megg, img_dark_megg) = {distance}')
# Result from JAX: d(img_megg, img_dark_megg) = 1.369293212890625
# Result from PyTorch: d(img_megg, img_dark_megg) = 1.3687046766281128
# Difference: 0.0005885362625122

# Efficiently compute the distance between multiple images relative to a reference (img_megg).
img_megg_offset = torch.clip(img_megg + 20, 0 ,255)
distances = lasi.compute_distance_multiple(
    ref=img_megg, p0=img_dark_megg, p1=img_megg_offset)
print(f"d(ref, p0) = {distances['p0']}")
print(f"d(ref, p1) = {distances['p1']}")
# Result from JAX: d(ref, p1) = 1.3496346473693848
# Result from PyTorch: d(ref, p1) = 1.349355697631836
# Difference: 0.0002789497375488
```
