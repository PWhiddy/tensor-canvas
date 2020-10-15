
# Tensor Canvas 🎨  
[![PyPI version](https://badge.fury.io/py/tensor-canvas.svg)](https://badge.fury.io/py/tensor-canvas)  
  
Tensor Canvas provides a standard API for 2D rendering directly onto tensors with pytorch, tensorflow, jax, and numpy.
SDF representations are used to implement rendering in these gpu-accelerated frameworks, which is inefficient compared to normal gpu rasterization but much more predictable than matplotlib. Integration with ML frameworks also means that it is fully-differentiable. Cross-framework support is possible thanks to the [eagerpy](https://github.com/jonasrauber/eagerpy) library.  
  
Currently only cirlces and lines are supported, but it is straightforward to port [any 2D SDF](https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm).

### Installation  

```bash
pip install tensor-canvas
```

### Example
```python
import tensorcanvas as tc
import torch
import tensorflow as tf
import jax.numpy as jnp
import numpy as np

# define 3 cirlces with different positions, radii, and colors
x1, y1, r1, c1 = 34.8,  5.3, 2.0, [0.3, 0.2, 1.0]
x2, y2, r2, c2 = 14.8, 15.3, 5.0, [0.1, 0.9, 0.8]
x3, y3, r3, c3 = 30.8, 20.3, 3.0, [0.0, 0.9, 0.0]

# canvas dimensions
height, width, channels = 32, 64, 3

# draw 3 colored circles on a pytorch image tensor
pt_canvas = torch.zeros(channels, height, width)
pt_canvas = tc.draw_circle(x1, y1, r1, torch.tensor(c1), pt_canvas)
pt_canvas = tc.draw_circle(x2, y2, r2, torch.tensor(c2), pt_canvas)
pt_canvas = tc.draw_circle(x3, y3, r3, torch.tensor(c3), pt_canvas)

# draw 3 colored cirlces on a tensorflow image tensor
tf_canvas = tf.zeros([height, width, channels])
tf_canvas = tc.draw_circle(x1, y1, r1, tf.convert_to_tensor(c1), tf_canvas)
tf_canvas = tc.draw_circle(x2, y2, r2, tf.convert_to_tensor(c2), tf_canvas)
tf_canvas = tc.draw_circle(x3, y3, r3, tf.convert_to_tensor(c3), tf_canvas)

# draw 3 colored cirlces on a jax image tensor
jx_canvas = jnp.zeros([height, width, channels])
jx_canvas = tc.draw_circle(x1, y1, r1, jnp.array(c1), jx_canvas)
jx_canvas = tc.draw_circle(x2, y2, r2, jnp.array(c2), jx_canvas)
jx_canvas = tc.draw_circle(x3, y3, r3, jnp.array(c3), jx_canvas)

# draw 3 colored cirlces on a numpy image tensor
np_canvas = np.zeros([height, width, channels])
np_canvas = tc.draw_circle(x1, y1, r1, np.array(c1), np_canvas)
np_canvas = tc.draw_circle(x2, y2, r2, np.array(c2), np_canvas)
np_canvas = tc.draw_circle(x3, y3, r3, np.array(c3), np_canvas)

# check results are indentical
assert(np.allclose(np_canvas, pt_canvas.permute(1,2,0), atol=1e-6))
assert(np.allclose(np_canvas, tf_canvas, atol=1e-6))
assert(np.allclose(np_canvas, jx_canvas, atol=1e-6))

```

### Notebook Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PWhiddy/TensorCanvasDemo/blob/master/TensorCanvasDemo.ipynb)  
  
<img src="https://i.imgur.com/sspmxHa.png" width="653" height="763">
