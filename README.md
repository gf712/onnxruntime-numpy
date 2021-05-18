# onnxruntime-numpy

[![onnxruntime-numpy main branch build status](https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_apis/build/status/gf712.onnxruntime-numpy?branchName=main)](https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_build?definitionId=1)

NumPy API with [onnxruntime](https://github.com/microsoft/onnxruntime/)
backend.

## Description

onnxruntime-numpy provides the same API as you would expect from
[NumPy](https://github.com/numpy/numpy), but all the execution happens
using [onnxruntime](https://github.com/microsoft/onnxruntime/).

```python
import onnxruntime_numpy as onp
X = onp.array([[1, 2, 3]])
w = onp.array([2, 4, 6])
b = onp.array([1])
y = X @ w + b
```

All computations are performed lazily, which means that in the example above `y` is only evaluated when its values are required (for example `print(y)` or looping through `y`). 

This means that the operations with `onp` do not perform any computation, and these are all dispatched to `onnxruntime`. In fact, in the backend `onp` builds a `ONNX` graph that is then consumed by `onnxruntime`.

Currently it is only possible to build (some) neural networks to perform inference, but there are plans to enable neural network training using a `grad` function (and some additional helper functions).

The current implementation progress of the ONNX operators can be found [here](Implementation_Progress.md). Once this list is complete work will start to align the API closer to numpy and add all numpy functions that can be (trivially) implemented with ONNX operators.

## Gradients

The gradient framework is slowly progressing but unstable. The first iteration will have the same functionality as TensorFlow's GradientTape, but without the need to work explicitly with tape. Later iterations will refine this so that the user can compute Jacobian vector products and vector Jacobian products, take the gradient of functions rather than pass Tensors/Arrays to `gradients` (this will then be `jit`able) and compose `grad` functions (e.g. `grad(grad(f))`).

Currently, the plan is to have a simple example such as the one below working (due to lazy evaluation this is currently a bit unstable):

```python
batch_size = 32
n_features = 1
epochs = 100
learning_rate = onp.array(0.01)

def relu(x):
    return np.maximum(0, x)

X = np.random.rand(batch_size, n_features).astype(np.float32)
y = X + (np.random.rand(batch_size, 1).astype(np.float32) / 10.)

w1 = np.random.rand(n_features, 64).astype(np.float32)
b1 = np.random.rand(64).astype(np.float32)

w2 = np.random.rand(64, 1).astype(np.float32)
b2 = np.random.rand(1).astype(np.float32)

X = onp.array(X)
y = onp.array(y)
w1 = onp.array(w1)
b1 = onp.array(b1)
w2 = onp.array(w2)
b2 = onp.array(b2)

for _ in range(epochs):
    result = onp.nn.relu(X @ w1 + b1)
    result = result @ w2 + b2

    loss = onp.square(result - y).mean()

    dw1, db1, dw2, db2 = onp.gradients(loss, [w1, b1, w2, b2])

    w1 += dw1 * learning_rate
    b1 += db1 * learning_rate
    w2 += dw2 * learning_rate
    b2 += db2 * learning_rate

    w1._eval()
    b2._eval()
    w2._eval()
    b2._eval()
```

## Roadmap

### Pre-release

-   Come up with a name!
-   Support all ONNX operators supported by onnxruntime

  -   Design how classes are registered
  -   Handle functions that return multiple (potentially optional)
      arrays (e.g. GRU/LSTM return last state and/or all hidden states)

-   Support all ONNX operator type/shape constraints

-   Design the minimum required utility functions to perform checks

-   Support all ONNX types that make sense with python

-   Support `@jit` decorator for graph caching

    ```python
    # the correponding graph is generated once and cached
    @jit
    def linear(X, w, b):
      return X @ w + b

    # expression is evaluated in every function call
    def linear_slow(X, w, b):
      return X @ w + b
    ```

-   Minimal (does not have to be efficient) `grad` and `jacobian` support

-   PRNG
-   Have clear strategy on how to align with onnxruntime releases

  -   Should this library only support a single onnxruntime version in
      each release?
      -   If not:
          -   define how to register functions + version

-   Define to what extent and how to interact with onnxruntime

    -   how to set the provider? Maybe try GPU, if not possible use CPU?
    -   custom operators possible? if so how? (not required for
        pre-release, but should know if makes sense to accomodate this
        possibility)
    -   how to set graph optimisation? always extended?
    -   should it handle minimal builds? i.e. certain ops may not be
        available

-   Sort out CI
    -   Azure pipelines?
    -   Which platforms to support? Build against all of these?

- Add `ONNX` graph visualisation. Maybe something like:
    ```python
    import onnxruntime_numpy as onp
    def add(x, y):
      return x + y    
    
    # current design would require at least some dummy inputs here
    onp.display(add, x, y)
    ```

### Post-release

- Support more operations composed of ONNX operators
- I am looking at you obscure bessel functions needed for Matern
     kernel
