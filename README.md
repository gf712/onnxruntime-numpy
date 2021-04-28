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

-   Minimal (does not have to be efficient) [grad]{.title-ref} and
    [jacobian]{.title-ref} support

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
