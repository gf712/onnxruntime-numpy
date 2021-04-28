Pre-release
~~~~~~~~~~~

-  Come up with a name!

-  Support all ONNX operators supported by onnxruntime

-  Design how classes are registered

-  Handle functions that return multiple (potentially optional) arrays
   (e.g. GRU/LSTM return last state and/or all hidden states)

-  Support all ONNX operator type/shape constraints

-  Design the minimum required utility functions to perform checks

-  Support all ONNX types that make sense with python

-  Support ``@jit`` decorator for graph caching

   .. code:: python

      # the correponding graph is generated once and cached
      @jit
      def linear(X, w, b):
        return X @ w + b

      # expression is evaluated in every function call
      def linear_slow(X, w, b):
        return X @ w + b

-  Minimal (does not have to be efficient) grad and jacobian support

-  PRNG

-  Have clear strategy on how to align with onnxruntime releases

-  Should this library only support a single onnxruntime version in each
   release?

   -  If not:

      -  define how to register functions + version

-  Define to what extent and how to interact with onnxruntime

   -  how to set the provider? Maybe try GPU, if not possible use CPU?
   -  custom operators possible? if so how? (not required for
      pre-release, but should know if makes sense to accomodate this
      possibility)
   -  how to set graph optimisation? always extended?
   -  should it handle minimal builds? i.e. certain ops may not be
      available

-  Sort out CI

   -  Azure pipelines?
   -  Which platforms to support? Build against all of these?

-  Add ``ONNX`` graph visualisation. Maybe something like: \```python
   import onnxruntime_numpy as onp def add(x, y): return x + y

   # current design would require at least some dummy inputs here
   onp.display(add, x, y) \``\`

Post-release
~~~~~~~~~~~~

-  Support more operations composed of ONNX operators
-  I am looking at you obscure bessel functions needed for Matern kernel

.. |onnxruntime-numpy main branch build status| image:: https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_apis/build/status/gf712.onnxruntime-numpy?branchName=main
   :target: https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_build?definitionId=1
