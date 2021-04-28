=================
onnxruntime-numpy
=================
.. image:: https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_apis/build/status/gf712.onnxruntime-numpy?branchName=main
   :alt: onnxruntime-numpy main branch build status
   :target: https://dev.azure.com/OnnxruntimeNumpy/onnxruntime_numpy/_build?definitionId=1

NumPy API with `onnxruntime <https://github.com/microsoft/onnxruntime/>`__ backend.

Description
-----------

onnxruntime-numpy provides the same API as you would expect from
`NumPy <https://github.com/numpy/numpy>`__, but all the execution
happens using
`onnxruntime <https://github.com/microsoft/onnxruntime/>`__.

.. code:: python

   import onnxruntime_numpy as onp
   X = onp.array([[1, 2, 3]])
   w = onp.array([2, 4, 6])
   b = onp.array([1])
   y = X @ w + b

All computations are performed lazily, which means that in the example
above ``y`` is only evaluated when its values are required (for example
``print(y)`` or looping through ``y``).

This means that the operations with ``onp`` do not perform any
computation, and these are all dispatched to ``onnxruntime``. In fact,
in the backend ``onp`` builds a ``ONNX`` graph that is then consumed by
``onnxruntime``.

Currently it is only possible to build (some) neural networks to perform
inference, but there are plans to enable neural network training using a
``grad`` function (and some additional helper functions).

Roadmap
=======
.. include:: Roadmap.rst