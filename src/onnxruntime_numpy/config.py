import warnings
import threading
from enum import Enum
from .exceptions import InternalException
from onnxruntime import get_available_providers
import onnxruntime
try:
    import onnxruntime.training  # noqa
    _HAS_ONNXRUNTIME_TRAINING = True
except Exception:
    _HAS_ONNXRUNTIME_TRAINING = False


class GraphOptimizationLevel(Enum):
    DISABLED = 0
    BASIC = 1
    EXTENDED = 2
    ALL = 3


_AVAILABLE_PROVIDERS = get_available_providers()


class Config:
    """Onnxruntime-numpy configuration class.
    Only change configuration options with the setters since these provide
    additional checks and are thread-safe.

    This class uses the singleton pattern (see
    https://python-patterns.guide/gang-of-four/singleton/)
    """
    _instance = None
    _providers = None
    _lock = None
    _training_backend = None
    _optimization_level = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._providers = list(_AVAILABLE_PROVIDERS)
            cls._lock = threading.Lock()
            cls._training_backend = "onnxruntime" if _HAS_ONNXRUNTIME_TRAINING \
                                    else "ONP"
            cls._optimization_level = GraphOptimizationLevel.ALL
        return cls._instance

    def get_providers(self):
        return self._providers

    def get_available_providers(self):
        return _AVAILABLE_PROVIDERS

    def set_providers(self, *providers):
        with self._lock:
            new_providers = list(providers)
            for provider in self._providers:
                if provider not in new_providers:
                    self._providers.pop(self._providers.index(provider))
                elif provider not in _AVAILABLE_PROVIDERS:
                    warnings.warn(
                        f"Provider {provider} not available."
                        f"Available providers: {''.join(_AVAILABLE_PROVIDERS)}")

    def add_provider(self, provider):
        with self._lock:
            if provider not in self._providers and provider in _AVAILABLE_PROVIDERS:
                self._providers.append(provider)
            elif provider in self._providers:
                warnings.warn(f"Provider {provider} already in provider list")
            elif provider not in _AVAILABLE_PROVIDERS:
                warnings.warn(
                    f"Provider {provider} not available"
                    f"Available providers: {''.join(_AVAILABLE_PROVIDERS)}")

    def remove_provider(self, provider):
        with self._lock:
            if provider in self._providers:
                self._providers.pop(self._providers.index(provider))
            elif provider not in self._providers:
                warnings.warn(f"Provider {provider} not in provider list")

    def onnxruntime_training_available(self):
        return _HAS_ONNXRUNTIME_TRAINING

    def training_backend(self, backend):
        with self._lock:
            if backend not in ["ONP", "onnxruntime"]:
                warnings.warn(
                    "Available training backends are ONP and onnxruntime. "
                    f"{backend} is not known, falling back to ONP backend")
                self._training_backend = "ONP"
            elif backend == "onnxruntime" and not self.onnxruntime_training_available():
                warnings.warn(
                    "Onnxruntime training backend not available, "
                    "falling back to ONP backend")
                self._training_backend = "ONP"
            else:
                self._training_backend = backend

    def set_graph_optimization(self, opt: GraphOptimizationLevel):
        if not isinstance(opt, GraphOptimizationLevel):
            raise TypeError(
                "Expected opt to be of type GraphOptimizationLevel, but "
                f"got {type(opt)}")

        with self._lock:  # type: ignore
            self._optimization_level = opt

    def get_graph_optimization(self):
        return self._optimization_level


def get_ort_graph_optimization_level():
    graph_optimization = Config().get_graph_optimization()
    if graph_optimization == GraphOptimizationLevel.DISABLED:
        return onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if graph_optimization == GraphOptimizationLevel.BASIC:
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if graph_optimization == GraphOptimizationLevel.EXTENDED:
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if graph_optimization == GraphOptimizationLevel.ALL:
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # this should never happen, unless the user did not use a Config method
    # or things have been refactored
    raise InternalException(f"Unknown graph optimization {graph_optimization}")
