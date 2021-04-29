import warnings
from onnxruntime import get_available_providers
try:
    import onnxruntime.training  # noqa
    HAS_ONNXRUNTIME_TRAINING = True
except Exception:
    HAS_ONNXRUNTIME_TRAINING = False


AVAILABLE_PROVIDERS = get_available_providers()
PROVIDERS = list(AVAILABLE_PROVIDERS)


def set_providers(*providers):
    # is this OK to do in Python? What does thread safety look like?
    new_providers = list(providers)
    for provider in PROVIDERS:
        if provider not in new_providers:
            PROVIDERS.pop(PROVIDERS.index(provider))
        elif provider not in AVAILABLE_PROVIDERS:
            warnings.warn(
                f"Provider {provider} not available."
                f"Available providers: {''.join(AVAILABLE_PROVIDERS)}")


def add_provider(provider):
    if provider not in PROVIDERS and provider in AVAILABLE_PROVIDERS:
        PROVIDERS.append(provider)
    elif provider in PROVIDERS:
        warnings.warn(f"Provider {provider} already in provider list")
    elif provider not in AVAILABLE_PROVIDERS:
        warnings.warn(
            f"Provider {provider} not available"
            f"Available providers: {''.join(AVAILABLE_PROVIDERS)}")


def remove_provider(provider):
    if provider in PROVIDERS:
        PROVIDERS.pop(PROVIDERS.index(provider))
    if provider not in PROVIDERS:
        warnings.warn(f"Provider {provider} not in provider list")
