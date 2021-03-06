"""
    Dummy conftest.py for onnxruntime_numpy.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: integration test"
    )
