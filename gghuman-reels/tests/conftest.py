import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run the integration tests that render real video (~3 min)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow integration test: use --runslow")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: renders real video")
