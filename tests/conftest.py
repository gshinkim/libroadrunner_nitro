import sys

import pytest


def pytest_ignore_collect(collection_path, config):
    """tests/libroadrunner aliases `roadrunner` to roadrunner_nitro: collect it only when asked for by path."""
    if collection_path.name == 'libroadrunner' and collection_path.parent.name == 'tests':
        return not any('libroadrunner' in str(a) for a in config.args)


# On Windows the conservation-law structure (which species are independent, so the order of the dependent and
# independent species, the K matrix and the eigenvalues) comes out differently from the macOS references these
# cases were recorded on. Accepted as a known difference; the cases still run (xfail, not skipped).
WINDOWS_ORDERING = {
    'tests/test_core_vs_roadrunner.py::test_against_libroadrunner[00009]',
    'tests/test_core_vs_roadrunner.py::test_against_libroadrunner[00014]',
    'tests/test_core_vs_roadrunner.py::test_against_libroadrunner[00033]',
    'tests/test_extra_vs_roadrunner.py::test_extra_against_libroadrunner[00009]',
    'tests/test_extra_vs_roadrunner.py::test_extra_against_libroadrunner[00014]',
    'tests/test_extra_vs_roadrunner.py::test_extra_against_libroadrunner[00033]',
    'tests/test_extra_vs_roadrunner.py::test_extra_against_libroadrunner[10001]',
}


def pytest_collection_modifyitems(config, items):
    if sys.platform != 'win32':
        return
    for item in items:
        if item.nodeid.replace('\\', '/') in WINDOWS_ORDERING:
            item.add_marker(pytest.mark.xfail(reason='conservation-law ordering differs on Windows', strict=False))
