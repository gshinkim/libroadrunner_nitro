def pytest_ignore_collect(collection_path, config):
    """tests/libroadrunner aliases `roadrunner` to roadrunner_nitro: collect it only when asked for by path."""
    if collection_path.name == 'libroadrunner' and collection_path.parent.name == 'tests':
        return not any('libroadrunner' in str(a) for a in config.args)
