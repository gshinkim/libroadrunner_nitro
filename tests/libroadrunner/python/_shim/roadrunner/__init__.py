"""`import roadrunner` -> roadrunner_nitro, in the test process and in any worker process it starts
(this folder is first on sys.path, which spawned workers inherit). See ../../conftest.py."""
import sys

import roadrunner_nitro

for _name in ('roadrunner', 'roadrunner.roadrunner', 'roadrunner._roadrunner'):
    sys.modules[_name] = roadrunner_nitro
