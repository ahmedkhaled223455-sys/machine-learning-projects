# Empty on purpose: its presence at the project root makes pytest add
# this directory to sys.path when collecting tests/test_api.py, so
# `from src.app import app` resolves whether pytest is invoked from here
# or as part of the repo-wide `pytest` run in CI.
