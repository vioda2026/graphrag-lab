PYTHON ?= python3
PYTHONPATH := src

.PHONY: setup run-toy lint test check

setup:
	$(PYTHON) -m pip install -e .

run-toy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m graphrag_lab.cli --mode local-debug

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

check: lint test run-toy
