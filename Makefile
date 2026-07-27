.PHONY: all setup test benchmark report historical clean

# Default python executable in WSL/Linux environments
PYTHON = python3
PIP = pip3

all: test benchmark report

historical:
	$(PYTHON) scripts/historical_audit.py

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[test,waterfall]"

test:
	$(PYTHON) -m pytest tests/ -v

benchmark:
	$(PYTHON) benchmark.py

report:
	$(PYTHON) scripts/generate_reports.py

clean:
	rm -rf __pycache__
	rm -rf axiom/*/__pycache__
	rm -rf axiom/*/*/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf benchmarks/charts benchmarks/reports
