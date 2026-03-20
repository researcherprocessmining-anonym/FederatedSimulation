.PHONY: help install download-data test reproduce clean

help:
	@echo "Federated SFD — Makefile targets"
	@echo "  install        Install project and dependencies"
	@echo "  download-data  Download OCEL 2.0 Logistics dataset (or generate synthetic)"
	@echo "  test           Run all tests"
	@echo "  reproduce      Run all paper experiments"
	@echo "  clean          Remove results and generated data"

install:
	pip install -e ".[dev]"

download-data:
	bash data/download.sh

test:
	python -m pytest tests/ -v

reproduce:
	python experiments/reproduce.py

clean:
	rm -f results/*.csv results/*.mdl results/*.txt
	rm -rf results/plots
	rm -f data/ocel2-logistics.sqlite
