# Bandit--JKO experiments.
#
# PY defaults to whatever `python` is on PATH; override if the environment is
# elsewhere, e.g.  make main PY=~/miniconda3/envs/gaeexp/bin/python
PY ?= python
SEEDS ?= 200
SEEDS_D2 ?= 100
T ?=            # optional horizon override, e.g. make main T=300000

.PHONY: help smoke gate main main-d2 dmc dmc-landscape figures overnight clean-results

help:
	@echo "targets:"
	@echo "  smoke           end-to-end check, a few seeds           (~1 min)"
	@echo "  gate            Experiment 3 + the correctness gate     (~15 min)"
	@echo "  main            Experiments 1, 2, 4 for configs B and C"
	@echo "  main-d2         config D (d=2)"
	@echo "  dmc-landscape   cartpole return landscape for the figure"
	@echo "  dmc             cartpole run: Algorithm 1 vs one-point"
	@echo "  figures         analysis table + all figures"
	@echo "  overnight       gate + main + main-d2 + dmc + figures"
	@echo ""
	@echo "variables: PY (interpreter), SEEDS ($(SEEDS)), SEEDS_D2 ($(SEEDS_D2))"
	@echo "sharding:  $(PY) experiments/run_experiments.py main --configs B \\"
	@echo "             --seed-start 0 --seed-end 100 --tag -sh0"

smoke:
	$(PY) experiments/run_experiments.py smoke --seeds 4
	$(PY) experiments/analyze.py

gate:
	$(PY) experiments/run_experiments.py gate --seeds 40 --skip-existing

main:
	$(PY) experiments/run_experiments.py main --configs B C --seeds $(SEEDS) --skip-existing $(if $(T),--T $(T))

main-d2:
	$(PY) experiments/run_experiments.py main --configs D --seeds $(SEEDS_D2) --skip-existing $(if $(T),--T $(T))

dmc-landscape:
	$(PY) experiments/dmc_cartpole.py landscape --grid 21 --episodes 20

dmc:
	$(PY) experiments/dmc_cartpole.py run --T 1200 --seeds 8

figures:
	$(PY) experiments/analyze.py
	$(PY) experiments/make_figures.py

overnight: gate main main-d2 dmc-landscape dmc figures

clean-results:
	rm -f results/*.npz results/summary.json results/figures/*
