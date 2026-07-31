PY := .venv/bin/python
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: venv up down generate load rules queue model llm-eval reports cases test lint demo

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"

up:
	$(COMPOSE) up -d
	@echo "waiting for mysql healthcheck..." && sleep 5

down:
	$(COMPOSE) down

generate:
	$(PY) -m simulator.generate

load:
	$(PY) db/load.py

rules:
	$(PY) -m rules.engine
	$(PY) -m rules.tuning

queue:
	$(PY) -m queue_sim.simulate

model:
	$(PY) -m model.train
	$(PY) -m model.evaluate

llm-eval:
	$(PY) -m llm.eval.harness --offline

llm-eval-live:
	$(PY) -m llm.eval.harness

vendor:
	$(PY) -m vendor.enrich
	@echo "synthetic stand-in scores written — re-run 'make rules' to activate R12"

vendor-live:
	$(PY) -m vendor.enrich --live

test:
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check simulator db rules queue_sim model llm vendor tests

demo: up generate load rules model queue llm-eval
	@echo "demo complete — see reports/ and cases/"
