# Every target that produces results goes through pipeline.py, which runs the
# stages in their one order (python pipeline.py stages lists them).
PY := .venv/bin/python
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
MYSQL_TESTS := BNPL_REQUIRE_MYSQL=1 BNPL_DB_DISPOSABLE=1

.PHONY: venv up down schema dev canonical ci-world final docs docs-check llm-history \
	test test-mysql lint check

venv:  ## the locked environment
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements.lock
	.venv/bin/pip install -q --no-deps -e .

up:  ## local MySQL (docker-compose.yml), waiting until it answers
	$(COMPOSE) up -d
	until docker exec bnpl-mysql mysqladmin ping -h 127.0.0.1 -ufraud -pfraudpw --silent; do \
		sleep 2; done

down:
	$(COMPOSE) down

schema:  ## regenerate db/schema.sql from the world contract
	$(PY) db/load_world.py --write-schema

dev:  ## development seeds, every family but the lag sensitivities, into runs/dev (loads the canonical world)
	$(PY) pipeline.py run --profile dev

canonical:  ## the canonical world (seed 416) at full size, loaded into MySQL, into runs/canonical
	$(PY) pipeline.py run --profile canonical

ci-world:  ## one small world end to end, into runs/ci
	$(PY) pipeline.py run --profile ci

final:  ## the final seeds; writes results/ and the documents; refused before the freeze
	$(PY) pipeline.py run --profile final

docs:  ## render the documents from their templates and results/summary.json
	$(PY) -m report render

docs-check:  ## documents in sync with results, claims hold, no typed numbers
	$(PY) -m report check

llm-history:  ## replay the archived LLM benchmark offline and compare its corrected statistics
	$(PY) -m llm.eval.history --check

test:  ## tests; MySQL tests skip without a reachable database
	$(PY) -m pytest -q

test-mysql:  ## every test, failing if MySQL is unreachable; reloads the configured database
	$(MYSQL_TESTS) $(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

check: lint test docs-check
