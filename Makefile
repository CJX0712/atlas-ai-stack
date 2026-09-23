# ATLAS - developer shortcuts. Every target is a real command, not a note.
#
#   make verify        full chain: P0 scan -> tests -> E2E -> scorecard -> determinism
#   make verify-fast   same chain without the HTTP end-to-end stage
#   make test          unit and integration tests only
#   make e2e           self-contained HTTP end-to-end check
#   make gate          the P0 character scan
#   make run           start the API on 127.0.0.1:8077
#   make doctor        report which optional backends this interpreter has
#   make ask Q="..."   ask one question against the packaged corpus
#   make clean-room    build a fresh venv, install, and verify inside it
#
# Author: 晨星

PY ?= python
Q  ?= 混合检索里为什么不能直接对分数做加权求和？

.PHONY: help verify verify-fast test e2e gate run doctor ask clean-room lock

help:
	@echo "targets: verify verify-fast test e2e gate run doctor ask clean-room lock"

verify:
	$(PY) scripts/verify.py

verify-fast:
	$(PY) scripts/verify.py --skip-e2e

test:
	$(PY) -m pytest tests -q

e2e:
	$(PY) scripts/e2e.py

gate:
	$(PY) tools/scan_emoji.py .

run:
	$(PY) -m atlas serve --host 127.0.0.1 --port 8077

doctor:
	$(PY) -m atlas doctor

ask:
	$(PY) -m atlas ask "$(Q)"

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock.txt
	@echo "wrote requirements.lock.txt"

clean-room:
	$(PY) -m venv .cleanroom
	.cleanroom/bin/python -m pip install --upgrade pip
	.cleanroom/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
	.cleanroom/bin/python scripts/verify.py --skip-e2e
