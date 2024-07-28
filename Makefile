#
# Installation
#
.PHONY: setup
setup:
	pip install -U --no-cache-dir pip setuptools wheel poetry

.PHONY: install
install:
	poetry install

#
# linter/formatter/typecheck/testing
#

.PHONY: lint
lint: install
	poetry run ruff check --output-format=github .

.PHONY: format
format: install
	poetry run ruff format --check --diff .

.PHONY: typecheck
typecheck: install
	poetry run mypy --cache-dir=/dev/null .
