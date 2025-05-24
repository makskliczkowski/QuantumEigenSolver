# QES Development Makefile

.PHONY: help install install-dev test test-all lint format type-check docs clean build upload

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-all     - Run tests with tox"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  docs         - Build documentation"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  upload       - Upload to PyPI (test)"
	@echo "  help         - Show this help"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ general_python/tests/

test-all:
	tox

# Code quality
lint:
	flake8 QES

format:
	black QES
	isort QES

type-check:
	mypy QES

# Documentation
docs:
	sphinx-build -W -b html docs docs/_build/html

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .tox/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building and distribution
build: clean
	python -m build

upload: build
	python -m twine upload --repository testpypi dist/*

# Development workflow
dev-setup: install-dev
	@echo "Development environment set up successfully!"
	@echo "Run 'make test' to verify installation"

# Check everything before commit
check: format lint type-check test
	@echo "All checks passed!"

# Quick development cycle
dev: format lint test
	@echo "Development cycle complete!"
