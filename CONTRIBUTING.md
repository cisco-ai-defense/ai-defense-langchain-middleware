# How to Contribute

Thanks for your interest in contributing to `langchain-cisco-aidefense`! This
document outlines the process for contributing and the standards we follow.

Please note that all interactions are subject to the
[Cisco Open Source Code of Conduct](https://github.com/cisco/openSource/blob/master/CODE_OF_CONDUCT.md).

## Table of Contents

- [Reporting Issues](#reporting-issues)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Sending Pull Requests](#sending-pull-requests)

## Reporting Issues

Before reporting a new issue, please search the
[issues list](https://github.com/cisco-ai-defense/langchain-cisco-aidefense/issues)
to check if it has already been reported or fixed.

When creating an issue, please include:
- A clear **title and description**
- Steps to reproduce the problem
- SDK and Python version information
- Any relevant error output or tracebacks

**If you discover a security vulnerability, do not open a public issue.
See [SECURITY.md](/SECURITY.md) for responsible disclosure procedures.**

## Development Setup

### Prerequisites

- Python 3.10 or newer
- [pip](https://pip.pypa.io/) or [uv](https://github.com/astral-sh/uv)

### Getting Started

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/langchain-cisco-aidefense.git
cd langchain-cisco-aidefense

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Project Structure

```
aidefense_langchain/
├── __init__.py                      # Public exports
├── _env.py                          # Env-var parsing and region normalization
├── middleware_chat_client.py         # AIDefenseMiddleware (ChatInspectionClient)
├── middleware_agentsec.py            # AIDefenseAgentsecMiddleware (LLMInspector)
├── middleware_tool_inspection.py     # AIDefenseToolMiddleware (MCPInspectionClient)
└── middleware_tool_agentsec.py       # AIDefenseAgentsecToolMiddleware (MCPInspector)

examples/     # End-to-end usage examples
tests/        # Unit tests for all middleware classes
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for all public function signatures.
- Write docstrings for public classes and methods.
- Keep lines to 100 characters where practical.

## Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=aidefense_langchain --cov-report=term-missing

# Run a specific test file
pytest tests/test_middleware_chat_client.py

# Run tests matching a keyword
pytest -k "test_enforce"
```

- All new features and bug fixes must include tests.
- We use [pytest](https://docs.pytest.org/) with `pytest-asyncio` for async tests.

## Sending Pull Requests

1. Fork the repository and create your branch from `main`.
2. Make your changes, add tests, and ensure the full test suite passes.
3. Update documentation if your change affects the public API.
4. Write a clear PR description explaining the **what** and **why**.

We follow semantic versioning and may reserve breaking changes for major version bumps.
