"""Global pytest configuration.

Defines the ``integration`` marker so pytest doesn't warn about
unrecognised markers, and skips all integration tests unless
``--run-integration`` is passed on the command line.

Usage:
    # Run only unit tests (default):
    pytest -q

    # Run integration tests too (requires DEEPSEEK_API_KEY in .env):
    pytest -q --run-integration
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests that make real external API calls or require local models.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring external resources (skipped by default).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-integration"):
        return  # run everything
    skip_integration = pytest.mark.skip(
        reason="integration test — pass --run-integration to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
