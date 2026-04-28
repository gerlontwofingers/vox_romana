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
    # Turn all warnings into errors so silent regressions fail loudly.
    # We use config.addinivalue_line (not warnings.filterwarnings) so that
    # filters apply inside pytest's catch_warnings(record=True) test wrapper.
    # Per pytest docs, later entries take precedence — so "error" goes first
    # (lowest priority) and the "ignore" lines below override it for noise.
    config.addinivalue_line("filterwarnings", "error")
    # ── Third-party noise whitelist ──────────────────────────────────
    # Later entries win → these "ignore" rules override the broad "error" above.
    # pydub: 'audioop' DeprecationWarning (Python 3.13 removal scheduled)
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:pydub")
    # mlx / Apple MLX framework
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:mlx")
    # numpy internal deprecations
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:numpy")
    # huggingface_hub internal deprecations
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:huggingface_hub")
    # SwigPy C-extension: "builtin type SwigPy* has no __module__ attribute"
    # Raised by frozen importlib when loading SWIG-wrapped C extensions.
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:builtin type Swig:DeprecationWarning",
    )
    # pytest's own collection heuristic fires on `TestResult` (imported from
    # vox_romana.analysis.models) because the class name starts with "Test".
    # This is a pytest naming collision, not a bug in our code.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pytest.PytestCollectionWarning",
    )

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
