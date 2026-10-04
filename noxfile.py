"""Nox tasks definition."""
from __future__ import annotations

import shutil
from pathlib import Path

import nox

nox.options.sessions = ["lint", "coverage"]


@nox.session(tags=["annotations"])
def mypy(session: nox.Session) -> None:
    """(annotations) Run mypy."""
    session.install(".[dev]")
    session.install("types-dataclasses")
    session.run("mypy")


@nox.session(python=False, tags=["style", "fix"])
def lint(session: nox.Session) -> None:
    """(quality) Run Black and isort."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.run("pipx", "run", "black", "src", *session.posargs)
    session.run("pipx", "run", "isort", "src", *session.posargs)


@nox.session(python=False, tags=["style"])
def flake(session: nox.Session) -> None:
    """(quality) Run Flake8."""
    session.run("pipx", "run", "pyproject-flake8")


@nox.session(tags=["full fix and style"])
def precommit_checks(session: nox.Session) -> None:
    """(quality) Run all linters. Will be a bit long the first time only."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session
def tests(session: nox.Session) -> None:
    """(test) Run tests."""
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def coverage(session: nox.Session) -> None:
    """(test) Run tests and show coverage."""
    session.posargs.append("--cov=iorec")
    # session.posargs.append("--cov-report")
    # session.posargs.append("html")
    tests(session)


@nox.session(python=False)
def check_manifest(session: nox.Session) -> None:
    """(build) Run check-manifest."""
    session.run("pipx", "run", "check-manifest", "-c", "-v")


DIR = Path(__file__).parent.resolve()


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"])
def build(session: nox.Session) -> None:
    """(build) Build an SDist and wheel, locally."""
    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build")


@nox.session
def publish(session: nox.Session) -> None:
    """Publish the built packages on the koda repository.

    You must have a koda entry in your ~/.pypirc
    with access tokens for the package registry.
    """
    dist_dir = DIR.joinpath("dist")
    if not dist_dir.exists():
        print(dist_dir.resolve(), "not found, have you built the package ?")
    else:
        session.install("twine")
        session.run("twine", "upload", "--repository", "koda", "dist/*")


@nox.session
def lock_file(session: nox.Session) -> None:
    """Generate a locked-requirements.txt file."""
    session.install(".")
    session.install("pipdeptree")
    session.run("sh", "-c", "pipdeptree -f | tee locked-requirements.txt")


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    """Delete cache folders and files."""
    folders = [
        ".nox",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        ".mypy_cache",
    ]
    for folder in folders:
        session.run("rm", "-rf", folder)


@nox.session(python=False)
def install_precommit_hooks(session: nox.Session) -> None:
    """(installation) Install the pre-commit hooks."""
    session.run("pipx", "run", "pre-commit", "install")
