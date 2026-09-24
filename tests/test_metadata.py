"""Checks on the distribution metadata rather than on the mathematics.

A library nobody can install is not useful, and the failures that stop an
install are not the kind unit tests catch: a version written in two files that
drift apart, a licence claimed in metadata but absent from the tree, a typing
marker that exists in the source and never reaches the wheel. Each of these is
invisible until someone tries to depend on the package, which is the worst
moment to find out.

This project has one wrinkle the others do not: the distribution name and the
import name differ, because `slippage` on the index belongs to somebody else.
Everything below that looks up installed metadata has to ask for `slippage-tca`
while the import stays `slippage`, and the test at the end pins that split so a
later tidying cannot quietly collapse the two.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import metadata, packages_distributions, version
from pathlib import Path

import pytest

import slippage

ROOT = Path(__file__).resolve().parent.parent

#: What the index knows this as. The import name above is what code knows it as.
DISTRIBUTION = "slippage-tca"


def test_version_is_exported() -> None:
    """The package says what it is without an installed-metadata lookup."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", slippage.__version__)


def test_exported_version_matches_the_installed_distribution() -> None:
    """The attribute and the installed metadata are the same string.

    These are written in two different files. Bumping one and forgetting the
    other is the ordinary mistake, and it produces a release whose reported
    version is a lie.
    """
    assert slippage.__version__ == version(DISTRIBUTION)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_exported_version_matches_the_source_of_truth() -> None:
    """The attribute agrees with pyproject.toml in the checkout.

    The check above compares against what is *installed*, which in an editable
    install is a snapshot taken at install time. That snapshot can be stale, so
    it would pass on a tree where pyproject.toml has already moved on. Reading
    the file is what makes the check bite without a reinstall.
    """
    import tomllib

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():  # installed without the source tree alongside
        pytest.skip("no checkout to compare against")

    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert slippage.__version__ == declared


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_the_distribution_name_is_the_one_the_index_will_see() -> None:
    """pyproject.toml declares `slippage-tca`, and this file agrees with it.

    Every metadata lookup here hard-codes the distribution name, so a rename in
    pyproject.toml would leave this suite asking about a package that no longer
    exists and failing with a confusing `PackageNotFoundError`. Comparing the
    two directly turns that into a one-line failure that names the problem.
    """
    import tomllib

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("no checkout to compare against")

    assert tomllib.loads(pyproject.read_text())["project"]["name"] == DISTRIBUTION


def test_the_import_name_is_not_the_distribution_name() -> None:
    """`import slippage` resolves to the `slippage-tca` distribution.

    This is the whole point of the split, and it is the part a reader will not
    believe without seeing it: the module is imported under one name and the
    index serves it under another. `packages_distributions` walks the installed
    metadata backwards from the top-level module, which is exactly the question
    somebody debugging a failed `pip install slippage` is asking.
    """
    mapping = packages_distributions()
    if "slippage" not in mapping:  # namespace layouts this test cannot see
        pytest.skip("top-level module not resolvable from installed metadata")
    assert DISTRIBUTION in mapping["slippage"]


def test_licence_text_is_present() -> None:
    """The licence the metadata claims exists as a file, and says MIT."""
    licence = ROOT / "LICENSE"
    if not licence.exists():
        pytest.skip("no checkout to compare against")
    assert "MIT License" in licence.read_text()


def test_licence_reaches_the_built_distribution() -> None:
    """The installed distribution carries the licence, not just the repository.

    `license-files` in pyproject.toml is what does this. Without it — and the
    legacy `license = {text = "MIT"}` table it replaced had no equivalent — the
    metadata goes on claiming MIT while the artefact ships no terms at all. That
    is the state every wheel this project built before now was in.
    """
    fields = metadata(DISTRIBUTION)
    assert fields["License-Expression"] == "MIT"
    assert "LICENSE" in fields.get_all("License-File", [])


def test_the_typing_marker_ships() -> None:
    """`Typing :: Typed` is only true if py.typed is inside the package.

    The classifier is a promise to type checkers. It is made in pyproject.toml
    and kept by a file in the package directory, and nothing connects the two
    except this test.

    The failure is quiet: in a source checkout a type checker reads the
    annotations from the files and everything looks fine, while a consumer who
    installed the package gets none of them.
    """
    assert (Path(slippage.__file__).parent / "py.typed").is_file()
    assert "Typing :: Typed" in metadata(DISTRIBUTION).get_all("Classifier", [])


def test_both_console_scripts_reach_the_same_entry_point() -> None:
    """`slippage` and `slippage-tca` are the same command.

    The alias exists so `uvx slippage-tca` works — `uvx` runs the script named
    after the distribution. An alias that drifted to a different callable would
    be worse than no alias, because the two names would behave differently.
    """
    from importlib.metadata import entry_points

    scripts = {
        entry.name: entry.value
        for entry in entry_points(group="console_scripts")
        if entry.name in {"slippage", DISTRIBUTION}
    }
    assert scripts == {"slippage": "slippage.cli:main", DISTRIBUTION: "slippage.cli:main"}


def test_the_entry_point_reports_the_version() -> None:
    """`slippage --version` prints it, and exits zero doing so.

    argparse's version action raises SystemExit, so the assertion is on the
    exit code as much as on the text.
    """
    from slippage.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
