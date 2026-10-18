"""The figures on the front page are the figures the code produces.

The README opens with a table of results, two of them checked against
enumeration rather than against a formula. A table like that is worth exactly
as much as the guarantee that it still matches, and a number copied into a
document once stops being true quietly: the library changes, the suite stays
green because nothing in it reads the document, and the front page goes on
claiming an accuracy the code no longer has.

So the claim is pinned here in three directions. Every row the script renders
has to be on the front page as rendered; every row the front page quotes has to
be one the script renders; and the number each row *claims* has to be the
number the check actually enforces, which is the direction nothing else covers.

This test is also what makes "runs in CI" true. CI runs the suite, so the
evidence script runs here rather than in a workflow step of its own.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EVIDENCE = ROOT / "examples" / "evidence.py"

#: Row labels of the README's evidence table, as the script renders them.
#: Written out rather than parsed from the README so that deleting a row from
#: the table does not silently delete the check that defends it.
ROWS = (
    "The schedule solver against exhaustive enumeration of all 1,820 schedules, 12 cases",
    "The router against exhaustive search, 250 random problems",
    "A 2,000-lot grid against the Almgren-Chriss closed form",
    "The same grid's objective against the continuum's",
    "The response kernel at zero urgency, against a tent",
    "Implementation shortfall against the same sum done by hand",
)

#: The bound each row enforces. Pinned here as well as read off the script, so
#: that all three of bound, rendering and document have to agree.
BOUNDS = {
    "The schedule solver against exhaustive enumeration of all 1,820 schedules, 12 cases": 1e-14,
    "The router against exhaustive search, 250 random problems": 0.0,
    "A 2,000-lot grid against the Almgren-Chriss closed form": 1.0,
    "The same grid's objective against the continuum's": 0.0,
    "The response kernel at zero urgency, against a tent": 1e-14,
    "Implementation shortfall against the same sum done by hand": 0.0,
}


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    """Run the evidence script once and share its figures across the tests."""
    with redirect_stdout(StringIO()):
        result = runpy.run_path(str(EVIDENCE))["main"]()
    assert isinstance(result, dict)
    return result


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str, readme: str) -> None:
    assert row in readme, f"the evidence table no longer has: {row}"


def test_every_row_the_script_renders_is_on_the_front_page(
    evidence: dict[str, Any], readme: str
) -> None:
    """Label and result together, as one line.

    Checking the pair rather than each half is what makes this bite: a figure
    that moves changes the result while the label stays put, and a test looking
    only for labels would not notice.
    """
    rendered = evidence["rendered"]
    assert isinstance(rendered, dict)
    assert set(rendered) == set(ROWS), "the script's checks and the pinned rows disagree"
    for label, result in rendered.items():
        assert f"| {label} | {result} |" in readme, f"not on the front page: {label}"


@pytest.mark.parametrize("label", ROWS)
def test_the_measurement_is_inside_the_bound_the_script_enforces(
    label: str, evidence: dict[str, Any]
) -> None:
    checks = evidence["checks"]
    enforced = evidence["bounds"]
    assert checks[label] <= enforced[label], (
        f"{label} measured {checks[label]}, bound {enforced[label]}"
    )
    assert enforced[label] == BOUNDS[label], (
        f"{label} enforces {enforced[label]}, pinned here as {BOUNDS[label]}"
    )


def test_the_grid_objective_is_above_the_continuum_and_the_readme_says_which_way(
    evidence: dict[str, Any], readme: str
) -> None:
    """The inequality, not the gap.

    A lot grid is a subset of the continuum, so the best schedule on the grid
    cannot cost less than the best schedule in it. The failure worth catching
    is a programme that reports a *lower* objective, which a tolerance on the
    unsigned difference would accept as a very good agreement — so the
    direction is asserted on its own, and the two figures the README prints are
    checked to be the two the script measured.
    """
    grid = evidence["grid_objective"]
    continuum = evidence["continuum_objective"]
    assert isinstance(grid, float)
    assert isinstance(continuum, float)
    assert grid >= continuum, (
        f"the grid objective {grid} is below the continuum's {continuum}, "
        "which is not possible for a subset and means something is wrong"
    )
    # Close, though: a grid that beat the continuum by a mile would pass the
    # inequality above while meaning the two were not solving the same problem.
    assert (grid - continuum) / continuum < 1e-6
    assert f"{grid:.6f}" in readme and f"{continuum:.6f}" in readme


def test_the_kernel_is_a_tent_at_zero_urgency_and_is_not_at_a_positive_one(
    evidence: dict[str, Any],
) -> None:
    """Both halves, because either alone says nothing.

    A vanishing second difference at zero urgency is the tent. A vanishing
    second difference at *every* urgency would mean the kernel had been
    linearised somewhere and the shape was an assumption rather than a result.
    The gap between the two figures is therefore the claim, and it is asserted
    as a gap of at least ten orders of magnitude rather than as two separate
    tolerances.
    """
    flat, urgent = evidence["curvature"]
    assert flat < 1e-14, f"the zero-urgency kernel is not a tent: {flat}"
    assert urgent > 1e-4, f"the kernel is a tent at a positive urgency too: {urgent}"
    assert urgent / flat > 1e10


def test_the_enumerations_are_the_size_the_front_page_claims(
    evidence: dict[str, Any], readme: str
) -> None:
    """The counts, which are the strength of the two best rows.

    "Checked against enumeration" is only as strong as the set enumerated, so
    the set's size is part of the claim rather than decoration, and a quiet
    reduction of it would weaken the front page without changing a figure.
    """
    schedules = evidence["schedules"]
    cases = evidence["cases"]
    checked = evidence["checked"]
    optimal = evidence["optimal"]
    assert schedules == 1820, f"the schedule enumeration is now {schedules} wide"
    assert cases == 12
    assert checked >= 200, f"only {checked} routing problems were enumerable"
    assert optimal == checked, f"{checked - optimal} routing problems were not optimal"
    assert f"all {schedules:,} schedules" in readme
    assert f"{checked} random problems" in readme


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. A quarter either way
    still catches the README claiming hundreds when there are thousands.
    """
    claimed = re.search(r"([\d,]+) tests", README.read_text(encoding="utf-8"))
    assert claimed is not None, "the README no longer states a suite size"
    stated = int(claimed.group(1).replace(",", ""))
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-o", "addopts="],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    found = re.search(r"(\d+) tests collected", collected.stdout)
    assert found is not None, collected.stdout[-500:]
    actual = int(found.group(1))
    assert 0.75 * actual <= stated <= 1.25 * actual, (
        f"the README says {stated} tests and there are {actual}"
    )
