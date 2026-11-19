"""The README and the roadmap agree with each other, and the roadmap with itself.

The install section opened with "Not on the package index yet ... until the
first release lands, install from source" for weeks after 0.1.0 was on the
index, and the release was an open item twice in the roadmap: under phase 8,
and again at the end of phase 17, which is about capacity. Both were green the
whole time, because nothing in the suite read either document.

So two kinds of claim are recomputed here:

* whether there has been a release: the roadmap's phase 8 item and the
  README's install section must say the same thing, and since 0.1.0 is on the
  index and a published version cannot be withdrawn, the item stays checked;
* the roadmap's own numbering: no duplicate, no gap, no descent, and every
  cross-reference naming an earlier phase that exists, which is the invariant
  a renumber finished halfway breaks.

The agreement rule on its own could not have caught the original state,
because the two documents were wrong *together*: the box was open and the page
said there was no release, which is consistent. Nothing offline knows what is
on the index, so that state was corrected by hand, and what is pinned from here
is that the release is recorded and the page does not contradict it.

Each rule is a function of text and is tested against text built for it as
well as against the documents, so a rule the documents do not currently
exercise is still shown to fire.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"

HEADING = re.compile(r"^## Phase (\d+) — (.+)$", re.MULTILINE)
REFERENCE = re.compile(r"\b[Pp]hases? (\d+)(?: (and|to) (\d+))?")

#: The roadmap item that records the release, checked or not.
RELEASE_ITEM = re.compile(r"^- \[( |x)\] First release on the index", re.MULTILINE)

#: What the README said while there was no release. Any of them on the page
#: is a claim that the package cannot be installed from the index.
UNRELEASED = re.compile(
    r"Not on the package index yet|until the first release|is what it will be",
    re.IGNORECASE,
)


def section(markdown: str, heading: str) -> str:
    """The body under a ``## `` heading, up to the next ``## `` heading."""
    start = markdown.find(f"\n{heading}\n")
    assert start != -1, f"no {heading!r} section"
    body = start + len(heading) + 2
    following = re.search(r"^## ", markdown[body:], re.MULTILINE)
    return markdown[body : body + following.start()] if following else markdown[body:]


def release_recorded(roadmap: str) -> bool:
    found: list[str] = RELEASE_ITEM.findall(roadmap)
    assert len(found) == 1, f"expected one release item in the roadmap, found {len(found)}"
    return found[0] == "x"


def install_says_unreleased(readme: str) -> bool:
    return UNRELEASED.search(section(readme, "## Install")) is not None


def phase_numbers(roadmap: str) -> list[int]:
    return [int(number) for number, _ in HEADING.findall(roadmap)]


def numbering_faults(numbers: list[int]) -> list[str]:
    faults: list[str] = []
    repeated = sorted(n for n, count in Counter(numbers).items() if count > 1)
    if repeated:
        faults.append(f"repeated {repeated}")
    expected = set(range(1, len(numbers) + 1))
    if set(numbers) != expected:
        faults.append(
            f"missing {sorted(expected - set(numbers))}, "
            f"unexpected {sorted(set(numbers) - expected)}"
        )
    descents = [
        (numbers[i], numbers[i + 1])
        for i in range(len(numbers) - 1)
        if numbers[i + 1] <= numbers[i]
    ]
    if descents:
        faults.append(f"out of order at {descents}")
    return faults


def cross_references(roadmap: str) -> list[tuple[int, int]]:
    """Every prose reference as (phase it is written in, phase it names).

    Only a heading's own number is skipped; a phase its title names is still a
    reference. Text before the first heading is attributed to phase 0.
    """
    bounds = [(match.start(), int(match.group(1))) for match in HEADING.finditer(roadmap)]
    own_numbers = {start + len("## ") for start, _ in bounds}
    found: list[tuple[int, int]] = []
    for match in REFERENCE.finditer(roadmap):
        if match.start() in own_numbers:
            continue
        written_in = 0
        for start, number in bounds:
            if start > match.start():
                break
            written_in = number
        first, joiner, last = int(match.group(1)), match.group(2), match.group(3)
        if joiner == "to":
            named = list(range(first, int(last) + 1))
        elif joiner == "and":
            named = [first, int(last)]
        else:
            named = [first]
        found.extend((written_in, n) for n in named)
    return found


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def test_the_release_stays_recorded(roadmap: str) -> None:
    """0.1.0 is published, and a published version is permanent."""
    assert release_recorded(roadmap), "phase 8's release item was unchecked"


#: An open roadmap item that mentions a release, however it is worded.
OPEN_RELEASE_ITEM = re.compile(r"^- \[ \] .*\brelease\b", re.MULTILINE | re.IGNORECASE)


def test_no_open_item_anywhere_waits_on_a_release(roadmap: str) -> None:
    """The release had a second open copy, worded differently, under phase 17."""
    assert OPEN_RELEASE_ITEM.findall(roadmap) == []


def test_an_open_release_item_is_seen_however_it_is_worded() -> None:
    text = "- [x] Done\n- [ ] A first release on the package index\n"
    assert len(OPEN_RELEASE_ITEM.findall(text)) == 1
    assert OPEN_RELEASE_ITEM.findall("- [x] First release on the index\n") == []


def test_the_two_documents_agree_about_the_release(readme: str, roadmap: str) -> None:
    recorded = release_recorded(roadmap)
    unreleased = install_says_unreleased(readme)
    assert recorded != unreleased, (
        "the roadmap records the release and the install section says there is none"
        if recorded
        else "the install section offers the index and the roadmap records no release"
    )


def test_the_roadmap_numbering_is_sound(roadmap: str) -> None:
    assert numbering_faults(phase_numbers(roadmap)) == []


def test_every_reference_names_an_earlier_phase_that_exists(roadmap: str) -> None:
    numbers = set(phase_numbers(roadmap))
    references = cross_references(roadmap)
    assert sorted({named for _, named in references} - numbers) == []
    assert [(at, named) for at, named in references if named >= at] == []


def test_the_reference_probe_reads_what_is_there(roadmap: str) -> None:
    """Both reference checks pass vacuously if the pattern stops matching."""
    assert sorted(cross_references(roadmap)) == [(20, 2), (20, 13), (28, 27)]


# --- the rules against text built for them -------------------------------- #

UNRELEASED_INSTALL = (
    "# T\n\n## Install\n\n> **Not on the package index yet.** Use git.\n\n## Next\n"
)
RELEASED_INSTALL = "# T\n\n## Install\n\n```bash\npip install x\n```\n\n## Next\n"


@pytest.mark.parametrize(
    ("readme", "expected"),
    [(UNRELEASED_INSTALL, True), (RELEASED_INSTALL, False)],
)
def test_install_says_unreleased_reads_the_section(readme: str, expected: bool) -> None:
    assert install_says_unreleased(readme) is expected


def test_a_claim_outside_the_install_section_is_not_read() -> None:
    text = RELEASED_INSTALL + "\nNot on the package index yet, said the old page.\n"
    assert install_says_unreleased(text) is False


@pytest.mark.parametrize(("box", "expected"), [("x", True), (" ", False)])
def test_release_recorded_reads_the_box(box: str, expected: bool) -> None:
    assert release_recorded(f"## Phase 8 — X\n\n- [{box}] First release on the index\n") is expected


def test_a_missing_or_doubled_release_item_is_an_error() -> None:
    with pytest.raises(AssertionError, match="found 0"):
        release_recorded("## Phase 8 — X\n")
    doubled = "- [x] First release on the index\n- [ ] First release on the index\n"
    with pytest.raises(AssertionError, match="found 2"):
        release_recorded(doubled)


@pytest.mark.parametrize(
    ("numbers", "fragment"),
    [([1, 2, 2, 4], "repeated [2]"), ([1, 2, 4], "missing [3]"), ([1, 3, 2], "out of order")],
)
def test_numbering_faults_fires_on_each_fault(numbers: list[int], fragment: str) -> None:
    assert any(fragment in fault for fault in numbering_faults(numbers))


def test_numbering_faults_is_silent_on_a_clean_list() -> None:
    assert numbering_faults([1, 2, 3]) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Phase 1 — A\n\nSee phase 2.\n\n## Phase 2 — B\n", [(1, 2)]),
        ("## Phase 1 — A\n\n## Phase 2 — B\n\nAs phase 2 says.\n", [(2, 2)]),
        ("Read phase 1 first.\n\n## Phase 1 — A\n", [(0, 1)]),
        ("## Phase 1 — A\n\n## Phase 2 — After phase 1\n", [(2, 1)]),
        ("## Phase 1 — A\n## Phase 2 — B\n## Phase 3 — C\n\nPhases 1 to 2.\n", [(3, 1), (3, 2)]),
    ],
)
def test_cross_references_reads_each_form(text: str, expected: list[tuple[int, int]]) -> None:
    assert cross_references(text) == expected
