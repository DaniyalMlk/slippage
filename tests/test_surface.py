"""The package's public surface is what its modules declare.

``src/slippage/__init__.py`` re-exports the package's public names, and it is a
hand-maintained list. Seven names were missing from it:

    >>> import slippage
    >>> slippage.to_bps(0.0012)
    AttributeError: module 'slippage' has no attribute 'to_bps'

``costs`` declares ``to_bps``, ``from_bps`` and ``BPS_PER_UNIT`` in its own
``__all__``, and the root imported only the three cost functions beside them.
Every figure this library prints is in basis points, so the conversion a caller
needs to compare one with their own number was the one thing they could not
reach. ``capacity`` declares ``MAX_PERIODS``, ``DEFAULT_CEILING_IN_VOLUME`` and
``BISECTION_STEPS`` and its docstrings refer the reader to
``:data:`MAX_PERIODS``` by name, a cross-reference to something the package
does not offer. ``io`` declares ``read_rows``, the reader its three loaders are
built on.

Nothing noticed, because each module's tests import from the module itself —
the right style for a test, and the one style that cannot see a missing
re-export.

**A module's own ``__all__`` is the authority.** What is asserted is a
disagreement inside the package: a module declaring a name public while the
root does not carry it. Deciding here which names *ought* to be public would
put the judgement somewhere other than beside the code it is about.

**Equality, now that every module can carry it.** It could not be asserted at
first: ``basket`` and ``reversion`` declared no ``__all__``, and eight modules
defined public names their own ``__all__`` omitted — sixteen documented
thresholds and layouts, such as ``calibration.MIN_OBSERVATIONS`` and
``io.ORDER_COLUMNS``, and an array alias, ``FloatArray``, defined separately in
five modules. Declaring the thresholds recorded what their docstrings already
said. The alias is now defined once, in ``types``, and imported by the five.

That makes a further claim available, and it is the one that catches the next
omission rather than this one: **a module's ``__all__`` contains every public
name the module defines.** A name added and not declared fails at the module
instead of travelling to the root for somebody to notice.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

import slippage

#: Neither is part of the importable surface: ``cli`` is an entry point whose
#: names are wired to ``argparse``, and ``__main__`` exists so that
#: ``python -m slippage`` works.
NOT_SURFACE = frozenset({"cli", "__main__"})

#: Modules that declare no ``__all__``. Pinned rather than discovered, so that
#: a new module arriving without one is a decision somebody makes here instead
#: of a silent exemption from every check in this file.
NO_DUNDER_ALL: frozenset[str] = frozenset()

#: Names in the root's ``__all__`` that no module declares. ``__version__`` is
#: assigned in ``__init__.py`` itself, so it is the one legitimate entry.
ROOT_ONLY = frozenset({"__version__"})

#: Source written to exercise :func:`public_names_in`, not to be a module.
PROBE_SOURCE = '''
"""A docstring, which defines nothing."""

from __future__ import annotations

import math                      # an import is not a definition
from typing import Literal

PUBLIC_CONSTANT = 1.0            # a bare float: no __module__ to sweep by
Choice = Literal["a", "b"]       # an alias: likewise
ANNOTATED: int = 2               # an annotated assignment
PAIR_ONE = PAIR_TWO = 3          # one statement, two names
_PRIVATE = 4                     # underscore: not public
__all__ = ["PUBLIC_CONSTANT"]    # the declaration is never a definition


def public_function() -> None:
    """A function."""
    local_name = 5               # a name in a body is not top level
    return None


async def public_coroutine() -> None:
    """An async function, which this package has none of."""


class PublicClass:
    """A class."""

    attribute = 6                # a class attribute is not a module name
'''


def public_names_in(source: str) -> frozenset[str]:
    """The public top-level names a piece of module source defines.

    Separated from the file it usually reads so that it can be tested against
    source written for the purpose.
    """
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        found.update(name for name in names if not name.startswith("_") and name != "__all__")
    return frozenset(found)


def public_definitions(module: str) -> frozenset[str]:
    """Every public name a module defines at its top level, read from source."""
    source = Path(__file__).resolve().parent.parent / "src" / "slippage" / f"{module}.py"
    return public_names_in(source.read_text(encoding="utf-8"))


def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry points."""
    return {
        info.name: importlib.import_module(f"slippage.{info.name}")
        for info in pkgutil.iter_modules(slippage.__path__)
        if info.name not in NOT_SURFACE
    }


def declared() -> dict[str, frozenset[str]]:
    """Each module's own declaration of what it makes public."""
    return {
        name: frozenset(module.__all__)
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is not None
    }


@pytest.fixture(scope="module")
def root() -> frozenset[str]:
    return frozenset(slippage.__all__)


def test_every_declared_name_reaches_the_root(root: frozenset[str]) -> None:
    """A name a module declares public is importable from the package.

    Seven names failed this: three from ``costs``, three from ``capacity`` and
    one from ``io``.
    """
    missing: dict[str, list[str]] = {}
    for module, names in declared().items():
        absent = sorted(names - root)
        if absent:
            missing[module] = absent
    assert not missing, f"declared public but not re-exported from slippage/__init__.py: {missing}"


def test_no_module_is_absent_from_the_root_entirely(root: frozenset[str]) -> None:
    """A module with no name at the root is reported as the module.

    The check above would catch it as loose names, which reads as several small
    omissions rather than one missing import line.
    """
    absent = sorted(module for module, names in declared().items() if names and not (names & root))
    assert not absent, (
        "these modules declare public names and none of them is at the root, so "
        f"the module was never wired into slippage/__init__.py: {absent}"
    )


def test_the_unit_conversions_are_at_the_root(root: frozenset[str]) -> None:
    """The basis-point conversions, named, because every figure needs them.

    Every cost this package reports is in basis points. A caller holding a
    fraction, or a basis-point figure to turn back into one, reaches for these
    first, and the README's first line is ``import slippage``.
    """
    for name in ("to_bps", "from_bps", "BPS_PER_UNIT"):
        assert name in root, f"slippage.{name} is not importable from the package root"


def test_every_exported_name_is_actually_there(root: frozenset[str]) -> None:
    """Nothing in ``__all__`` is a name the package does not have.

    ``__all__`` is a list of strings and nothing checks it against the module,
    so a rename that updates the import and not the list leaves an entry that
    only fails for somebody doing ``from slippage import *``.
    """
    phantom = sorted(name for name in root if not hasattr(slippage, name))
    assert not phantom, f"in slippage.__all__ but not attributes of it: {phantom}"


def test_every_module_declaration_names_real_attributes() -> None:
    """A module's ``__all__`` holds nothing the module lacks."""
    phantom = {
        module: sorted(name for name in names if not hasattr(submodules()[module], name))
        for module, names in declared().items()
    }
    phantom = {module: names for module, names in phantom.items() if names}
    assert not phantom, f"declared in a module's __all__ but not defined there: {phantom}"


def test_the_root_exports_each_name_once() -> None:
    """``__all__`` does not repeat itself."""
    repeated = {name: count for name, count in Counter(slippage.__all__).items() if count > 1}
    assert not repeated, f"repeated in slippage.__all__: {repeated}"


def test_every_module_declares_something() -> None:
    """No module is exempt from the checks above without being named.

    Every other assertion here is silent about a module with no ``__all__``,
    so such a module is exempt from all of them. The exemptions are pinned: a
    new module arriving without one fails here, and so does one of the pinned
    two gaining a declaration without the pin being removed.
    """
    silent = sorted(
        name for name, module in submodules().items() if getattr(module, "__all__", None) is None
    )
    assert silent == sorted(NO_DUNDER_ALL), (
        f"modules carrying no __all__, and so exempt from every check here: "
        f"{silent}; pinned: {sorted(NO_DUNDER_ALL)}"
    )


def test_a_module_declares_every_public_name_it_defines() -> None:
    """``__all__`` is not a curated subset of what the module defines.

    A name added to a module and left out of its ``__all__`` fails here, at the
    module, instead of reaching the root check as a name nobody wired in.
    Twenty-two definitions failed it, and two modules had no declaration at all.
    """
    undeclared: dict[str, list[str]] = {}
    for module, names in declared().items():
        missing = sorted(public_definitions(module) - names)
        if missing:
            undeclared[module] = missing
    assert not undeclared, (
        f"defined at a module's top level and absent from its own __all__: {undeclared}"
    )


def test_no_public_name_is_defined_twice() -> None:
    """One definition per public name across the package.

    ``FloatArray`` was assigned in five modules. The five agreed, so nothing
    was wrong yet, but a flat root namespace can carry only one of them, and
    which one it carried would have been decided by import order.
    """
    owners: dict[str, list[str]] = {}
    for module in submodules():
        for name in public_definitions(module):
            owners.setdefault(name, []).append(module)
    repeated = {name: sorted(mods) for name, mods in owners.items() if len(mods) > 1}
    assert not repeated, f"public names defined in more than one module: {repeated}"


def test_the_root_is_exactly_the_union_of_the_declarations(root: frozenset[str]) -> None:
    """Equality, not containment.

    Containment catches a name the root is missing. This catches the other
    direction — a name the root offers that no module claims — which is what a
    hand-kept list accumulates: an entry left behind by a rename, or one
    exported from the wrong place.
    """
    union = frozenset().union(*declared().values())
    assert root == union | ROOT_ONLY, (
        f"only at the root: {sorted(root - union - ROOT_ONLY)}; "
        f"only in a module: {sorted(union - root)}"
    )


def test_the_probe_reads_a_surface_worth_checking(root: frozenset[str]) -> None:
    """The checks above are not passing because they found nothing.

    Every assertion in this file is satisfied by an empty set, so a probe that
    stops importing modules or stops finding ``__all__`` turns the whole file
    green. The sizes are loose bounds — the claim is that a surface was read at
    all — and ``costs`` is named because it is the module the defect centred on.
    """
    modules = declared()
    assert len(modules) >= 25, f"only {len(modules)} modules declared an __all__"
    assert len(root) >= 210, f"the root surface came back as {len(root)} names"
    assert {"to_bps", "from_bps", "BPS_PER_UNIT"} <= modules["costs"]
    assert "basket_frontier" in modules["basket"]
    assert "price_reversion" in modules["reversion"]


def test_the_probe_reads_each_kind_of_definition_and_skips_each_non_definition() -> None:
    """The probe, against source written for it.

    The exclusions matter as much as the inclusions. A probe that counted
    imports would report every module as failing to declare ``math``; one that
    counted names inside function bodies would report ``local_name``; one that
    missed ``__all__`` itself would have every module failing to declare its
    own declaration. An attribute sweep instead of a source read would miss
    every bare ``float`` constant, which is most of what was missing here.
    """
    found = public_names_in(PROBE_SOURCE)
    assert found == {
        "PUBLIC_CONSTANT",
        "Choice",
        "ANNOTATED",
        "PAIR_ONE",
        "PAIR_TWO",
        "public_function",
        "public_coroutine",
        "PublicClass",
    }, f"the probe read {sorted(found)}"
    for absent in ("math", "Literal", "_PRIVATE", "__all__", "local_name", "attribute"):
        assert absent not in found, f"the probe counted {absent!r} as a definition"
