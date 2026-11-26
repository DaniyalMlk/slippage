"""A caller who can call something can name what it takes and returns.

``tests/test_surface.py`` asks whether every name a module declares reaches the
root, and whether any public name is defined twice. Neither can see a type the
root does not bind because nothing *declares* it: a private class in a public
signature, or an alias a field is written in that only its own module carries.
The sibling packages each had one. ``moneyness.adaptive_quad`` took a
``budget: _Budget | None`` that only a private import could fill, and two of
``holdout``'s public classes declared their fields with a copy of
``IndexArray`` that the root did not bind. Nothing here has either today; this
keeps it so.

For everything reachable from the root (each root function, each public method
and property of each root class including those inherited from a package
class, and each field of a root class) every package class its annotations
mention, and every package type alias they are written in, must be bound at the
root.

Three things make the obvious version of this wrong:

* **Identity, not name.** The root binding *something* called ``FloatArray``
  says nothing about whether it binds the one a signature means.
* **An alias is invisible once resolved.** ``typing.get_type_hints`` expands
  ``FloatArray`` into ``numpy.ndarray[...]`` before anything can look at it, so
  a check built on resolved hints passes every alias whatever the root does.
  Aliases are read from source.
* **A spelled name means what its own module binds.** Each name an annotation
  spells is resolved in the globals of the module that wrote it, rather than
  matched against alias names.

``typing.get_args(Callable[[A], B])`` is ``([A], B)``: a callback's parameters
arrive as a plain list, which a walk recursing only through ``get_args`` never
enters. The class walk enters lists and tuples, and a rule test pins it.

A parameter named with an underscore is private by name and exempt, as is a
member named with one, except ``__init__``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import textwrap
import typing
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import slippage

PACKAGE = Path(slippage.__file__).resolve().parent
PREFIX = "slippage."

#: The command line is an entry point wired to ``argparse``, not to callers.
NOT_SURFACE = frozenset({"cli", "__main__"})

#: Subscripted names that make a module-level assignment a type expression.
TYPE_CONSTRUCTORS = frozenset(
    {
        "Annotated",
        "Callable",
        "Literal",
        "Mapping",
        "NDArray",
        "Optional",
        "Sequence",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)


# ---------------------------------------------------------------------------
# The probe.


def package_classes_in(hint: object, prefix: str = PREFIX) -> set[type]:
    """Every class from the package mentioned anywhere inside a resolved hint."""
    if isinstance(hint, list | tuple):
        return set().union(*(package_classes_in(item, prefix) for item in hint))
    found: set[type] = set()
    if isinstance(hint, type) and hint.__module__.startswith(prefix):
        found.add(hint)
    origin = typing.get_origin(hint)
    if isinstance(origin, type) and origin.__module__.startswith(prefix):
        found.add(origin)
    for argument in typing.get_args(hint):
        found |= package_classes_in(argument, prefix)
    return found


def _subscripted_name(node: ast.expr) -> str | None:
    """``list`` for ``list[...]``, ``Callable`` for ``abc.Callable[...]``."""
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _is_type_expression(node: ast.expr) -> bool:
    """Whether a right-hand side reads as a type rather than a value."""
    if _subscripted_name(node) in TYPE_CONSTRUCTORS:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # ``int | str`` and ``float | None`` are types; ``1 | 2`` is a number.
        return all(_is_type_operand(side) for side in (node.left, node.right))
    return False


def _is_type_operand(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    return isinstance(node, ast.Name | ast.Attribute) or _is_type_expression(node)


def aliases_in(source: str) -> frozenset[str]:
    """The module-level names a source file binds to a type expression.

    ``X: TypeAlias = ...`` and a ``type X = ...`` statement always count. A
    plain assignment counts when its value is a subscripted type constructor
    or a ``|`` union. A number, a string, a call, or a bare name does not.
    """
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if "TypeAlias" in ast.unparse(node.annotation):
                found.add(node.target.id)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and _is_type_expression(node.value):
                found.add(target.id)
        elif type(node).__name__ == "TypeAlias":  # ``type X = ...``, 3.12 and later
            found.add(node.name.id)  # type: ignore[attr-defined]
    return frozenset(found)


def names_in_annotation(node: ast.expr) -> frozenset[str]:
    """Every bare name an annotation spells, including inside a string."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                inner = ast.parse(child.value, mode="eval")
            except SyntaxError:
                continue
            found |= names_in_annotation(inner.body)
    return frozenset(found)


def public_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr]:
    """The annotations of a function's public parameters and its return."""
    arguments = function.args
    parameters = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
    ]
    found = [
        parameter.annotation
        for parameter in parameters
        if parameter.annotation is not None and not parameter.arg.startswith("_")
    ]
    if function.returns is not None:
        found.append(function.returns)
    return found


def field_annotations(cls: ast.ClassDef) -> list[ast.expr]:
    """The annotations of a class body's public fields."""
    return [
        node.annotation
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and not node.target.id.startswith("_")
    ]


def aliases_spelled(
    annotations: list[ast.expr],
    scope: Mapping[str, object],
    aliases: Mapping[int, str],
) -> dict[str, object]:
    """The package aliases some annotations spell, resolved where they were written.

    ``scope`` is the namespace of the module that wrote the annotations and
    ``aliases`` maps the identity of each package alias to its qualified name.
    A name the scope binds to anything that is not a package alias, a class
    or a builtin or nothing at all, contributes nothing.
    """
    found: dict[str, object] = {}
    for annotation in annotations:
        for name in names_in_annotation(annotation):
            if name not in scope:
                continue
            value = scope[name]
            qualified = aliases.get(id(value))
            if qualified is not None:
                found[qualified] = value
    return found


# ---------------------------------------------------------------------------
# What the probe is pointed at.


def package_aliases() -> dict[int, str]:
    """Each public package alias, keyed by the identity of the object it binds."""
    found: dict[int, str] = {}
    for info in pkgutil.iter_modules([str(PACKAGE)]):
        if info.name in NOT_SURFACE:
            continue
        module = importlib.import_module(f"{PREFIX}{info.name}")
        source = (PACKAGE / f"{info.name}.py").read_text(encoding="utf-8")
        for name in aliases_in(source):
            if not name.startswith("_"):
                found[id(getattr(module, name))] = f"{module.__name__}.{name}"
    return found


def root_objects() -> dict[str, object]:
    return {name: getattr(slippage, name) for name in slippage.__all__}


def reachable_callables() -> Iterator[tuple[str, Callable[..., object]]]:
    """Each root function, and each public method or property of each root class."""
    for name, obj in sorted(root_objects().items()):
        if inspect.isfunction(obj):
            yield name, obj
        elif inspect.isclass(obj) and obj.__module__.startswith(PREFIX):
            seen: set[str] = set()
            for klass in obj.__mro__:
                if not klass.__module__.startswith(PREFIX):
                    continue
                for attribute, value in vars(klass).items():
                    if attribute in seen:
                        continue
                    seen.add(attribute)
                    if attribute.startswith("_") and attribute != "__init__":
                        continue
                    member = value
                    if isinstance(member, classmethod | staticmethod):
                        member = member.__func__
                    if isinstance(member, property):
                        member = member.fget
                    if inspect.isfunction(member):
                        yield f"{name}.{attribute}", member


def reachable_classes() -> Iterator[tuple[str, type]]:
    for name, obj in sorted(root_objects().items()):
        if inspect.isclass(obj) and obj.__module__.startswith(PREFIX):
            yield name, obj


def _function_node(
    function: Callable[..., object],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    # A dataclass's generated ``__init__`` has no source to read; its fields
    # are covered through the class body instead.
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    node = ast.parse(source).body[0]
    return node if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) else None


def unreachable_classes() -> dict[str, list[str]]:
    """Reachable signatures and fields written in a class the root lacks."""
    bound = {id(obj) for obj in root_objects().values()}
    missing: dict[str, set[str]] = {}

    def note(label: str, hints: list[object]) -> None:
        for kind in set().union(*(package_classes_in(hint) for hint in hints)):
            if id(kind) not in bound:
                missing.setdefault(f"{kind.__module__}.{kind.__qualname__}", set()).add(label)

    for label, function in reachable_callables():
        hints = typing.get_type_hints(function)
        note(label, [hint for name, hint in hints.items() if not name.startswith("_")])
    for name, cls in reachable_classes():
        hints = typing.get_type_hints(cls)
        note(f"{name} (fields)", [h for f, h in hints.items() if not f.startswith("_")])
    return {kind: sorted(labels) for kind, labels in sorted(missing.items())}


def unreachable_aliases() -> dict[str, list[str]]:
    """Reachable signatures and fields spelled with an alias the root lacks."""
    aliases = package_aliases()
    bound = {id(obj) for obj in root_objects().values()}
    missing: dict[str, set[str]] = {}

    def note(label: str, annotations: list[ast.expr], scope: Mapping[str, object]) -> None:
        for qualified, value in aliases_spelled(annotations, scope, aliases).items():
            if id(value) not in bound:
                missing.setdefault(qualified, set()).add(label)

    for label, function in reachable_callables():
        node = _function_node(function)
        if node is not None:
            note(label, public_annotations(node), function.__globals__)
    for name, cls in reachable_classes():
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            continue
        statement = ast.parse(source).body[0]
        if isinstance(statement, ast.ClassDef):
            scope = vars(sys.modules[cls.__module__])
            note(f"{name} (fields)", field_annotations(statement), scope)
    return {alias: sorted(labels) for alias, labels in sorted(missing.items())}


# ---------------------------------------------------------------------------
# The checks.


def test_the_survey_is_worth_checking() -> None:
    # Every check below passes trivially over an empty walk, so the walk is
    # asserted first. The floors sit under what is there (236 root names, 387
    # callables and 91 classes when this was written) so that adding a module
    # does not move them.
    assert len(slippage.__all__) > 200
    assert sum(1 for _ in reachable_callables()) > 330
    assert sum(1 for _ in reachable_classes()) > 75
    found = set(package_aliases().values())
    assert {"slippage.types.FloatArray", "slippage.types.Floats"} <= found


def test_a_class_in_a_reachable_signature_is_the_one_at_the_root() -> None:
    unreachable = unreachable_classes()
    assert not unreachable, (
        f"reachable from slippage, written in classes the root does not bind: {unreachable}"
    )


def test_an_alias_a_reachable_signature_is_written_in_is_at_the_root() -> None:
    unreachable = unreachable_aliases()
    assert not unreachable, (
        f"reachable from slippage, written in type aliases the root does not bind: {unreachable}"
    )


def test_the_alias_survey_resolves_real_spellings() -> None:
    # A survey that resolved nothing would let the alias check pass vacuously,
    # so both aliases are shown to be spelled in reachable signatures.
    aliases = package_aliases()
    spelled: set[str] = set()
    for _, function in reachable_callables():
        node = _function_node(function)
        if node is not None:
            spelled |= set(aliases_spelled(public_annotations(node), function.__globals__, aliases))
    assert {"slippage.types.FloatArray", "slippage.types.Floats"} <= spelled


# ---------------------------------------------------------------------------
# The probe's own rules, against inputs written for each, since the package
# does not exercise every branch.


class _Local:
    pass


def test_classes_are_found_inside_every_kind_of_hint() -> None:
    prefix = _Local.__module__
    hints: list[object] = [
        _Local,
        list[_Local],
        _Local | None,
        dict[str, tuple[int, _Local]],
        Callable[[_Local], float],
        type[_Local],
    ]
    for hint in hints:
        assert package_classes_in(hint, prefix) == {_Local}, hint


def test_a_class_only_in_a_callbacks_parameters_is_found() -> None:
    prefix = _Local.__module__
    assert package_classes_in(Callable[[int, _Local], None], prefix) == {_Local}
    assert package_classes_in(Callable[[Callable[[_Local], float]], float], prefix) == {_Local}


def test_classes_outside_the_package_are_not_found() -> None:
    assert package_classes_in(list[int] | dict[str, float] | None) == set()
    assert package_classes_in(Path) == set()


def test_aliases_are_the_assignments_that_read_as_types() -> None:
    source = textwrap.dedent(
        """
        from typing import Literal, TypeAlias
        from collections import abc
        Vector = list[float]
        Pair = tuple[float, float]
        Side = Literal["call", "put"]
        Shape = abc.Callable[[float], float]
        Either = int | str
        Maybe = float | None
        Explicit: TypeAlias = "dict[str, float]"
        """
    )
    assert aliases_in(source) == {"Vector", "Pair", "Side", "Shape", "Either", "Maybe", "Explicit"}


def test_values_are_not_aliases() -> None:
    source = textwrap.dedent(
        """
        TOLERANCE = 1e-12
        NAME = "slippage"
        DEFAULT = make_default()
        Same = Vector
        FLAGS = 1 | 2
        TABLE = {"a": 1}
        CHOICES = ("a", "b")
        LOOKUP = mapping["key"]
        count: int = 3
        """
    )
    assert aliases_in(source) == set()


def test_an_alias_inside_a_function_is_not_module_level() -> None:
    assert aliases_in("def f():\n    Local = list[float]\n") == set()


def test_annotation_names_include_those_inside_a_string() -> None:
    node = ast.parse("x: 'Vector | None'").body[0]
    assert isinstance(node, ast.AnnAssign)
    assert names_in_annotation(node.annotation) == {"Vector"}


def test_an_unparseable_string_annotation_contributes_nothing() -> None:
    node = ast.parse("x: 'not a [type'").body[0]
    assert isinstance(node, ast.AnnAssign)
    assert names_in_annotation(node.annotation) == set()


def test_public_annotations_skip_private_parameters_and_keep_the_return() -> None:
    node = ast.parse(
        "def f(a: A, /, b: B, *args: C, _c: Hidden, d: D = 1, **kw: E) -> R: ..."
    ).body[0]
    assert isinstance(node, ast.FunctionDef)
    spelled = {ast.unparse(annotation) for annotation in public_annotations(node)}
    assert spelled == {"A", "B", "C", "D", "E", "R"}


def test_field_annotations_skip_private_fields_and_methods() -> None:
    node = ast.parse(
        "class C:\n    a: A\n    _b: Hidden\n    c = 1\n    def m(self) -> R: ...\n"
    ).body[0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(a) for a in field_annotations(node)] == ["A"]


def test_a_spelled_name_resolves_in_the_scope_that_wrote_it() -> None:
    """The rule the name-matching probe lacked.

    Two scopes spell ``Method``. In one it is a package alias, in the other
    an unrelated class. Only the first is reported, and it is reported under
    the alias's own qualified name rather than the spelling.
    """
    alias = Callable[[float], float]
    aliases = {id(alias): "pkg.risk.Shape"}
    node = ast.parse("def f(w: Method) -> 'list[Method]': ...").body[0]
    assert isinstance(node, ast.FunctionDef)
    annotations = public_annotations(node)
    assert aliases_spelled(annotations, {"Method": alias}, aliases) == {"pkg.risk.Shape": alias}
    assert aliases_spelled(annotations, {"Method": _Local}, aliases) == {}


def test_a_name_the_scope_does_not_bind_contributes_nothing() -> None:
    alias = Callable[[float], float]
    node = ast.parse("def f(w: Shape) -> None: ...").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert aliases_spelled(public_annotations(node), {}, {id(alias): "pkg.Shape"}) == {}
