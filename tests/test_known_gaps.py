"""The README's known-gaps list defends itself.

A list of shortcomings is the first thing in a document to go stale, because
closing a gap is satisfying and deleting the line that described it is not. So
each line carries a predicate, and the predicate holds only while the gap is
real: implement the feature and its line fails, forcing the line out.

Two rules about what belongs on that list, both learned the hard way:

**Absences, not refusals.** Most of what this library declines to do it
declines deliberately — VWAP over an empty interval raises, a final-period
drift is accepted and ignored — and those are documented as behaviour where
they belong. A list of deliberate refusals relabelled as gaps would read as
candour and be worth less than no list at all.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under a different one, and is useless for a gap of the form "X cannot
be written down". Each predicate below either calls an entry point and observes
what happens, or asserts the shape of a public type — never greps the source.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

import slippage
from slippage import (
    Bar,
    Benchmark,
    Fill,
    Order,
    Placement,
    Side,
    build_report,
    implementation_shortfall,
)
from slippage.exceptions import ValidationError

README = Path(__file__).resolve().parent.parent / "README.md"

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Layout"


def field_names(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(cls))


def an_order(**kwargs: object) -> Order:
    """A one-fill order, good enough to reach any measurement entry point."""
    defaults: dict[str, object] = {
        "symbol": "AAA",
        "side": Side.BUY,
        "quantity": 100.0,
        "decision_time": datetime(2024, 1, 2, 9, 30),
        "arrival_time": datetime(2024, 1, 2, 9, 31),
        "decision_price": 10.0,
        "fills": (
            Fill(timestamp=datetime(2024, 1, 2, 9, 32), quantity=100.0, price=10.0, venue="X"),
        ),
    }
    defaults.update(kwargs)
    return Order(**defaults)  # type: ignore[arg-type]


# --- one predicate per line of the list ----------------------------------- #


def no_rebate_can_be_recorded() -> None:
    """Placement prices a maker rebate; nothing downstream will accept one."""
    assert "maker_rebate" in field_names(Placement)
    priced = Placement(
        distance=0.01,
        horizon=60.0,
        volatility=0.2,
        drift=0.0,
        half_spread=0.005,
        taker_fee=0.0002,
        maker_rebate=0.0001,
    )
    assert priced.maker_rebate > 0.0

    with pytest.raises(ValidationError):
        Fill(timestamp=datetime(2024, 1, 2, 9, 32), quantity=100.0, price=10.0, commission=-0.5)
    with pytest.raises(ValidationError):
        implementation_shortfall(an_order(), arrival_price=10.0, final_price=10.0, fees=-1.0)
    with pytest.raises(ValidationError):
        build_report({"AAA": an_order()}, {}, fees_per_share=-0.001)

    # The same calls with a non-negative figure go through, so the refusal is
    # about the sign and not about the arguments being wrong.
    assert implementation_shortfall(
        an_order(), arrival_price=10.0, final_price=10.0, fees=1.0
    ).explicit == pytest.approx(1.0, abs=0.0)


def a_bar_carries_no_vwap() -> None:
    """The interval VWAP is reconstructed, because the input cannot carry one."""
    assert field_names(Bar) == frozenset({"timestamp", "open", "high", "low", "close", "volume"})
    bar = Bar(
        timestamp=datetime(2024, 1, 2, 9, 30),
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=1000.0,
    )
    assert not hasattr(bar, "vwap")
    with pytest.raises(TypeError):
        Bar(  # type: ignore[call-arg]
            timestamp=datetime(2024, 1, 2, 9, 30),
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            volume=1000.0,
            vwap=10.2,
        )


def there_is_no_auction_price() -> None:
    """OPEN and CLOSE are a bar's prices, so no benchmark names an auction."""
    names = {member.name for member in Benchmark}
    assert names == {
        "DECISION",
        "ARRIVAL",
        "INTERVAL_VWAP",
        "INTERVAL_VWAP_EX_OWN",
        "INTERVAL_TWAP",
        "CLOSE",
        "OPEN",
    }
    assert not any("AUCTION" in name for name in names)


def there_is_one_currency() -> None:
    """Nothing records a currency, and nothing accepts a rate to convert one."""
    assert not {name for name in field_names(Order) | field_names(Fill) if "curren" in name}
    report_parameters = set(inspect.signature(build_report).parameters)
    assert not {name for name in report_parameters if "rate" in name or "curren" in name}


def a_fee_schedule_is_one_number() -> None:
    """A fill knows its venue; the fee applied to it does not."""
    assert "venue" in field_names(Fill)
    fees = inspect.signature(build_report).parameters["fees_per_share"]
    assert fees.annotation == "float"
    assert fees.default == 0.0


def there_is_no_quote_or_tick_data() -> None:
    """Every estimator reads bars, so a quoted spread has nowhere to go."""
    assert {member.name for member in slippage.Estimator} == {
        "ROLL",
        "CORWIN_SCHULTZ",
        "ABDI_RANALDO",
    }
    bars = inspect.signature(slippage.estimate_spread).parameters["bars"]
    assert bars.annotation == "Sequence[Bar]"


def fill_probability_is_a_barrier_not_a_queue() -> None:
    """Third in the queue and first in it are the same order to Placement."""
    assert field_names(Placement) == frozenset(
        {"distance", "horizon", "volatility", "drift", "half_spread", "taker_fee", "maker_rebate"}
    )
    assert not {name for name in field_names(Placement) if "queue" in name or "depth" in name}
    simulate = set(inspect.signature(slippage.simulate_placement).parameters)
    assert not {name for name in simulate if "queue" in name or "depth" in name}


#: Each line of the README's list, paired with the predicate that keeps it
#: true. The text is the line's leading sentence, bold in the document, which is
#: what a reader scanning the list actually takes in.
GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("A maker rebate cannot be recorded.", no_rebate_can_be_recorded),
    ("A bar carries no VWAP.", a_bar_carries_no_vwap),
    ("There is no auction price.", there_is_no_auction_price),
    ("One currency.", there_is_one_currency),
    ("A fee schedule is one number.", a_fee_schedule_is_one_number),
    ("No quote or tick data.", there_is_no_quote_or_tick_data),
    (
        "Fill probability is a barrier model, not a queue.",
        fill_probability_is_a_barrier_not_a_queue,
    ),
)


def gaps_section() -> str:
    """The list as the document holds it, stopping at the next heading.

    Reading to the end of the file instead would let a bullet anywhere below
    satisfy a membership check, which is how a gate comes to be defended by
    prose that has nothing to do with it.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index(NEXT_SECTION, start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The leading bold sentence of each bullet in the section."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.M)


@pytest.mark.parametrize("line, predicate", GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    """Close the gap and this fails, which is how the line gets deleted."""
    predicate()


@pytest.mark.parametrize("line", [line for line, _ in GAPS])
def test_the_line_is_still_on_the_page(line: str) -> None:
    """A predicate defending a line nobody can read defends nothing."""
    assert line in rendered_bullets(), f"not a bullet under {SECTION!r}: {line!r}"


def test_the_page_lists_nothing_that_has_no_predicate() -> None:
    """A gap added to the document without a predicate is the failure mode."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    """One predicate copied onto two lines would pass for the wrong reason."""
    predicates = [predicate for _, predicate in GAPS]
    assert len({p.__name__ for p in predicates}) == len(predicates)


def test_the_section_ends_at_the_heading_that_follows_it() -> None:
    """The boundary has to be the next heading, or it bounds nothing.

    ``NEXT_SECTION not in gaps_section()`` looks like it checks this and is a
    tautology: whatever the terminator is, slicing up to it excludes it. The
    claim worth making is that the terminator is the heading immediately after
    the list, so widening it to some later heading — which would let a bullet
    written anywhere in between satisfy the membership checks above — fails.
    """
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.M)
    assert SECTION in headings
    following = headings[headings.index(SECTION) + 1]
    assert following == NEXT_SECTION, (
        f"the list is bounded by {NEXT_SECTION!r} and is followed by {following!r}"
    )
    section = gaps_section()
    assert section.startswith(SECTION)
    assert len(rendered_bullets()) == len(GAPS)
