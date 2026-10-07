from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from slippage.cli import main

AC_ARGS = [
    "--quantity", "1000000",
    "--horizon", "5",
    "--periods", "5",
    "--volatility", "0.95",
    "--gamma", "2.5e-7",
    "--eta", "2.5e-6",
    "--epsilon", "0.0625",
]  # fmt: skip


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("book")
    code, text = run("sample-data", "--out", str(folder), "--seed", "3", "--orders", "40")
    assert code == 0
    assert "wrote 40 orders" in text
    return folder


def tca_args(folder: Path) -> list[str]:
    return [
        "tca",
        "--orders", str(folder / "orders.csv"),
        "--fills", str(folder / "fills.csv"),
        "--bars", str(folder / "bars.csv"),
    ]  # fmt: skip


class TestTca:
    def test_text_report(self, sample: Path) -> None:
        code, text = run(*tca_args(sample))
        assert code == 0
        assert "40 orders" in text
        assert "basis points of paper notional" in text
        assert "\nall " in text

    def test_json_report_adds_up(self, sample: Path) -> None:
        code, text = run(*tca_args(sample), "--json", "--fees-per-share", "0.002")
        assert code == 0
        payload = json.loads(text)
        total = payload["total"]
        assert total["orders"] == 40
        assert sum(total["components_bps"].values()) == pytest.approx(total["total_bps"])
        assert sum(g["orders"] for g in payload["by_symbol"].values()) == 40

    def test_group_by_side(self, sample: Path) -> None:
        code, text = run(*tca_args(sample), "--group-by", "side")
        assert code == 0
        assert "\nbuy " in text or "\nsell " in text

    def test_bad_input_exits_2_with_a_located_message(
        self, sample: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "fills.csv"
        lines = (sample / "fills.csv").read_text().splitlines()
        lines[3] = lines[3].rsplit(",", 2)[0] + ",oops,0.0"
        broken.write_text("\n".join(lines) + "\n")
        args = tca_args(sample)
        args[args.index("--fills") + 1] = str(broken)
        code, _ = run(*args)
        assert code == 2
        assert "fills.csv:4" in capsys.readouterr().err

    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        code, _ = run(*tca_args(tmp_path))
        assert code == 2


class TestSchedule:
    def test_closed_form_for_linear_unconstrained(self) -> None:
        code, text = run("schedule", *AC_ARGS, "--risk-aversion", "1e-6", "--json")
        assert code == 0
        payload = json.loads(text)
        assert payload["method"] == "closed form"
        assert payload["expected_cost"] == pytest.approx(911_227, rel=1e-6)
        assert sum(payload["trades"]) == pytest.approx(1e6)

    def test_caps_switch_to_the_dynamic_programme(self) -> None:
        code, text = run(
            "schedule", *AC_ARGS, "--risk-aversion", "1e-6", "--max-trade", "250000",
            "--lot-size", "1000", "--json",
        )  # fmt: skip
        assert code == 0
        payload = json.loads(text)
        assert payload["method"].startswith("dynamic programme")
        assert max(payload["trades"]) <= 250_000
        assert sum(payload["trades"]) == pytest.approx(1e6)

    def test_per_period_caps_and_power_law(self) -> None:
        code, text = run(
            "schedule", *AC_ARGS, "--eta", "1e-3", "--beta", "0.5", "--risk-aversion", "1e-6",
            "--max-trade", "300000,300000,200000,200000,200000", "--lot-size", "10000",
        )  # fmt: skip
        assert code == 0
        assert "dynamic programme, lot 10000" in text
        assert "expected cost" in text

    def test_infeasible_caps_exit_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _ = run(
            "schedule", *AC_ARGS, "--risk-aversion", "1e-6", "--max-trade", "1000",
            "--lot-size", "1000",
        )  # fmt: skip
        assert code == 2
        assert "no schedule completes" in capsys.readouterr().err

    def test_text_table(self) -> None:
        code, text = run("schedule", *AC_ARGS, "--risk-aversion", "0")
        assert code == 0
        assert text.count("200,000") >= 5


class TestFrontier:
    def test_default_grid(self) -> None:
        code, text = run("frontier", *AC_ARGS, "--json")
        assert code == 0
        rows = json.loads(text)
        assert [r["risk_aversion"] for r in rows] == [0.0, 1e-8, 1e-7, 1e-6, 1e-5]
        assert rows[0]["half_life"] is None
        costs = [r["expected_cost"] for r in rows]
        assert costs == sorted(costs)

    def test_text_output(self) -> None:
        code, text = run("frontier", *AC_ARGS, "--risk-aversions", "1e-7,1e-6")
        assert code == 0
        assert "half-life" in text
        assert "458,044" in text

    def test_power_law_is_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _ = run("frontier", *AC_ARGS, "--beta", "0.5")
        assert code == 2
        assert "linear impact" in capsys.readouterr().err

    def test_bad_number_list_is_an_argument_error(self) -> None:
        with pytest.raises(SystemExit):
            run("frontier", *AC_ARGS, "--risk-aversions", "1e-6,lots")


@pytest.fixture(scope="module")
def decaying(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A book whose prints carry decaying impact, plus the index it moved with.

    ``sample-data`` writes the other generator, whose bars are the unaffected
    market — correct for calibrating against an arrival benchmark and empty of
    anything a mark-out could read.
    """
    import numpy as np

    from slippage.io import write_bars, write_orders
    from slippage.synthetic import decaying_book

    folder = tmp_path_factory.mktemp("marks")
    book = decaying_book(np.random.default_rng(7), orders=60)
    write_orders(book.orders, folder / "orders.csv", folder / "fills.csv")
    write_bars(book.bars, folder / "bars.csv")
    write_bars({"INDEX": book.index}, folder / "index.csv")
    return folder


def markout_args(folder: Path) -> list[str]:
    return [
        "markouts",
        "--orders", str(folder / "orders.csv"),
        "--fills", str(folder / "fills.csv"),
        "--bars", str(folder / "bars.csv"),
    ]  # fmt: skip


class TestMarkOuts:
    def test_text_report(self, decaying: Path) -> None:
        code, text = run(*markout_args(decaying))
        assert code == 0
        assert "Mark-outs over 60 orders" in text
        assert "persisted" in text
        assert "reverted" in text
        assert "half-life" in text

    def test_an_unadjusted_report_says_so_twice(self, decaying: Path) -> None:
        """In the header and again at the end, because it is the one thing that
        decides whether any of the numbers mean anything."""
        code, text = run(*markout_args(decaying))
        assert code == 0
        assert "NOT market-adjusted" in text
        assert "No benchmark was given" in text

    def test_the_benchmark_sharpens_every_figure(self, decaying: Path) -> None:
        plain = json.loads(run(*markout_args(decaying), "--json")[1])
        adjusted = json.loads(
            run(
                *markout_args(decaying),
                "--benchmark",
                str(decaying / "index.csv"),
                "--json",
            )[1]
        )
        assert adjusted["benchmark_adjusted"] is True
        assert adjusted["beta"] == 1.0
        assert plain["beta"] is None
        assert adjusted["impact_standard_error"] < 0.25 * plain["impact_standard_error"]
        for near, far in zip(plain["horizons"], adjusted["horizons"], strict=True):
            assert far["standard_error"] < near["standard_error"]
        # The truth in the generator is 10.0 basis points of impact at completion,
        # 4.0 of it permanent, decaying with a 208-second half-life.
        assert adjusted["mean_impact_bps"] == pytest.approx(10.0, abs=1.0)
        assert adjusted["decay"]["asymptote_bps"] == pytest.approx(4.0, abs=1.0)
        assert not text_contains_nan(json.dumps(adjusted))

    def test_the_payload_is_json_a_strict_parser_accepts(self, decaying: Path) -> None:
        """Horizons past the end of the bars carry no mean, and a NaN there is not
        JSON: ``json.dumps`` writes it bare and ``json.loads`` reads it back, so a
        round trip does not catch it. Those fields are null."""
        code, text = run(*markout_args(decaying), "--horizons", "1,5,600", "--json")
        assert code == 0
        payload = json.loads(text)
        assert json.dumps(payload, allow_nan=False)
        unreachable = payload["horizons"][-1]
        assert unreachable["orders"] == 0
        assert unreachable["mean_permanent_bps"] is None
        assert unreachable["standard_error"] is None

    def test_a_beta_is_applied_and_reported(self, decaying: Path) -> None:
        half = json.loads(
            run(
                *markout_args(decaying),
                "--benchmark",
                str(decaying / "index.csv"),
                "--beta",
                "0.5",
                "--json",
            )[1]
        )
        full = json.loads(
            run(
                *markout_args(decaying),
                "--benchmark",
                str(decaying / "index.csv"),
                "--json",
            )[1]
        )
        assert half["beta"] == 0.5
        # The generator moves every symbol one for one with the index, so half the
        # adjustment leaves half the market in and the estimate loses precision.
        assert half["impact_standard_error"] > full["impact_standard_error"]

    def test_a_negative_horizon_is_refused_by_the_parser(self, decaying: Path) -> None:
        with pytest.raises(SystemExit):
            run(*markout_args(decaying), "--horizons", "5,-10")

    def test_a_benchmark_file_with_several_symbols_asks_which(self, decaying: Path) -> None:
        code, _ = run(*markout_args(decaying), "--benchmark", str(decaying / "bars.csv"))
        assert code == 2
        code, _ = run(
            *markout_args(decaying),
            "--benchmark",
            str(decaying / "bars.csv"),
            "--benchmark-symbol",
            "SYM0000",
        )
        assert code == 0
        code, _ = run(
            *markout_args(decaying),
            "--benchmark",
            str(decaying / "index.csv"),
            "--benchmark-symbol",
            "NOPE",
        )
        assert code == 2

    def test_a_curve_with_no_decay_reports_why_rather_than_a_half_life(
        self, tmp_path: Path
    ) -> None:
        import numpy as np

        from slippage.io import write_bars, write_orders
        from slippage.synthetic import decaying_book

        book = decaying_book(
            np.random.default_rng(2),
            orders=20,
            permanent_bps=10.0,
            temporary_bps=-6.0,
            market_volatility_bps=0.0,
            idiosyncratic_volatility_bps=0.0,
        )
        write_orders(book.orders, tmp_path / "o.csv", tmp_path / "f.csv")
        write_bars(book.bars, tmp_path / "b.csv")
        code, text = run(
            "markouts",
            "--orders", str(tmp_path / "o.csv"),
            "--fills", str(tmp_path / "f.csv"),
            "--bars", str(tmp_path / "b.csv"),
        )  # fmt: skip
        assert code == 0
        assert "no half-life" in text
        assert "does not decay" in text


def text_contains_nan(text: str) -> bool:
    return "NaN" in text or "Infinity" in text


# -- the basket command -------------------------------------------------------


SIGMA = 0.945


@pytest.fixture
def basket_files(tmp_path: Path) -> tuple[Path, Path]:
    """A correlated long/short pair whose short leg is four times as expensive."""
    holdings = tmp_path / "holdings.csv"
    holdings.write_text("symbol,quantity,eta,gamma\nLIQ,100000,1e-6,1e-7\nILQ,-100000,4e-6,4e-7\n")
    covariance = tmp_path / "cov.csv"
    variance = SIGMA * SIGMA
    covariance.write_text(f"{variance},{0.9 * variance}\n{0.9 * variance},{variance}\n")
    return holdings, covariance


def test_the_basket_command_reports_the_directions(
    basket_files: tuple[Path, Path],
) -> None:
    holdings, covariance = basket_files
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
    )
    assert code == 0
    assert "2 assets over 20 intervals" in output
    assert "risk/impact" in output
    assert "half-life" in output
    assert "liquidated along these directions" in output


def test_the_basket_command_compares_against_leg_by_leg(
    basket_files: tuple[Path, Path],
) -> None:
    holdings, covariance = basket_files
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
        "--compare",
    )
    assert code == 0
    assert "Solved leg by leg" in output
    assert "times as much exposure" in output
    # The counterintuitive half has to be in the output, not only in the docs.
    assert "deliberately the riskier one" in output


def test_the_basket_command_emits_json(basket_files: tuple[Path, Path]) -> None:
    holdings, covariance = basket_files
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
        "--compare",
        "--json",
    )
    assert code == 0
    payload = json.loads(output)
    assert payload["assets"] == ["LIQ", "ILQ"]
    assert len(payload["directions"]) == 2
    assert len(payload["holdings"]) == 21
    assert payload["objective_saving"] > 0.05
    assert payload["exposure_ratio"] > 5.0
    assert payload["peak_risk_ratio"] < 1.0
    # Nothing non-finite: an infinite half-life is written as null on purpose,
    # because Infinity is not JSON and a strict reader rejects the document.
    json.dumps(payload, allow_nan=False)


def test_a_risk_neutral_basket_writes_a_null_half_life(
    basket_files: tuple[Path, Path],
) -> None:
    holdings, covariance = basket_files
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "0",
        "--json",
    )
    assert code == 0
    payload = json.loads(output)
    assert all(one["half_life"] is None for one in payload["directions"])
    json.dumps(payload, allow_nan=False)


def test_a_symmetric_basket_says_why_the_exposure_ratio_is_missing(
    tmp_path: Path,
) -> None:
    holdings = tmp_path / "holdings.csv"
    holdings.write_text("symbol,quantity,eta,gamma\nA,100000,1e-6,1e-7\nB,-100000,1e-6,1e-7\n")
    covariance = tmp_path / "cov.csv"
    variance = SIGMA * SIGMA
    covariance.write_text(f"{variance},{0.9 * variance}\n{0.9 * variance},{variance}\n")
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
        "--compare",
    )
    assert code == 0
    assert "rounding rather than exposure" in output


def test_cross_impact_can_be_supplied_as_a_matrix(
    basket_files: tuple[Path, Path], tmp_path: Path
) -> None:
    holdings, covariance = basket_files
    cross = tmp_path / "cross.csv"
    cross.write_text("1e-6,5e-7\n5e-7,4e-6\n")
    code, output = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--cross-impact",
        str(cross),
        "--risk-aversion",
        "1e-5",
        "--json",
    )
    assert code == 0
    assert json.loads(output)["expected_cost"] > 0.0


def test_a_ragged_covariance_row_names_its_line(
    basket_files: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    holdings, _ = basket_files
    covariance = tmp_path / "bad.csv"
    covariance.write_text("1.0,0.5\n0.5\n")
    code, _ = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
    )
    assert code != 0
    assert "line 2 has 1 entries for 2 assets" in capsys.readouterr().err


def test_a_non_numeric_covariance_entry_names_its_line(
    basket_files: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    holdings, _ = basket_files
    covariance = tmp_path / "bad.csv"
    covariance.write_text("1.0,0.5\n0.5,par\n")
    code, _ = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
    )
    assert code != 0
    assert "not a row of numbers" in capsys.readouterr().err


def test_a_holdings_file_missing_a_column_says_which(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    holdings = tmp_path / "holdings.csv"
    holdings.write_text("symbol,quantity\nA,100\n")
    covariance = tmp_path / "cov.csv"
    covariance.write_text("1.0\n")
    code, _ = run(
        "basket",
        "--holdings",
        str(holdings),
        "--covariance",
        str(covariance),
        "--risk-aversion",
        "1e-5",
    )
    assert code != 0
    assert "missing eta, gamma" in capsys.readouterr().err


# -- transient ---------------------------------------------------------------

TRANSIENT_ARGS = [
    "transient",
    "--quantity", "100000",
    "--horizon", "8",
    "--periods", "8",
    "--gamma", "1e-6",
    "--eta", "2e-5",
]  # fmt: skip


def test_the_transient_command_prints_the_block_rate_block_shape() -> None:
    code, text = run(*TRANSIENT_ARGS, "--half-life", "2.7726")
    assert code == 0
    assert "ExponentialDecay" in text
    assert "% saved" in text
    assert "impact left after the order" in text


def test_the_transient_command_reports_a_saving_and_the_impact_left_behind() -> None:
    code, text = run(*TRANSIENT_ARGS, "--half-life", "2.7726", "--json")
    assert code == 0
    payload = json.loads(text)
    trades = payload["trades"]
    assert len(trades) == 8
    assert sum(trades) == pytest.approx(100_000.0, rel=1e-9)
    # The two end blocks are equal and the middle is flat, which is the whole
    # shape the kernel produces and nothing in the command imposes.
    assert trades[0] == pytest.approx(trades[-1], rel=1e-9)
    assert trades[1:-1] == pytest.approx([trades[1]] * 6, rel=1e-9)
    assert payload["front_load"] > 2.0
    assert 0.0 < payload["saving"] < 0.1
    assert payload["cost"] < payload["uniform_cost"]
    # The impact left behind decays rather than staying put or vanishing.
    residual = [one["impact"] for one in payload["residual_impact"]]
    assert residual == sorted(residual, reverse=True)
    assert residual[-1] > payload["permanent_impact_per_share"] * 100_000.0


def test_the_half_life_and_the_resilience_are_the_same_parameter() -> None:
    by_life = json.loads(run(*TRANSIENT_ARGS, "--half-life", "2.7726", "--json")[1])
    by_rate = json.loads(run(*TRANSIENT_ARGS, "--resilience", "0.25", "--json")[1])
    assert by_life["trades"] == pytest.approx(by_rate["trades"], rel=1e-4)
    assert by_life["cost"] == pytest.approx(by_rate["cost"], rel=1e-4)


def test_giving_both_decay_arguments_is_refused_rather_than_one_winning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*TRANSIENT_ARGS, "--half-life", "2.0", "--resilience", "0.3"]) == 2
    assert "not both" in capsys.readouterr().err


def test_giving_neither_decay_argument_is_refused() -> None:
    assert main(TRANSIENT_ARGS) == 2


def test_a_power_law_kernel_is_available_and_needs_a_scale() -> None:
    code, text = run(*TRANSIENT_ARGS, "--exponent", "0.6", "--half-life", "2.0", "--json")
    assert code == 0
    assert json.loads(text)["kernel"] == "PowerLawDecay"
    # A power law has no resilience rate to be given instead of a scale.
    assert main([*TRANSIENT_ARGS, "--exponent", "0.6", "--resilience", "0.3"]) == 2


# -- adaptive -----------------------------------------------------------------

ADAPTIVE_ARGS = [
    "adaptive",
    "--quantity", "1000000",
    "--horizon", "1",
    "--periods", "20",
    "--gamma", "2.5e-7",
    "--eta", "2.5e-6",
    "--illiquid-eta", "1.25e-5",
    "--volatility", "0.3",
    "--risk-aversion", "2e-6",
]  # fmt: skip


def test_adaptive_reports_the_saving_and_the_policy_table() -> None:
    code, text = run(*ADAPTIVE_ARGS, "--persistence", "0.8")
    assert code == 0
    assert "starting regime: liquid" in text
    assert "24.028% of the static objective" in text
    # Twenty periods plus a header, and the last row trades everything.
    rows = [line for line in text.splitlines() if line.strip()[:1].isdigit()]
    assert len(rows) == 20
    assert rows[-1].split()[1:4] == ["1.000000", "1.000000", "1.000000"]


def test_adaptive_in_json_carries_the_whole_policy() -> None:
    code, text = run(*ADAPTIVE_ARGS, "--persistence", "0.8", "--json")
    assert code == 0
    payload = json.loads(text)
    assert payload["start"] == "liquid"
    assert payload["saved_fraction"] == pytest.approx(0.2403, abs=5e-4)
    assert payload["adaptive_objective"] < payload["static_objective"]
    assert len(payload["periods"]) == 20
    assert payload["periods"][0]["liquid_fraction"] < payload["periods"][0]["static_fraction"]
    assert payload["periods"][10]["liquid_fraction"] > payload["periods"][10]["static_fraction"]
    assert payload["static_expected_impact"] > 0.0
    assert payload["static_expected_risk"] > 0.0
    assert "simulated" not in payload


def test_adaptive_simulates_the_policy_when_asked() -> None:
    code, text = run(
        *ADAPTIVE_ARGS, "--persistence", "0.8", "--draws", "500", "--seed", "11", "--json"
    )
    assert code == 0
    simulated = json.loads(text)["simulated"]
    assert simulated["draws"] == 500
    assert simulated["standard_error"] > 0.0
    assert simulated["covers_the_recursion"] is True


@pytest.mark.parametrize("persistence", ["0.0", "1.0"])
def test_adaptive_reports_nothing_to_gain_on_a_predictable_chain(persistence: str) -> None:
    code, text = run(*ADAPTIVE_ARGS, "--persistence", persistence, "--json")
    assert code == 0
    assert json.loads(text)["saved_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_adaptive_starting_illiquid_gains_more_at_four_fifths() -> None:
    _, liquid = run(*ADAPTIVE_ARGS, "--persistence", "0.8", "--start", "0", "--json")
    _, illiquid = run(*ADAPTIVE_ARGS, "--persistence", "0.8", "--start", "1", "--json")
    assert json.loads(illiquid)["start"] == "illiquid"
    assert json.loads(illiquid)["saved_fraction"] > json.loads(liquid)["saved_fraction"]


def test_adaptive_takes_a_separate_volatility_for_the_illiquid_regime() -> None:
    code, text = run(
        *ADAPTIVE_ARGS, "--persistence", "0.8", "--illiquid-volatility", "0.9", "--json"
    )
    assert code == 0
    assert json.loads(text)["saved_fraction"] > 0.0


def test_adaptive_refuses_a_persistence_that_is_not_a_probability() -> None:
    assert main([*ADAPTIVE_ARGS, "--persistence", "1.4"]) == 2


def test_adaptive_refuses_a_regime_whose_permanent_impact_dominates() -> None:
    """A gamma large enough that one period of it outruns the concession.

    Refused per regime rather than on average: the policy can be routed through
    any state, so a chain is admissible only if every state in it is.
    """
    heavy = [
        "adaptive",
        "--quantity", "1000000",
        "--horizon", "1",
        "--periods", "20",
        "--gamma", "1e-3",
        "--eta", "2.5e-6",
        "--illiquid-eta", "1.25e-5",
        "--volatility", "0.3",
        "--persistence", "0.8",
    ]  # fmt: skip
    assert main(heavy) == 2


PLACEMENT_ARGS = [
    "placement",
    "--distance", "0.05",
    "--horizon", "1",
    "--volatility", "0.2",
    "--half-spread", "0.0005",
    "--taker-fee", "0.0001",
    "--maker-rebate", "0.0001",
]  # fmt: skip


def test_placement_reports_the_factor_of_two_and_the_conditional_mid() -> None:
    code, text = run(*PLACEMENT_ARGS)
    assert code == 0
    assert "ratio 2.000000" in text
    # Driftless, so the mid conditional on a fill is the limit price itself.
    assert "mid conditional on a fill: -0.05, against a limit price of -0.05" in text


def test_placement_in_json_carries_every_quantity() -> None:
    code, text = run(*PLACEMENT_ARGS, "--json")
    assert code == 0
    payload = json.loads(text)
    assert payload["running_over_terminal"] == pytest.approx(2.0, abs=1e-12)
    assert payload["mid_if_filled"] == pytest.approx(-0.05, rel=1e-12)
    assert payload["mid_if_unfilled"] > 0.0
    assert payload["expected_cost"] == pytest.approx(
        0.0006 * (1.0 - payload["fill_probability"]) - 0.0001 * payload["fill_probability"],
        rel=1e-12,
    )
    assert payload["advantage"] > 0.0


def test_placement_sweep_moves_both_columns_the_same_way() -> None:
    code, text = run(*PLACEMENT_ARGS, "--json", "--sweep", "0.01", "0.5", "1", "2", "4")
    assert code == 0
    rows = json.loads(text)["sweep"]
    assert [row["deviations"] for row in rows] == [0.01, 0.5, 1.0, 2.0, 4.0]
    means = [row["expected_cost"] for row in rows]
    spreads = [row["cost_deviation"] for row in rows]
    assert means == sorted(means)
    assert spreads == sorted(spreads)


def test_placement_sweep_in_text_says_there_is_nothing_to_trade_off() -> None:
    code, text = run(*PLACEMENT_ARGS, "--sweep", "0.5", "1")
    assert code == 0
    assert "sd out" in text
    assert "tightest price the book allows" in text


def test_placement_refuses_a_non_positive_sweep() -> None:
    assert main([*PLACEMENT_ARGS, "--sweep", "0.5", "0"]) == 2


def test_placement_simulates_and_reports_the_monitoring_bias() -> None:
    code, text = run(*PLACEMENT_ARGS, "--json", "--draws", "20000", "--steps", "500", "--seed", "4")
    assert code == 0
    simulated = json.loads(text)["simulated"]
    assert simulated["paths"] == 20000
    assert simulated["steps"] == 500
    # The formula reads high against a discretely monitored path, and the
    # continuity correction closes most of that gap.
    assert simulated["monitoring_bias"] > 0.0
    assert abs(simulated["corrected_gap"]) < 0.5 * simulated["monitoring_bias"]


def test_placement_text_output_mentions_the_correction() -> None:
    code, text = run(*PLACEMENT_ARGS, "--draws", "10000", "--steps", "400", "--seed", "1")
    assert code == 0
    assert "discrete monitoring reads" in text
    assert "continuity-corrected formula" in text


def test_placement_with_a_drift_loses_to_crossing() -> None:
    code, text = run(*PLACEMENT_ARGS, "--json", "--drift", "0.3")
    assert code == 0
    payload = json.loads(text)
    assert payload["advantage"] < 0.0
    # An adverse drift lifts the conditional mid off the limit price.
    assert payload["mid_if_filled"] > -0.05


def test_placement_refuses_a_zero_volatility() -> None:
    arguments = list(PLACEMENT_ARGS)
    arguments[arguments.index("--volatility") + 1] = "0"
    assert main(arguments) == 2


VWAP_ARGS = [
    "vwap",
    "--quantity", "1000000",
    "--profile", "1.9", "1.3", "1.0", "0.85", "0.8", "0.75", "0.72",
    "0.75", "0.8", "0.9", "1.1", "1.4", "2.1",
    "--concentration", "60",
    "--volatility", "0.004",
    "--price", "50",
    "--bucket-hours", "0.5",
]  # fmt: skip


def test_vwap_leads_with_the_floor() -> None:
    """Because the floor decides whether anything else in the report is actionable."""
    out = io.StringIO()
    assert main([*VWAP_ARGS, "--json"], out) == 0
    payload = json.loads(out.getvalue())
    assert payload["buckets"] == 13
    assert payload["irreducible_bps"] == pytest.approx(8.05, abs=0.05)
    # the volume-matching schedule is exactly at the floor, by construction
    assert payload["tracking_error_bps"] == pytest.approx(payload["irreducible_bps"], rel=1e-12)
    assert payload["concentration"] == pytest.approx(60.0)
    assert sum(payload["schedule"]) == pytest.approx(1.0, rel=1e-12)
    assert payload["schedule"][0] > payload["schedule"][6]


def test_vwap_frontier_runs_from_impact_to_the_floor() -> None:
    out = io.StringIO()
    extra = ["--eta", "2e-6", "--gamma", "2e-7", "--risk-aversion", "0", "1000", "1000000"]
    assert main([*VWAP_ARGS, *extra, "--json"], out) == 0
    frontier = json.loads(out.getvalue())["frontier"]
    assert len(frontier) == 3
    assert frontier[0]["tracking_error_bps"] > frontier[-1]["tracking_error_bps"]
    assert frontier[0]["impact_bps"] < frontier[-1]["impact_bps"]
    # no weight on tracking error means an equal slice regardless of the volume
    assert frontier[0]["schedule"] == pytest.approx([1.0 / 13.0] * 13, abs=1e-4)


def test_vwap_frontier_needs_an_impact_parameter() -> None:
    out = io.StringIO()
    assert main([*VWAP_ARGS, "--risk-aversion", "100"], out) == 2


def test_vwap_compares_an_arrival_price_schedule_against_the_benchmark() -> None:
    """The number that says what the other objective costs here."""
    front_loaded = [f"{0.7**k:.6f}" for k in range(13)]
    out = io.StringIO()
    assert main([*VWAP_ARGS, "--compare", *front_loaded, "--json"], out) == 0
    comparison = json.loads(out.getvalue())["comparison"]
    assert comparison["other_schedule_error_bps"] > comparison["vwap_schedule_error_bps"]
    assert comparison["excess_over_floor"] > 3.0
    text = io.StringIO()
    assert main([*VWAP_ARGS, "--compare", *front_loaded], text) == 0
    assert "times the floor" in text.getvalue()


def test_vwap_fits_a_concentration_to_a_dispersion_vector() -> None:
    """Which is the field volume.estimate_profile has always produced and nothing read."""
    out = io.StringIO()
    dispersion = [
        "0.0434", "0.0349", "0.0310", "0.0288", "0.0280", "0.0272", "0.0266",
        "0.0272", "0.0280", "0.0296", "0.0325", "0.0362", "0.0455",
    ]  # fmt: skip
    base = [*VWAP_ARGS[:-8], "--volatility", "0.004", "--dispersion", *dispersion]
    assert main([*base, "--json"], out) == 0
    payload = json.loads(out.getvalue())
    assert payload["concentration"] > 1.0
    assert payload["irreducible_bps"] > 0.0


def test_vwap_needs_uncertainty_from_somewhere() -> None:
    """A VWAP benchmark is only unreachable because volume is uncertain."""
    out = io.StringIO()
    bare = [one for one in VWAP_ARGS if one not in {"--concentration", "60"}]
    assert main(bare, out) == 2


def test_vwap_refuses_mismatched_vector_lengths() -> None:
    out = io.StringIO()
    assert main([*VWAP_ARGS, "--compare", "1", "2", "3"], out) == 2
    assert main([*VWAP_ARGS, "--dispersion", "0.04", "0.03"], out) == 2
