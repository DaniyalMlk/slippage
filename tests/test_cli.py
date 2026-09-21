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
