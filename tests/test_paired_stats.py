"""Offline checks with hand-counted paired windows and seed measurements."""

import json

import pytest

from eval.paired_stats import compare_audits, main, mcnemar_from_outcomes, seed_stats


def audit(wrong, *, order=None, scene="clip"):
    rows = [{"k": k, "t": k * 0.5, "wrong": value} for k, value in enumerate(wrong)]
    if order is not None:
        rows = [rows[k] for k in order]
    return {"scenes": [{"scene": scene, "target": "a", "other": "b", "rows": rows}]}


def test_pair_by_identity_and_report_all_four_cells():
    a = audit([False, True, False, True])
    b = audit([False, False, True, True], order=[3, 1, 0, 2])
    result = compare_audits(a, b)
    assert result["table"] == [[1, 1], [1, 1]]
    assert result["b"] == result["c"] == 1
    assert result["n_pairs"] == 4
    assert result["p"] == 1.0
    assert result["chi2"] == 0.0


def test_exact_small_sample_probability_and_direction():
    result = mcnemar_from_outcomes([False] * 6, [True] * 6)
    assert result["b"] == 6
    assert result["c"] == 0
    assert result["p"] == pytest.approx(0.03125)
    assert result["chi2"] == pytest.approx(25 / 6)
    assert result["table"] == [[0, 6], [0, 0]]


@pytest.mark.parametrize("a,b", [([True], [True]), ([False], [False])])
def test_no_discordance(a, b):
    result = mcnemar_from_outcomes(a, b)
    assert result["p"] == 1.0
    assert result["chi2"] == 0.0


@pytest.mark.parametrize("a,b", [([], []), ([True], []), ([1], [True])])
def test_invalid_binary_pairs_rejected(a, b):
    with pytest.raises(ValueError):
        mcnemar_from_outcomes(a, b)


@pytest.mark.parametrize("change", ["missing", "duplicate", "time", "speaker", "nonbool"])
def test_invalid_audit_pair_is_not_silently_dropped(change):
    a, b = audit([False, True]), audit([True, False])
    scene = b["scenes"][0]
    if change == "missing":
        scene["rows"].pop()
    elif change == "duplicate":
        scene["rows"].append(scene["rows"][0])
    elif change == "time":
        scene["rows"][0]["t"] = 0.1
    elif change == "speaker":
        scene["target"] = "different"
    else:
        scene["rows"][0]["wrong"] = 1
    with pytest.raises(ValueError):
        compare_audits(a, b)


def test_seed_sample_sd_and_count_exclusion():
    data = {f"seed{k}": {"eval": {"raw": {"flip_rate": value, "n_queries": 10}}}
            for k, value in enumerate([1.0, 2.0, 3.0])}
    assert seed_stats(data, list(data)) == {"eval": {"raw": {"flip_rate": (2.0, 1.0)}}}
    assert seed_stats(data, ["seed0"])["eval"]["raw"]["flip_rate"] == (1.0, None)
    del data["seed1"]["eval"]["raw"]["flip_rate"]
    with pytest.raises(ValueError):
        seed_stats(data, list(data))


def test_cli_two_audit_files_and_seed_table(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(audit([True, False])))
    b.write_text(json.dumps(audit([False, False])))
    assert main(["mcnemar", str(a), str(b)]) == 0
    assert json.loads(capsys.readouterr().out)["b"] == 1
    a.write_text(json.dumps({"seed0": {"eval": {"raw": {"flip_rate": 0.1}}},
                             "seed1": {"eval": {"raw": {"flip_rate": 0.3}}}}))
    assert main(["seeds", str(a)]) == 0
    output = capsys.readouterr().out
    assert "0.2000+/-0.1414" in output


def test_same_window_index_in_distinct_scenes_is_not_collapsed():
    a = audit([True])
    a["scenes"].extend(audit([False], scene="second")["scenes"])
    b = audit([False], scene="second")
    b["scenes"].extend(audit([False])["scenes"])
    result = compare_audits(a, b)
    assert result["n_pairs"] == 2
    assert result["table"] == [[0, 1], [0, 1]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5"])
def test_seed_measurements_must_be_finite_numbers(value):
    with pytest.raises(ValueError):
        seed_stats({"seed0": {"eval": {"raw": {"flip_rate": value}}}}, ["seed0"])


def test_empty_and_duplicate_scenes_rejected():
    with pytest.raises(ValueError):
        compare_audits({"scenes": []}, {"scenes": []})
    duplicated = audit([False])
    duplicated["scenes"] *= 2
    with pytest.raises(ValueError):
        compare_audits(duplicated, audit([False]))


def test_cli_missing_pairs_reports_error_without_statistics(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(audit([True, False])))
    b.write_text(json.dumps(audit([False])))
    with pytest.raises(SystemExit) as error:
        main(["mcnemar", str(a), str(b)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "A-only=1, B-only=0" in output.err
