"""`evaluation/summarize_eval_suite.py`: flattening of the long-context suite JSON and the merged
markdown table (missing pieces must render as '-' instead of failing)."""
import csv
import json

from evaluation import summarize_eval_suite as agg


def _write_suite(root, tag, results, errors=None):
    d = root / tag
    d.mkdir(parents=True)
    (d / "longctx_suite.json").write_text(json.dumps(
        {"suite": list(results), "results": results, "errors": errors or {}, "elapsed_s": {}}))


def test_flatten_longctx_keeps_exact_and_token_acc_and_buckets(tmp_path):
    _write_suite(tmp_path, "a", {
        "passkey": {"passkey@8192": 1.0, "passkey_token_acc@8192": 1.0, "passkey_first_token_acc@8192": 1.0,
                    "passkey_n@8192": 8},
        "vt": {"vt@8192": 0.25, "vt_token_acc@8192": 0.5, "vt_first_token_acc@8192": 0.5, "vt_n@8192": 8},
        "buckets": {"rows": 10, "ce[8192,32768)": 2.5},
    }, errors={"fwe": "RuntimeError: boom\n  trace"})
    flat = agg.load_longctx(tmp_path, "a")
    assert flat["passkey@8192"] == 1.0 and flat["passkey_tok@8192"] == 1.0
    assert "passkey_first_token_acc@8192" not in flat and "passkey_n@8192" not in flat
    assert flat["vt_tok@8192"] == 0.5 and flat["ce[8192,32768)"] == 2.5 and "rows" not in flat
    assert flat["error:fwe"] == "RuntimeError: boom"
    assert agg.load_longctx(tmp_path, "missing") == {}


def test_table_merges_lm_eval_rows_and_longctx_with_dashes_for_missing(tmp_path):
    _write_suite(tmp_path / "eval", "ck", {"passkey": {"passkey@8192": 0.75, "passkey_token_acc@8192": 0.9}})
    csv_path = tmp_path / "summary.csv"
    with csv_path.open("w", newline="") as f:
        cols = ["tag", "model", "avg_acc", "n_acc_tasks", "hellaswag/acc_norm", "hellaswag/acc_norm_stderr",
                "wikitext/word_perplexity"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"tag": "ck", "model": "x", "avg_acc": 0.4123, "n_acc_tasks": 2, "hellaswag/acc_norm": 0.31,
                    "hellaswag/acc_norm_stderr": 0.01, "wikitext/word_perplexity": 40.5})
        w.writerow({"tag": "ref", "model": "y", "avg_acc": 0.5, "n_acc_tasks": 2, "hellaswag/acc_norm": 0.42,
                    "hellaswag/acc_norm_stderr": 0.01, "wikitext/word_perplexity": 30.25})
    table = agg.build_table(["ck", "ref"], tmp_path / "eval", csv_path)
    lines = table.splitlines()
    assert lines[0] == "| metric | ck | ref |"
    row = {l.split("|")[1].strip(): l for l in lines if l.startswith("| ") and "**" not in l}
    assert row["avg_acc"].endswith("| 0.412 | 0.500 |")
    assert row["n_acc_tasks"].endswith("| 2 | 2 |")
    assert row["hellaswag"].endswith("| 0.310 | 0.420 |")
    assert row["wikitext"].endswith("| 40.50 | 30.25 |")
    assert row["passkey@8192"].endswith("| 0.750 | - |")       # ref has no long-context run
    assert row["passkey_tok@8192"].endswith("| 0.900 | - |")
    # ordering: reasoning block first, long-context block after
    assert lines.index(row["avg_acc"]) < lines.index(row["passkey@8192"])


def test_cli_writes_markdown(tmp_path, capsys):
    _write_suite(tmp_path / "eval", "only", {"buckets": {"ce[0,8192)": 3.0}})
    out = tmp_path / "t.md"
    agg.main(["--all", "--eval_root", str(tmp_path / "eval"), "--lm_csv", str(tmp_path / "none.csv"),
              "--out", str(out)])
    text = out.read_text()
    assert "| metric | only |" in text and "ce[0,8192)" in text and "3.000" in text
