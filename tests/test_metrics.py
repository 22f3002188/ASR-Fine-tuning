"""Unit tests: WER / CER computation and error analysis."""

import pytest


def test_wer_perfect():
    from jiwer import wer
    assert wer(["hello world"], ["hello world"]) == 0.0


def test_wer_all_wrong():
    from jiwer import wer
    assert wer(["hello world"], ["foo bar"]) == 1.0


def test_cer_basic():
    from jiwer import cer
    score = cer(["abc"], ["abd"])
    assert 0.0 < score <= 1.0


def test_compute_metrics_returns_wer_cer():
    from src.evaluation.metrics import make_compute_metrics
    from unittest.mock import MagicMock
    import numpy as np

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.batch_decode = lambda ids, **kw: ["hello world", "foo bar"]

    compute = make_compute_metrics(tokenizer)

    class FakeEvalPred:
        predictions = np.array([[1, 2, 3], [4, 5, 6]])
        label_ids   = np.array([[1, 2, 3], [4, 5, 6]])

    result = compute(FakeEvalPred())
    assert "wer" in result
    assert "cer" in result
    assert isinstance(result["wer"], float)


def test_error_analyser_breakdown():
    from src.evaluation.error_analysis import ErrorAnalyser

    refs = ["the cat sat on the mat", "hello world"]
    hyps = ["the cat sat on mat",     "hello world"]
    analyser = ErrorAnalyser(refs, hyps)
    bd = analyser.error_breakdown()

    assert "substitutions" in bd
    assert "deletions"     in bd
    assert "insertions"    in bd
    assert bd["deletions"] >= 1    # "the" was deleted in first example


def test_error_analyser_per_domain():
    from src.evaluation.error_analysis import ErrorAnalyser

    refs    = ["hello world", "foo bar baz"]
    hyps    = ["hello world", "foo bar"]
    domains = ["domainA", "domainB"]

    analyser = ErrorAnalyser(refs, hyps, domains)
    per_domain = analyser.per_domain_wer()

    assert "domainA" in per_domain
    assert "domainB" in per_domain
    assert per_domain["domainA"]["wer"] == 0.0
    assert per_domain["domainB"]["wer"] > 0.0


def test_error_analyser_worst_examples():
    from src.evaluation.error_analysis import ErrorAnalyser

    refs = ["good transcription", "completely wrong output here"]
    hyps = ["good transcription", "xyz abc def ghi"]

    analyser = ErrorAnalyser(refs, hyps)
    worst = analyser.worst_examples(n=1)

    assert len(worst) == 1
    assert worst[0]["reference"] == "completely wrong output here"