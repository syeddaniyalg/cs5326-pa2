from alignment.files import read_jsonl
from alignment.judge import build_judge_prompt, judge_pairs, parse_winners, request_batch, resolve_winner
from alignment.pairs import build_pair_rows, candidate_and_baseline
from alignment.prompts import clean_response, format_prompt
from alignment.stats import binomial_two_sided_p, wilson_interval
from alignment.text_stats import is_refusal, response_stats


def make_rows(prefix, count):
    return [{"id": f"p{index}", "instruction": f"question {index}", "output": f"{prefix} {index}", "category": "c"} for index in range(count)]


def test_parse_winners_reads_a_json_list():
    text = 'Here you go: [{"item": 1, "winner": "A"}, {"item": 2, "winner": "tie"}, {"item": 3, "winner": "b"}]'
    assert parse_winners(text, 3) == ["A", "tie", "B"]


def test_parse_winners_marks_missing_items_unknown():
    text = '[{"item": 2, "winner": "B"}]'
    assert parse_winners(text, 3) == ["unknown", "B", "unknown"]


def test_parse_winners_recovers_from_broken_json():
    text = '{"item": 1, "winner": "A"}, {"item": 2, "winner": '
    assert parse_winners(text, 2) == ["A", "unknown"]


def test_request_batch_retries_partial_responses():
    responses = iter(
        [
            '[{"item": 1, "winner": "A"}]',
            '[{"item": 1, "winner": "A"}, {"item": 2, "winner": "B"}]',
        ]
    )

    labels, attempts, error = request_batch(lambda _: next(responses), "prompt", 2, max_retries=3, sleep_seconds=0)

    assert labels == ["A", "B"]
    assert attempts == 2
    assert error is None


def test_request_batch_combines_complementary_partial_responses():
    responses = iter(
        [
            '[{"item": 1, "winner": "A"}]',
            '[{"item": 2, "winner": "tie"}]',
        ]
    )

    labels, attempts, error = request_batch(lambda _: next(responses), "prompt", 2, max_retries=2, sleep_seconds=0)

    assert labels == ["A", "tie"]
    assert attempts == 2
    assert error is None


def test_unresolved_partial_judgments_are_logged_as_failures(tmp_path):
    pairs = build_pair_rows(make_rows("dpo", 2), make_rows("sft", 2), "dpo", "sft", seed=42)
    judged_path = tmp_path / "judged.jsonl"
    failure_path = tmp_path / "failures.jsonl"

    judged = judge_pairs(
        pairs,
        lambda _: '[{"item": 1, "winner": "A"}]',
        "general",
        "toy-judge",
        batch_size=2,
        sleep_seconds=0,
        max_retries=2,
        max_chars=500,
        judged_path=judged_path,
        failure_path=failure_path,
        reuse=False,
    )

    assert set(judged) == {pairs[0]["pair_id"]}
    failures = read_jsonl(failure_path)
    assert [row["pair_id"] for row in failures] == [pairs[1]["pair_id"]]
    assert "parsed 1/2" in failures[0]["error"]


def test_pairs_are_balanced_and_deterministic():
    candidate, baseline = make_rows("dpo", 10), make_rows("sft", 10)
    first = build_pair_rows(candidate, baseline, "dpo", "sft", seed=42)
    second = build_pair_rows(candidate, baseline, "dpo", "sft", seed=42)

    assert first == second
    assert sum(pair["a_source"] == "candidate" for pair in first) == 5


def test_candidate_and_baseline_recover_the_original_outputs():
    candidate, baseline = make_rows("dpo", 6), make_rows("sft", 6)
    pairs = build_pair_rows(candidate, baseline, "dpo", "sft", seed=7)

    for pair, original in zip(pairs, candidate):
        recovered_candidate, recovered_baseline = candidate_and_baseline(pair)
        assert recovered_candidate == original["output"]
        assert recovered_baseline.startswith("sft")


def test_judge_prompt_never_mentions_model_identity():
    candidate, baseline = make_rows("dpo", 2), make_rows("sft", 2)
    pairs = build_pair_rows(candidate, baseline, "my-dpo-model", "my-sft-model", seed=1)
    prompt = build_judge_prompt(pairs, "general", max_chars=500)

    assert "my-dpo-model" not in prompt
    assert "my-sft-model" not in prompt
    assert "candidate" not in prompt.lower()
    assert "baseline" not in prompt.lower()
    assert "Response A" in prompt and "Response B" in prompt


def test_resolve_winner_translates_positions_back_to_models():
    pair = {"a_source": "baseline", "b_source": "candidate"}
    assert resolve_winner(pair, "A") == "baseline"
    assert resolve_winner(pair, "B") == "candidate"
    assert resolve_winner(pair, "tie") == "tie"


def test_wilson_interval_contains_the_point_estimate():
    low, high = wilson_interval(60, 100)
    assert low < 0.6 < high
    assert 0.0 <= low and high <= 1.0


def test_sign_test_is_one_for_a_perfect_split():
    assert binomial_two_sided_p(5, 10) == 1.0
    assert binomial_two_sided_p(10, 10) < 0.01


def test_prompt_template_and_cleaning():
    assert format_prompt("  hi ").endswith("### Response:\n")
    assert clean_response("answer\n### Instruction:\nnext") == "answer"


def test_text_statistics():
    assert is_refusal("I’m sorry, but I can’t help with that.")
    assert not is_refusal("Here is a recipe.")

    stats = response_stats("STOP this NOW!")
    assert stats["words"] == 3
    assert stats["caps_words_per_100_words"] > 0
    assert stats["exclamations_per_100_words"] > 0
