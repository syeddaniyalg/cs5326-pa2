import random


def build_pair_rows(candidate_rows, baseline_rows, candidate_name, baseline_name, seed):
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    matched = [row for row in candidate_rows if row["id"] in baseline_by_id]

    candidate_first = [index % 2 == 0 for index in range(len(matched))]
    random.Random(seed).shuffle(candidate_first)

    pairs = []
    for candidate, put_first in zip(matched, candidate_first):
        baseline = baseline_by_id[candidate["id"]]
        if put_first:
            response_a, response_b = candidate["output"], baseline["output"]
            a_source, b_source = "candidate", "baseline"
        else:
            response_a, response_b = baseline["output"], candidate["output"]
            a_source, b_source = "baseline", "candidate"

        pair = {
            "pair_id": candidate["id"],
            "id": candidate["id"],
            "instruction": candidate["instruction"],
            "category": candidate.get("category", "all"),
            "dataset": candidate.get("dataset", ""),
            "response_a": response_a,
            "response_b": response_b,
            "a_source": a_source,
            "b_source": b_source,
            "candidate_name": candidate_name,
            "baseline_name": baseline_name,
        }
        pairs.append(pair)
    return pairs


def candidate_and_baseline(pair):
    if pair["a_source"] == "candidate":
        return pair["response_a"], pair["response_b"]
    return pair["response_b"], pair["response_a"]
