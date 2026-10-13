import math


def wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return 0.0, 0.0

    proportion = successes / trials
    denominator = 1 + z ** 2 / trials
    center = (proportion + z ** 2 / (2 * trials)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def binomial_two_sided_p(successes, trials):
    if trials == 0:
        return 1.0

    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / 2 ** trials
    return min(1.0, 2 * probability)
