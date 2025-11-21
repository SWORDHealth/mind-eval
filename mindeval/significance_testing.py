from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

SegmentScores = pd.DataFrame
ResamplingScores = pd.DataFrame
HeadToHeads = pd.DataFrame


def rank_systems(
    seg_scores: SegmentScores,
    threshold_p_value: float = 0.05,
    bootstrap_resampling: bool = False,
    sample_size: int = 100,
    num_splits: int = 100,
    tie_epsilon: float = 0.01,
) -> pd.Series:
    """Rank systems using average scores."""
    scores = seg_scores
    if bootstrap_resampling:
        scores = bootstrap_resampling(seg_scores, sample_size, num_splits)

    head_to_heads = build_head_to_heads(scores, tie_epsilon)
    clusters = build_clusters(seg_scores, head_to_heads, threshold_p_value)

    result = pd.Series(index=scores.columns)
    for i, cluster in enumerate(clusters):
        for system in cluster:
            result.loc[system] = i + 1

    return result


def bootstrap_resampling(
    seg_scores: SegmentScores, sample_size: int, num_splits: int
) -> ResamplingScores:
    """Computes Bootstrap Resampling.

    seg_scores: scores for each systems' translation, aka a score matrix
        [num_sentences x num_systems]
    sample_size: Number of examples for each partition.
    num_splits: Number of partitions.

    Return:
        Scores matrix for each system [num_splits x num_systems]
    """
    seg_scores_values = seg_scores.values.T  # num_systems x num_sentences
    population_size = seg_scores_values.shape[1]
    # Subsample the gold and system outputs (with replacement)
    subsample_ids = np.random.choice(
        population_size, size=(sample_size, num_splits), replace=True
    )
    subsamples = np.take(
        seg_scores_values, subsample_ids, axis=1
    )  # num_systems x sample_size x num_splits
    resample_scores_values = np.mean(subsamples, axis=1)  # num_systems x num_splits
    return pd.DataFrame(resample_scores_values.T, columns=seg_scores.columns)


@dataclass
class HeadToHead:
    ties: int
    wins: int
    losses: int
    p_value: float

    def __str__(self) -> str:
        return f"p-value: {self.p_value:.3f} ties: {self.ties} wins: {self.wins} losses: {self.losses}"


def build_head_to_heads(scores, eps=0.01) -> HeadToHeads:
    """Compare systems using resampling scores."""

    systems = scores.columns
    comparisons = pd.DataFrame(index=systems, columns=systems, dtype=object)

    for system1, system2 in product(systems, repeat=2):
        if system1 == system2:
            comparisons[system1][system1] = None
            continue
        delta = scores[system1] - scores[system2]
        ties = (delta.abs() < eps).sum()
        wins = (delta >= eps).sum()
        losses = (delta <= -eps).sum()
        ttest_result = stats.ttest_rel(
            scores[system1], scores[system2], alternative="greater"
        )
        comparisons.loc[system1, system2] = HeadToHead(
            ties, wins, losses, ttest_result.pvalue
        )

    return comparisons


def build_clusters(
    seg_scores: SegmentScores,
    head_to_heads: HeadToHeads,
    threshold_p_value: float = 0.05,
):
    avg_scores = seg_scores.mean()
    p_values = head_to_heads.map(lambda x: x.p_value if x else np.nan)
    diffs = p_values.map(lambda x: x < threshold_p_value)
    clusters = [[]]
    descending_systems = avg_scores.sort_values(ascending=False).index
    for system in descending_systems:
        curr_cluster = clusters[-1]
        if any(diffs.loc[curr_cluster, system]):
            clusters.append([system])
        else:
            curr_cluster.append(system)
    return clusters
