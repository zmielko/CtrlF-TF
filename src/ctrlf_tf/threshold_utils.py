"""Functions for classifying thresholds."""

from collections import namedtuple
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ThresholdTuple = namedtuple("ThresholdTuple", ["definition", "value"])


def mod_zscore_median(values: Iterable[float]) -> Tuple[float]:
    """Return modified z-scores for an iterable of values.

    Given an iterable set of values, returns a tuple of modifed z-scores
    based on the median.

    :param values: Numerical values to calculate the modified z-scores of
    :type values: Iterable[float]
    :returns: Tuple of modified z-scores
    """
    median_value = np.median(values)
    mad = stats.median_abs_deviation(values)
    zscore_list = []
    for value in values:
        zscore = (value - median_value) / (1.4826 * mad)
        zscore_list.append(zscore)
    zscore_tuple = tuple(zscore_list)
    return zscore_tuple


def threshold_from_zscore(values: Iterable[float],
                          zscore_threshold: Iterable[float]) \
                          -> ThresholdTuple:
    """Calculate threshold based of z-score threshold."""
    zscore_df = pd.DataFrame({"Value": values,
                             "Zscore": mod_zscore_median(values)})
    threshold = max(zscore_df[zscore_df["Zscore"] < zscore_threshold]["Value"])
    result = ThresholdTuple(f"Modified Z-score = {zscore_threshold}",
                            threshold)
    return result


def kde_dataframe_from_values(values: Iterable[int]):
    """Calculate kernal density estimate from values.

    Given an array of signal for debruijn sequences, returns the threshold
    for fliping the left tail around the maximum kde value for all signals.
    """
    # Calculate kernal density estimate for all values
    kde = stats.gaussian_kde(values)
    kde_values = kde.pdf(np.array(values))
    kde_df = pd.DataFrame({"kde": kde_values, "value": values})
    return kde_df


def left_tail_flip_threshold(values: Iterable[int]) -> float:
    """Threshold from taking the left tail and flipping it to the right.

    Given an iterable of numeric values, calculates the kernel density
    estimate function using a gaussian distribution. From the input numeric
    values, inputs them into the kde function and returns the value that
    gave the maximum output as the middle value. Returns the result from
    adding the minimum value to twice the distance to the middle value
    as the threshold.

    :param values: Values to calculate a kde from
    :type values: Iterable[float]
    :returns: The threshold value
    """
    # Calculate kernal density estimate for all values
    kde = stats.gaussian_kde(values)
    kde_values = kde.pdf(np.array(values))
    kde_dataframe = pd.DataFrame({"kde": kde_values, "value": values})
    # Get the maximum kde value, the mode is the corresponding signal
    max_kde = kde_dataframe[kde_dataframe["kde"] == np.max(kde_dataframe["kde"])]
    max_kde = max_kde.sort_values(by="value").reset_index(drop=True)
    middle = max_kde["value"][0]
    # Threshold is the mode + minimum value
    minimum = np.min(kde_dataframe["value"])
    threshold = middle + (middle - minimum)
    return threshold


def thresholds_kde_zscore(values: Iterable[float]) -> Tuple[float, float]:
    """Threshold based on KDE and a modified zscore of 4.

    Given an iterable of values, calculates kde max + (kdx max - min value)
    to determine one threshold and a modified zscore of 4 as another. The
    minimum of those values is the negative threshold and the maximum of
    those values is the positive threshold.

    :param values: Iterable of values to calculate thresholds from
    :returns: A tuple of negative and positive threshold values
    """
    # Calculate Z-score and KDE thresholds
    zscore_threshold = threshold_from_zscore(values, 4)
    kde_threshold = left_tail_flip_threshold(values)
    # Assign them to the min and max threshold
    negative = min([zscore_threshold.value, kde_threshold])
    positive = max([zscore_threshold.value, kde_threshold])
    # Create threshold tuples for the output
    if positive == kde_threshold:
        positive_threshold = ThresholdTuple("Kernal Density Estimate",
                                            positive)
        negative_threshold = ThresholdTuple("Modified Z-score = 4",
                                            negative)
    else:
        positive_threshold = ThresholdTuple("Modified Z-score = 4",
                                            positive)
        negative_threshold = ThresholdTuple("Kernal Density Estimate",
                                            negative)
    return (negative_threshold, positive_threshold)


def threshold_from_kde(values: Iterable[float], positive_ratio: float) -> Tuple[float, float]:
    """Threshold based on the distance from the max KDE to the left tail

    Given an iterable of values, calculates kde max + (kdx max - min value)
    to determine the negative threshold. The positive threshold is a ratio
    from the negative threshold that is 1 or more to ensure the positive
    threshold is larger than the negative threshold.

    :param values: Iterable of values to calculate thresholds from
    :param positive_ratio: Ratio of positive threshold from negative threshold
    :returns: A tuple of negative and positive threshold values
    """
    # Calculate thresholds for negative and positive values
    if positive_ratio < 1:
        raise ValueError(f"positive ratio is {positive_ratio}, must be 1 or greater.")
    negative = left_tail_flip_threshold(values)
    positive = negative * positive_ratio
    negative_threshold = ThresholdTuple("Kernal Density Estimate",
                                        negative)
    positive_threshold = ThresholdTuple(f"{positive_ratio} x KDE",
                                        positive)
    return (negative_threshold, positive_threshold)


def classify_values(values: Iterable[float],
                    negative_threshold: float,
                    positive_threshold: float) -> Tuple[str]:
    """Classify values based on positive and negative thresholds.

    Given values, a positive threshold, and a negative threshold, Returns
    a tuple with '+', '-', or '.' groups.

    :param values: Iterable of values to classify
    :param negative_threshold: Threshold value below which sequences are
        not bound
    :param positive_threshold: Threshold value above which sequences are
        bound
    :returns: Tuple the length of values with classifications for each
        value
    """
    classification_list = []
    for i in values:
        if i > positive_threshold:
            classification_list.append('+')
        elif i < negative_threshold:
            classification_list.append('-')
        else:
            classification_list.append('.')
    classification_tuple = tuple(classification_list)
    return classification_tuple
