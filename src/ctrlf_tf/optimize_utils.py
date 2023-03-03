"""Optimization functions."""

from collections import namedtuple
import copy
import math
from typing import Iterable, List
import sys

import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import ctrlf_tf.ctrlf_core
import ctrlf_tf.site_call_utils

IterationTuple = namedtuple("IterationTuple", ["direction",
                                               "start",
                                               "end",
                                               "model_gaps",
                                               "kmer_gap_limit",
                                               "auroc",
                                               "tpr_fpr_dataframe"])


def tpr_fpr_from_calls(max_scores: Iterable[float],
                       group_iterable: Iterable[str]) -> pd.DataFrame:
    """Generate TPR and FPR dataframe from called sites.

    Generates TPR and FPR data, returns dataframe with scores, true
    positive rates, and false positive rates.

    :param max_scores: Maximum threshold values for CtrlF-TF calls on a
        single sequence
    :type max_scores: float
    :param group_iterable: Iterable of +, -, and . groups
    :type group_iterable: Iterable[str]
    :returns: Pandas DataFrame of Scores, TPRs, and FPRs
    """
    true_positive_rates, false_positive_rates = [], []
    unique_scores = list(set(max_scores))
    unique_scores = sorted(unique_scores, reverse=True)
    query_dataframe = pd.DataFrame({"Score": max_scores,
                                    "Group": group_iterable})
    for score in unique_scores:
        true_positive = len(query_dataframe.query("Score >= @score & Group == '+'"))
        false_positive = len(query_dataframe.query("Score >= @score & Group == '-'"))
        true_negative = len(query_dataframe.query("Score < @score & Group == '-'"))
        false_negative = len(query_dataframe.query("Score < @score & Group == '+'"))
        true_positive_rate = (true_positive/(true_positive+false_negative))
        false_positive_rate = (false_positive/(true_negative+false_positive))
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)
    tpr_fpr_dataframe = pd.DataFrame({'Score': unique_scores,
                                      'TPR': true_positive_rates,
                                      'FPR': false_positive_rates})
    return tpr_fpr_dataframe


def estimate_true_positive_rate(tpr_fpr_dataframe: pd.DataFrame,
                                fpr_threshold: float) -> float:
    """Estimates TPR value based on linear interprolation

    Given a dataframe of true and false positive rates and a query
    false positive rate value, returns an estimated TPR based on the
    linear interprolation of the flanking values.

    :param tpr_fpr_dataframe: Dataframe of Scores, TPRs, and FPRs
    :type tpr_fpr_dataframe: pandas.DataFrame
    :returns: Estimated TPR at the FPR threshold
    """

    middle = tpr_fpr_dataframe[tpr_fpr_dataframe["FPR"] <= fpr_threshold]
    next_FPR = tpr_fpr_dataframe["FPR"][len(middle)]
    next_TPR = tpr_fpr_dataframe["TPR"][len(middle)]
    last_FPR = list(middle["FPR"])[-1]
    last_TPR = list(middle["TPR"])[-1]
    x = [last_FPR, next_FPR]
    y = [last_TPR, next_TPR]
    linear_interpolation = interp1d(x, y)
    estimated_tpr = float(linear_interpolation(fpr_threshold))
    return estimated_tpr


def shift_values(dataframe: pd.DataFrame,
                 column: str,
                 direction: str) -> np.ndarray:
    """Return an iterable of values as a shifted array."""
    if direction == "prior":
        shifted_values = dataframe[column].iloc[:-1]
    elif direction == "next":
        shifted_values = dataframe[column].iloc[1:]
    else:
        raise ValueError("Direction must be 'prior' or 'next'")
    shifted_values = shifted_values.reset_index(drop=True)
    shifted_array = np.array(list(shifted_values))
    return shifted_array


def meta_tpr_fpr_dataframe(tpr_fpr_dictionary):
    """Generate a dataframe made from tpr_fpr_dictionary.

    From each dataframe in the fpr_fpr_dictionary, creates a single
    dataframe with the ID indicated in an additioal column. Used for
    saving the data to a file

    :returns: Meta tpr_fpr_dataframe
    """
    dataframes = []
    for key in tpr_fpr_dictionary:
        dataframe = tpr_fpr_dictionary[key].copy()
        dataframe["ID"] = key
        dataframes.append(dataframe)
    meta_dataframe = pd.concat(dataframes)
    return meta_dataframe


def fpr_score_threshold(iterationtuple, fpr_threshold):
    """Return the score threshold at the FPR threshold."""
    dataframe = iterationtuple.tpr_fpr_dataframe
    dataframe = dataframe[dataframe["FPR"] <= fpr_threshold]
    if len(dataframe) == 0:
        return -math.inf
    min_score = min(dataframe["Score"])
    return min_score


def iterations_to_parameter_dataframe(iterations: Iterable[IterationTuple],
                                      fpr_threshold: float) -> pd.DataFrame:
    """Convert an iterable of IterationTuples to a dataframe."""
    tuple_list = []
    for index, i in enumerate(iterations):
        score_threshold = fpr_score_threshold(i, fpr_threshold)
        tuple_list.append((index,
                           i.kmer_gap_limit,
                           i.auroc,
                           score_threshold,
                           i.start,
                           i.end,
                           i.model_gaps))
    parameter_dataframe = pd.DataFrame(tuple_list)
    rename_dict = {0: "ID",
                   1: "Kmer_Gap_Limit",
                   2: "pAUROC",
                   3: "Score_Threshold",
                   4: "Core_Start",
                   5: "Core_End",
                   6: "Core_Gaps"}
    parameter_dataframe = parameter_dataframe.rename(columns=rename_dict)
    return parameter_dataframe


def iterations_to_tpr_fpr_dictionary(iterations) -> dict:
    """Convert an iterable of IterationTuples to a dictionary.

    Returns a dictoonary where the keys are the index positions of an
    iterable of IterationTuples and values are the tpr_fpr_dataframes from
    those tuples.

    :param iterations: Iterable of IterationTuples
    :returns: Dictionary of tpr_fpr_dataframes
    """
    tpr_fpr_dictionary = {}
    for index, i in enumerate(iterations):
        tpr_fpr_dictionary[index] = i.tpr_fpr_dataframe
    return tpr_fpr_dictionary


def find_best_iteration(iterations):
    """Return the IterationTuple with the max AUROC.

    Returns the IterationTuple with the max AUROC. If multiple
    IterationTuples have the same AUROC, prioritized by position in
    the list.

    :param iterations: Iterable of IterationTuples
    :returns: IterationTuple with the max AUROC
    """
    max_auroc = 0
    result = None
    for i in iterations:
        if i.auroc > max_auroc:
            max_auroc = i.auroc
            result = i
    return result



# def tpr_fpr_df_from_parameters(parameters, classified_df) -> float:
#     """AUROC from parameters.

#     Given a start, end, core_gaps, and gap_number, returns a
#     AUROC for the parameter set.
#     """
#     try:
#         ctrlf_obj = ctrlf_tf.ctrlf_core.CtrlF.from_parameters(parameters)
#     except:
#         print("Parameters could not be compiled", parameters, file=sys.stderr)
#         raise
#     scores = []
#     site_count = []
#     site_scores = []
#     site_list = []
#     for sequence in classified_df["Sequence"]:
#         sites = ctrlf_obj.call_sites(sequence, fixed_length=False)
#         site_count.append(len(sites))
#         if len(sites) == 0:
#             scores.append(-math.inf)
#             site_scores.append([-math.inf])
#             site_list.append([''])
#         else:
#             site_scores.append(sites)
#             max_score = -math.inf
#             this_site = []
#             for site in sites:
#                 this_site.append(sequence[site.start:site.end])
#                 if site.threshold > max_score:
#                     max_score = site.threshold
#             scores.append(max_score)
#             site_list.append(this_site)
#     classified_df["CtrlF_Threshold"] = scores
#     classified_df["Count"] = site_count
#     classified_df["Site_Calls"] = site_scores
#     classified_df["Sequence_Sites"] = site_list
#     tpr_fpr_df = tpr_fpr_from_calls(classified_df["CtrlF_Threshold"],
#                                     classified_df["Group"])
#     return tpr_fpr_df


def tpr_fpr_df_from_parameters(parameters, classified_df) -> float:
    """AUROC from parameters.

    Given a start, end, core_gaps, and gap_number, returns a
    AUROC for the parameter set.
    """
    try:
        ak_obj = ctrlf_tf.ctrlf_core.AlignedKmers.from_parameters(parameters)
    except:
        print("Parameters could not be compiled", parameters, file=sys.stderr)
        raise
    noncompiled_preprocess = ctrlf_tf.site_call_utils.noncompile_preprocessing_from_aligned_kmers(ak_obj.aligned_kmer_dataframe, ak_obj.core_positions)
    is_palindrome = parameters.palindrome
    scores = []
    site_count = []
    site_scores = []
    site_list = []
    for sequence in classified_df["Sequence"]:
        sites = ctrlf_tf.site_call_utils.call_sites_with_kmers(sequence, noncompiled_preprocess, is_palindrome)
        #sites = ctrlf_obj.call_sites(sequence, fixed_length=False)
        site_count.append(len(sites))
        if len(sites) == 0:
            scores.append(-math.inf)
            site_scores.append([-math.inf])
            site_list.append([''])
        else:
            site_scores.append(sites)
            max_score = -math.inf
            this_site = []
            for site in sites:
                this_site.append(sequence[site.start:site.end])
                if site.threshold > max_score:
                    max_score = site.threshold
            scores.append(max_score)
            site_list.append(this_site)
    classified_df["CtrlF_Threshold"] = scores
    classified_df["Count"] = site_count
    classified_df["Site_Calls"] = site_scores
    classified_df["Sequence_Sites"] = site_list
    tpr_fpr_df = tpr_fpr_from_calls(classified_df["CtrlF_Threshold"],
                                    classified_df["Group"])
    return tpr_fpr_df



def auroc_from_tpr_fpr(tpr_fpr_dataframe: pd.DataFrame,
                       fpr_threshold) -> float:
    """Calculate AUROC at FPR threshold from tpr_fpr_dataframe.

    Given a dataframe of TPR and FPR values per unique score, calculates
    the AUROC at the FPR threshold. The TPR at the FPR threshold is
    estimated using linear interpolation. The width of each rectangle is
    calculated by shifting the FPR values and subtracting each FPR value
    by the previous one. The height is the previous TPR values.

    :param tpr_fpr_dataframe: True and false positive rates with scores
    :returns: Partial AUROC at the FPR threshold value
    """
    # If FPR threshold below min FPR, return 0
    if min(tpr_fpr_dataframe["FPR"]) > fpr_threshold:
        return(0)
    # Estimate the tpr at the fpr threshold using linear interpolation
    tpr_threshold = estimate_true_positive_rate(tpr_fpr_dataframe, fpr_threshold)
    tpr_fpr_up_to_threshold = tpr_fpr_dataframe[(tpr_fpr_dataframe["FPR"] <=
                                                 fpr_threshold)]
    # Adjust tpr and fpr data to start with 0 and end with the threshold
    fpr_list = list(tpr_fpr_up_to_threshold["FPR"])
    tpr_list = list(tpr_fpr_up_to_threshold["TPR"])
    fpr_list = [0] + fpr_list + [fpr_threshold]
    tpr_list = [0] + tpr_list + [tpr_threshold]
    tpr_fpr_at_threshold = pd.DataFrame({"TPR": tpr_list,
                                         "FPR": fpr_list})
    # Take only the max FPR at each TPR
    tpr_fpr_at_threshold = tpr_fpr_at_threshold.groupby(by="FPR")
    tpr_fpr_at_threshold = tpr_fpr_at_threshold.aggregate(np.max)
    tpr_fpr_at_threshold = tpr_fpr_at_threshold.reset_index()
    # Shift values and convert to array
    fpr_prior = shift_values(tpr_fpr_at_threshold, "FPR", "prior")
    tpr_prior = shift_values(tpr_fpr_at_threshold, "TPR", "prior")
    fpr_next = shift_values(tpr_fpr_at_threshold, "FPR", "next")
    # Sum area of rectangles to calculate AUROC
    width_fpr = fpr_next - fpr_prior
    height_tpr = tpr_prior
    rectangle_areas = width_fpr * height_tpr
    auroc = sum(rectangle_areas)
    return auroc


def move_parameters_left(start, end, core_gaps):
    """Move AlignParameter settings left."""
    return (start - 1, end, [i + 1 for i in core_gaps])


def move_parameters_right(start, end, core_gaps):
    """Move AlignParameter settings right."""
    return (start, end + 1, core_gaps)


def update_core_parameters(alignparameters: ctrlf_tf.ctrlf_core.AlignParameters,
                           left: int,
                           right: int) -> ctrlf_tf.ctrlf_core.AlignParameters:
    """Return parameters modified for moving left and right a given amount."""
    start = alignparameters.core_start
    end = alignparameters.core_end
    core_gaps = alignparameters.core_gaps
    # Move left
    for i in range(left):
        start, end, core_gaps = move_parameters_left(start, end, core_gaps)
    for i in range(right):
        start, end, core_gaps = move_parameters_right(start, end, core_gaps)
    new_parameters = copy.deepcopy(alignparameters)
    new_parameters.core_start = start
    new_parameters.core_end = end
    new_parameters.core_gaps = core_gaps
    return new_parameters


def local_optimization_search(parameters: ctrlf_tf.ctrlf_core.AlignParameters,
                              left: int,
                              right: int,
                              seen_lr: set,
                              classified_df: pd.DataFrame,
                              fpr_threshold: float) -> List[IterationTuple]:
    """Performs a local search of optimization parameters.

    Given an AlignParameters object, the left and right position relative to the
    parameters, a set of seen positions, a dataframe to run the classification task
    on, and a fpr threshold, calls sites and benchmarks on neighboring parameters and
    returns the results as a list of IterationTuples.
    """
    left_search = [left + 1]
    right_search = [right + 1]
    if parameters.palindrome is False:
        left_search = left_search + [left + 1, left]
        right_search = right_search + [right, right + 1]
    iterations = []
    for l, r in zip(left_search, right_search):
        if (l, r) not in seen_lr:
            ext_params = update_core_parameters(parameters, l, r)
            ext_tpr_fpr_df = tpr_fpr_df_from_parameters(ext_params, classified_df)
            ext_auroc = auroc_from_tpr_fpr(ext_tpr_fpr_df, fpr_threshold)
            seen_lr.add((l, r))
            ext_name = "L" * l + "R" * r
            iterations.append(IterationTuple(ext_name,
                                             ext_params.core_start,
                                             ext_params.core_end,
                                             ext_params.core_gaps,
                                             ext_params.gap_limit,
                                             ext_auroc,
                                             ext_tpr_fpr_df))
    return iterations


def optimize_gap_parameters(gap_limit: int,
                            fpr_threshold: float,
                            threshold_value: float,
                            classified_df: pd.DataFrame,
                            init_parameters: ctrlf_tf.ctrlf_core.AlignParameters) -> IterationTuple:
    """Optimize for a single gap limit parameter"""
    # Determine initial AUROC and parameters
    gap_alignparams = copy.deepcopy(init_parameters)
    gap_alignparams.gap_limit = gap_limit
    gap_alignparams.threshold = threshold_value
    initial_tpr_fpr_df = tpr_fpr_df_from_parameters(gap_alignparams, classified_df.copy(deep=True))
    initial_auroc = auroc_from_tpr_fpr(initial_tpr_fpr_df, fpr_threshold)
    current_iteration = IterationTuple("Initial",
                                       gap_alignparams.core_start,
                                       gap_alignparams.core_end,
                                       gap_alignparams.core_gaps,
                                       gap_limit,
                                       initial_auroc,
                                       initial_tpr_fpr_df)
    iteration_list = [current_iteration]
    seen_parameters = set()
    left = 0
    right = 0
    # Keep extending the core model until AUROC does not increase
    not_optimal = True
    while not_optimal:
        iteration_list += local_optimization_search(gap_alignparams, left, right, seen_parameters, classified_df.copy(deep=True), fpr_threshold)
        best_iteration = find_best_iteration(iteration_list)
        if best_iteration == current_iteration:
            not_optimal = False
        else:
            current_iteration = best_iteration
            left = current_iteration.direction.count('L')
            right = current_iteration.direction.count("R")
    return iteration_list


def optimize_parameters(gap_limit: int,
                        fpr_threshold: float,
                        threshold_value_dict: dict,
                        classified_df: pd.DataFrame,
                        init_parameters: ctrlf_tf.ctrlf_core.AlignParameters):
    """Optimize parameters over multiple gaps."""
    all_iterations = []
    # For each gap choice
    for gap in range(gap_limit + 1):
        all_iterations += optimize_gap_parameters(gap,
                                                  fpr_threshold,
                                                  threshold_value_dict[gap],
                                                  classified_df,
                                                  init_parameters)
    all_iterations = tuple(all_iterations)
    parameter_dataframe = iterations_to_parameter_dataframe(all_iterations, fpr_threshold)
    tpr_fpr_dictionary = iterations_to_tpr_fpr_dictionary(all_iterations)
    return (parameter_dataframe, tpr_fpr_dictionary)


def optimal_parameters_from_df(parameter_dataframe: pd.DataFrame,
                               init_params: ctrlf_tf.ctrlf_core.AlignParameters) -> ctrlf_tf.ctrlf_core.AlignParameters:
    """Return optimal AlignParameters from a parameter dataframe."""
    # Find the row with the best performance
    index_max_auroc = parameter_dataframe["pAUROC"].idxmax()
    # Get the information from that row
    core_start = parameter_dataframe.iloc[index_max_auroc]["Core_Start"]
    core_end = parameter_dataframe.iloc[index_max_auroc]["Core_End"]
    core_gaps = parameter_dataframe.iloc[index_max_auroc]["Core_Gaps"]
    gap_limit = parameter_dataframe.iloc[index_max_auroc]["Kmer_Gap_Limit"]
    threshold = parameter_dataframe.iloc[index_max_auroc]["Score_Threshold"]
    # Update a copy of the initial parameters and return it
    result = copy.deepcopy(init_params)
    result.range_consensus = None
    result.core_start = core_start
    result.core_end = core_end
    result.core_gaps = core_gaps
    result.gap_limit = gap_limit
    result.threshold = threshold
    return result
