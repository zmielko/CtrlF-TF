"""Functions for compiling aligned, ranked k-mers into binding sites
"""

import math
from typing import Iterable, List

import numpy as np
import pandas as pd
import ctrlf_tf.str_utils

COMPILED_LABEL = "Aligned_Sequences"


def bounds_from_aligned_positions(aligned_positions: Iterable[int],
                                  max_word_len: int) -> tuple:
    """Returns a left and right bound integer as a tuple."""
    left_bound = min(aligned_positions)
    right_bound = max(aligned_positions) + max_word_len
    return (left_bound, right_bound)


def sequence_position(sequence):
    # Initialize parameters
    left = -1
    right = len(sequence) - 1
    found_sequence=False
    for idx, letter in enumerate(sequence):
        if letter != '.' and found_sequence is False:
            left = idx
            found_sequence = True
        elif letter == '.' and found_sequence:
            right = idx
            break
    if left == -1:
        raise ValueError("Input string is not in the form of characters with surrounding wildcards.")
    return (left, right)


def bounds_from_aligned_sequences(sequences):
    left = math.inf
    right = -math.inf
    for sequence in sequences:
        curr_left, curr_right = sequence_position(sequence)
        if curr_left < left:
            left = curr_left
        if curr_right > right:
            right = curr_right
    return (left, right)


def k_desc(kmer: str, ap: int, left_bound: int, right_bound: int) -> List[int]:
    """Returns a binary int list for a k-mers description. """
    left_pad = [0] * abs(left_bound - ap)
    right_pad = [0] * abs(right_bound - (ap + len(kmer)))
    kmer_isletter_list = [0 if i == '.' else 1 for i in kmer]
    return left_pad + kmer_isletter_list + right_pad


def calculate_core_position_array(core_positions, left_bound, right_bound):
    """Calculate a bounded 1D array with 1 for core positions, otherwise 0"""
    core_position_array = np.zeros(abs(left_bound - right_bound))
    for i in core_positions:
        core_position_array[i + abs(left_bound)] = 1
    return core_position_array


def description_matrix_from_kmers(kmers: Iterable[str],
                                  align_positions: Iterable[int],
                                  core_position_array: np.array,
                                  left_bound: int,
                                  right_bound: int) -> np.array:
    """Generate a matrix for which positions in the core k-mers describe."""
    description_lists = []
    for kmer, align_position in zip(kmers, align_positions):
        description_row = k_desc(kmer, align_position, left_bound, right_bound)
        description_lists.append(description_row)
    description_matrix = np.array(description_lists) * core_position_array
    description_matrix = description_matrix[:, np.nonzero(core_position_array)[0].tolist()]
    return description_matrix


def pad_k(kmer, align_position, left_bound, right_bound):
    """Returns k-mer strings in an aligned space."""
    left_pad = '.' * abs(left_bound - align_position)
    right_pad = '.' * abs(right_bound - (align_position + len(kmer)))
    return left_pad + kmer + right_pad


def is_core_described(core_descriptions) -> bool:
    """Check if a core description array can be considered called.

    Given a core description iterable, returns True if all core positions are
    described and N-2 of them are described twice. Otherwise, returns False.

    :param core_description_iterable: Iterable of numpy arrays of equal
        length that describe the core.
    :returns: True if the core is describe, else false
    """
    total_core_description = np.sum(core_descriptions, axis=0)
    one_count_positions = np.count_nonzero(total_core_description == 1)
    zero_count_positions = np.count_nonzero(total_core_description == 0)
    return one_count_positions <= 2 and zero_count_positions == 0


def left_bound_solution(kmer_idxs, description_matrix: np.array):
    """Finds the left-bound solutions from an ordered iterable of k-mers."""
    rev_idxs = kmer_idxs[::-1]
    for end_idx in range(2, len(rev_idxs) + 1):
        if is_core_described(description_matrix[rev_idxs[:end_idx], :]):
            return rev_idxs[:end_idx]
    raise ValueError(f"Indicies with no solution given as input: {kmer_idxs}")


def get_align_position(kmer: str, wildcard: str ='.') -> int:
    """Relative k-mer position in aligned strings."""
    for idx, i in enumerate(kmer):
        if i != wildcard:
            return idx
    return -1


def expand_kmer(kmer: str) -> List[str]:
    """Given a k-mer with wildcards as '.' returns all non-gapped sequences."""
    results = []

    def recurse_expand(fullword, result, idx):
        if idx == len(fullword):
            results.append(result)
            return
        if fullword[idx] == '.':
            for i in ["A", "C", "G", "T"]:
                recurse_expand(fullword, result + i, idx + 1)
        else:
            recurse_expand(fullword, result + fullword[idx], idx + 1)
    recurse_expand(kmer, '', 0)
    return results


def expand_kmer_maintain_pos(kmer: str, ap: int, r: int, total_len: int):
    """Returns expanded k-mers in the same aligned position."""
    expanded_kmers = expand_kmer(kmer[ap:r])
    return list(map(lambda x: ('.' * ap) + x + ('.' * (total_len - r)), expanded_kmers))


def merge_kmers(kmer_idxs: Iterable[int], kmer_dict: dict) -> str:
    """Projects k-mers onto a single string."""
    kmer_list = [kmer_dict[i] for i in kmer_idxs]
    consensus_sequence = ""
    for idx, letter in enumerate(kmer_list[0]):
        addition = '.'
        if letter != '.':
            addition = letter
        else:
            for kmer in kmer_list[1:]:
                if kmer[idx] != '.':
                    addition = kmer[idx]
                    break
        consensus_sequence += addition
    return consensus_sequence


def min_kmer_score(kmer_idxs: Iterable[int], score_dict: dict) -> int:
    return min([score_dict[i] for i in kmer_idxs])


def is_subset(str_a, str_b):
    """Returns True if str_a is a subset of str_b, otherwise False."""
    for idx in range(len(str_a)):
        if (str_a[idx] != ".") and (str_a[idx] != str_b[idx]):
            return False
    return True


def create_traverse_compatibility_matricies(padded_kmers: Iterable[str]) -> tuple:
    """

    Returns a traversal and compatibility matrix as a tuple.
    Traversal matrix: Binary adjacency matrix where 1 represents a hamming
    distance threshold
    Compatibility matrix: Binary adjacency matrix where 1 represents projection compatibility
    """
    # Generate COMPAT and TRAVERSE matricies
    traverse_matrix = []
    compatibility_matrix = []
    # Generate matricies (naive)
    for i in padded_kmers:
        t_row, c_row = [], []
        for j in padded_kmers:
            if ctrlf_tf.str_utils.hamming_distance(i, j) <= 2:
                t_row.append(1)
            else:
                t_row.append(0)
            if ctrlf_tf.str_utils.compatible_description(i, j):
                c_row.append(1)
            else:
                c_row.append(0)
        traverse_matrix.append(t_row)
        compatibility_matrix.append(c_row)
    traverse_matrix = np.array(traverse_matrix)
    compatibility_matrix = np.array(compatibility_matrix)
    return (traverse_matrix, compatibility_matrix)


def solve_kmer(kmer_idx,
               traverse_matrix,
               compatible_matrix,
               prior_compatibility=None,
               seen_idxs=None,
               description_matrix=None,
               result=None):
    """Returns minimal solutions for k-mers recursively.

    Given a kmer index, adjacency matricies for traversal and compatibility,
    and a result list to append values to, recursively searches paths for a
    k-mer and returns the minimal solution for each path.
    """
    if prior_compatibility is None:  # Initialize
        prior_compatibility = np.ones(traverse_matrix.shape[0])
        seen_idxs = [kmer_idx]
    # First, zero out the kmer_idx column to prevent backtravel
    traverse_matrix[:, kmer_idx] = 0
    # First, merge previous compatibility with current
    compatible_array = compatible_matrix[kmer_idx] * prior_compatibility
    # Then merge the compatibility with the traverse array
    traverse_array = traverse_matrix[kmer_idx] * compatible_array
    # Find all traversable kmer indexes
    traverseable_idxs = np.nonzero(traverse_array)[0]
    # If the solution is optimal, return answer
    if is_core_described(description_matrix[seen_idxs, :]):
        result.append(left_bound_solution(seen_idxs, description_matrix))
        return
    # If there are no more traversable indicies, stop
    if len(traverseable_idxs) == 0:
        return
    # For each node in traversable, repeat
    for next_idx in traverseable_idxs:
        solve_kmer(next_idx,
                   traverse_matrix,
                   compatible_matrix,
                   compatible_array,
                   seen_idxs + [next_idx],
                   description_matrix,
                   result)


def solve_matricies(kmer_idx_dict: dict,
                           traverse_m: np.array,
                           compat_m: np.array,
                           description_m: np.array) -> set:
    """Returns all possible KOMPAS solutions."""
    compiled_ans = set()
    for kmer_idx in kmer_idx_dict:
        result = []
        solve_kmer(kmer_idx,
                   traverse_m.copy(),
                   compat_m.copy(),
                   None,
                   None,
                   description_m,
                   result)
        for i in result:
            compiled_ans.add(tuple(i))
    if None in compiled_ans:
        compiled_ans.remove(None)
    return compiled_ans


def compile_solutions(solutions: set,
                      kmer_idx_dict: dict,
                      kmer_score_dict: dict) -> pd.DataFrame:
    output_set = set()
    for kmer_group in list(solutions):
        consensus = merge_kmers(kmer_group, kmer_idx_dict)
        score = min_kmer_score(kmer_group, kmer_score_dict)
        output_set.add((consensus, score, kmer_group))
    output_df = pd.DataFrame(output_set)
    output_df = output_df.rename(columns={0: COMPILED_LABEL,
                                          1: "Score",
                                          2: "Kmer_Idxs"})
    output_df = output_df.sort_values(by="Score", ascending=False)
    output_df = output_df.reset_index(drop=True)
    # Drop duplicate answers
    output_df = output_df.drop_duplicates(subset=COMPILED_LABEL,
                                          keep="first").reset_index(drop=True)
    return output_df


def expand_gapped_consensus(consensus_sites, scores, kmer_idxs):
    LEN = len(consensus_sites[0])
    output_exp = []
    for i, score, kmeridxs in zip(consensus_sites, scores, kmer_idxs):
        ap = get_align_position(i)
        end = len(i.rstrip('.'))
        results = expand_kmer_maintain_pos(i, ap,end, LEN)
        word_len = len(i.strip('.'))
        for j in results:
            output_exp.append((j, ap, score, kmeridxs, word_len))
    output_exp_df = pd.DataFrame(output_exp)
    output_exp_df = output_exp_df.rename(columns={0: COMPILED_LABEL,
                                                  1: "Core_Position",
                                                  2: "Rank_Score",
                                                  3: "Kmer_Idxs",
                                                  4: "Word_Len"})
    output_exp_df = output_exp_df.sort_values(by=["Rank_Score", "Word_Len"], ascending=[False, True])
    output_exp_df = output_exp_df.reset_index(drop=True)
    # Drop duplicates after expansion
    output_exp_df = output_exp_df.drop_duplicates(subset=COMPILED_LABEL,
                                                  keep="first")
    output_exp_df = output_exp_df.reset_index(drop=True)
    return output_exp_df


def determine_optimal_consensus_sites(sorted_consensus: Iterable[str]) -> set:
    """Filter non-optimal answers from an iterable of answers.

    Given an iterable of sorted consensus sites, returns a set of optimal
    answers.
    """
    optimal_ans = []
    for idx, solution_query in enumerate(sorted_consensus):
        is_opt = True
        for optimal_solution in optimal_ans:
            if is_subset(optimal_solution, solution_query):
                is_opt = False
                break
        if is_opt:
            optimal_ans.append(solution_query)
    return set(optimal_ans)



def bounds_from_aligned_kmer_df(kmer_dataframe):
    max_word_len = max([len(kmer) for kmer in kmer_dataframe.iloc[:, 0]])
    aligned_positions = kmer_dataframe.iloc[:, 1]
    left_bound, right_bound = bounds_from_aligned_positions(aligned_positions, max_word_len)
    return (left_bound, right_bound)


def relative_consensus_df_from_abs(input_df, abs_core_start):
    df = input_df.copy(deep=True)
    df["Align_Position"] = df[COMPILED_LABEL].apply(lambda x: get_align_position(x))
    df[COMPILED_LABEL] = df[COMPILED_LABEL].apply(lambda x: x.strip('.'))
    df["Core_Position"] = abs_core_start - df["Align_Position"] - 1
    df = df[[COMPILED_LABEL, "Core_Position", "Rank_Score"]]
    return df


def compile_consensus_sites(kmer_dataframe, core_positions):
    kmer_dataframe = kmer_dataframe.drop_duplicates().reset_index(drop=True)
    # Calculate bounds
    kmers = kmer_dataframe.iloc[:, 0]
    max_word_len = max([len(kmer) for kmer in kmer_dataframe.iloc[:, 0]])
    aligned_positions = kmer_dataframe.iloc[:, 1]
    left_bound, right_bound = bounds_from_aligned_positions(aligned_positions, max_word_len)
    # Pad k-mers relative to bounds
    padded_kmers = []
    for row in kmer_dataframe.itertuples():
        padded_kmer = pad_k(row.Kmer, row.Align_Position, left_bound, right_bound)
        padded_kmers.append(padded_kmer)
    #Create Matricies
    traverse_m, compat_m = create_traverse_compatibility_matricies(padded_kmers)
    core_position_array = calculate_core_position_array(core_positions, left_bound, right_bound)
    description_m = description_matrix_from_kmers(kmers,
                                                  aligned_positions,
                                                  core_position_array,
                                                  left_bound,
                                                  right_bound)
    # Create dictionaries for fast reference
    score_dict = dict(zip(range(len(kmer_dataframe["Rank_Score"])),
                          kmer_dataframe["Rank_Score"]))
    kmer_dict = dict(zip(range(len(padded_kmers)), padded_kmers))
    # Solve  matricies, returns set of tuples with kmer idxs required for a solution
    compiled_ans = solve_matricies(kmer_dict, traverse_m, compat_m, description_m)
    # Compile solutions into consensus sites
    output_df = compile_solutions(compiled_ans, kmer_dict, score_dict)
    # Expand gaps to create consensus sequences with no gaps
    output_exp_df = expand_gapped_consensus(output_df[COMPILED_LABEL],
                                            output_df["Score"],
                                            output_df["Kmer_Idxs"])
    optimal_ans = determine_optimal_consensus_sites(output_exp_df[COMPILED_LABEL])
    output_exp_df = output_exp_df[output_exp_df[COMPILED_LABEL].isin(optimal_ans)]
    output_exp_df = output_exp_df.sort_values(by=["Rank_Score", COMPILED_LABEL], ascending=False)
    output_exp_df = output_exp_df.reset_index(drop=True)
    return output_exp_df[[COMPILED_LABEL, "Rank_Score", "Kmer_Idxs"]]
