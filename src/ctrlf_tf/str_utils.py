"""String functions used in CtrlF-TF"""

from collections import defaultdict
from typing import List, Iterable

import pandas as pd
import pybktree
import networkx as nx
import numpy as np


def reverse_complement(sequence: str) -> str:
    """Reverse complement of a DNA sequence.

    Generates the reverse complement of a given nucleotide sequence in
    upper case

    :param sequence: String with 'A', 'C', 'G', 'T' in any case
    :returns: Reverse complement of the sequence
    """
    # Define dictionary, make input string upper case
    complement_dict = {'A': 'T',
                       'T': 'A',
                       'C': 'G',
                       'G': 'C'}
    sequence_upper = sequence.upper()
    complement_sequence = ''
    for letter in sequence_upper:
        # If there is no complement, use the letter
        if letter not in complement_dict:
            complement_sequence = complement_sequence + letter
        else:
            complement_sequence = complement_sequence + complement_dict[letter]
    return complement_sequence[::-1]


def compatible_description(string1: str, string2: str) -> bool:
    """Return True if 2 aligned strings can describe the same space.

    Determines if 2 aligned strings of the same length are able to describe
    the same search space. The '.' character acts as a wild card.

    :param kmer1: First string to compare
    :param kmer2: Second string to compare
    :returns: True if both strings are compatible, else false
    """
    if len(string1) != len(string2):
        raise ValueError("Strings compared are not equal length")
    for i in range(len(string1)):
        if (string1[i] != '.' and
           string2[i] != '.' and
           string1[i] != string2[i]):
            return False
    return True


def hamming_distance(string1: str, string2: str) -> int:
    if len(string1) != len(string2):
        raise ValueError("Strings compared are not equal length")
    result = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            result += 1
    return result


def is_within_gap_limit(kmer_iterable: Iterable[str],
                        gap_limit: int) -> List[bool]:
    """Return a list of booleans if kmers are within a gap limit.

    :param kmer_iterable: Iterable of kmers with or without gaps
    :param gap_limit: Maximum number of gaps for a kmer to return True
    """
    if gap_limit < 0:
        raise ValueError(f"Gap limit of {gap_limit} is less than 0")
    bool_list = [kmer.count('.') <= gap_limit for kmer in kmer_iterable]
    return bool_list


def k_from_kmers(kmers: Iterable[str]) -> int:
    """Derives k from an iterable of k-mers and ensures they share the same k."""
    # Derive k
    k = len(kmers[0].replace('.', ''))
    # Validate it being consistant in the iterable of k-mers
    for kmer in kmers:
        if len(kmer.replace('.', '')) != k:
            raise ValueError("Value of k is not consistent among kmers.")
    return k


def max_length_from_kmers(kmers: Iterable[str]) -> int:
    """Returns the max_length from an iterable of k-mers."""
    return max([len(kmer) for kmer in kmers])


def read_kmer_data(kmer_file: str,
                   threshold: float = None,
                   threshold_column: str = None,
                   gap_limit: int = None) -> pd.DataFrame:
    """Return a pandas dataframe from a path to a kmer file.

    Reads a kmer file. Filters kmers by a threshold for a given
    metric column if provided. Filters kmers by gap limit if provided.
    Returns a Pandas DataFrame of the kmers.

    :param kmer_file: File path for kmers in Seed-and-Wobble format
    :param threshold: Threshold value for a kmer (default = None)
    :param threshold_column: Column to threshold by (default = None)
    :param gap_limit: Maximum number of gaps, must be 0 or positive integer
    """
    kmer_df = pd.read_csv(kmer_file, sep='\t')
    if threshold:
        if threshold_column not in kmer_df.columns:
            raise ValueError(("Threshold column not found in"
                             f"{kmer_df.columns}"))
        kmer_df = kmer_df[kmer_df[threshold_column] >= threshold]
    if gap_limit is not None:
        kmer_series = kmer_df.iloc[:, 0]
        kmer_boolean_list = is_within_gap_limit(kmer_series, gap_limit)
        kmer_df = kmer_df[kmer_boolean_list]
    kmer_df = kmer_df.reset_index(drop=True)
    return kmer_df


def relative_end_positions(aligned_str: str,
                           start_position: int = 0,
                           wildcard: str = '.') -> int:
    end_position = len(aligned_str.rstrip(wildcard))
    return end_position - start_position


def total_length_aligned_strs(aligned_strs):
    total_length = len(aligned_strs[0])
    for i in aligned_strs[1:]:
        if len(i) != total_length:
            init = aligned_strs[0]
            raise ValueError(f"String length not consistant in total length calculation:\n{init}\n{i}")
    return total_length


def pad_k(kmer: str,
          align_position: int,
          left_bound: int,
          right_bound: int) -> str:
    """Returns k-mer strings in an aligned space."""
    left_pad = '.' * abs(left_bound - align_position)
    right_pad = '.' * abs(right_bound - (align_position + len(kmer)))
    return left_pad + kmer + right_pad


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


def bounds_from_aligned_positions(aligned_positions: Iterable[int],
                                  max_word_len: int) -> tuple:
    """Returns a left and right bound integer as a tuple."""
    left_bound = min(aligned_positions)
    right_bound = max(aligned_positions) + max_word_len
    return (left_bound, right_bound)


def padded_kmers_from_aligned_df(aligned_df):
    """Return padded k-mers from an aligned k-mer dataframe"""
    kmer_dataframe = aligned_df.drop_duplicates().reset_index(drop=True)
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
    return padded_kmers


def create_traverse_graph_from_padded_kmers(padded_kmers: Iterable[str]) -> nx.Graph:
    """Calculates all-v-all traversals in nlogn.

    Given an interable of padded k-mers, returns a graph where each
    node is an index number representing a k-mer and each link indicates
    that the hamming distance between the k-mers is within a given
    threshold (default = 2).

    """
    # Create index dictionary
    kmer_idx_dict = {}
    for idx, i in enumerate(padded_kmers):
        kmer_idx_dict[i] = idx
    # Setup Graph
    graph = nx.Graph()
    # Setup BKTree
    bktree = pybktree.BKTree(hamming_distance)
    for i in padded_kmers:
        bktree.add(i)
    # Search BKTree
    for i in padded_kmers:
        idx_a = kmer_idx_dict[i]
        r = bktree.find(i, 2)
        for j in r:
            if idx_a != kmer_idx_dict[j[1]]:  # Avoid self-loops
                graph.add_edge(idx_a, kmer_idx_dict[j[1]])
    return graph
