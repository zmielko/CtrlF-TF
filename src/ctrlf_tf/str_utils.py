"""String functions used in CtrlF-TF"""

from collections import defaultdict
from typing import List, Iterable

import pandas as pd
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
