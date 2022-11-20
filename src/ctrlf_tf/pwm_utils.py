"""PWM functions.

This module defines functions related to the use of Position Weight Matrices.
"""

from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import ctrlf_tf.str_utils


def alignment_score_position(kmer: str, pwm_dict: dict) -> Tuple[float, int]:
    """Return the max score and alignment position of a kmer.

    Takes a kmer and scans it along a PWM, returns all aligned positions
    with the max score

    :param kmer: kmer to determine the maximum PWM score of
    :param pwm_dict: Dictionary of each letter and their probabilities
    :returns: Tuple of the max alignment score and its position
    """
    score_list = []
    pwm_length = len(pwm_dict['A'])
    for pwm_position in range(pwm_length - len(kmer) + 1):
        score = 1
        for letter_position, letter in enumerate(kmer):
            if letter != '.':
                score = score * pwm_dict[letter][letter_position + pwm_position]
            else:
                score = score * 0.25
        score_list.append(score)
    max_score = max(score_list)
    align_position = [idx for idx, score in enumerate(score_list) if score == max_score]
    return (max_score, align_position)


def core_positions_from_pwm(pwm: pd.DataFrame,
                            core_gaps: Iterable[int] = None) -> Tuple[int]:
    """Calculate a tuple of alignment positions that describe the core.

    Calculates a tuple of alignment positions for a core region. If
    core_gaps are provided, will not include those positions.

    :param pwm: PWM alignment model
    :param pad_length: Length that the pwm will be padded
    :param core_gaps: Gap 1-base positions in the alignment model
    :returns: Tuple of core positions
    """
    core_length = len(pwm[0])
    core_list = list(range(core_length))
    if core_gaps:
        for i in core_list:
            if i + 1 in core_gaps:
                core_list.remove(i)
    return tuple(core_list)


def pad_pwm_equiprobable(pwm: np.ndarray, pad_len: int) -> np.ndarray:
    """Add a given number of equiprobable columns to both ends of a PWM.

    Takes a PWM in the form of a numpy array, pads it with equiprobable
    flanks of a given length, and returns the padded array.

    :param pwm_df: Pandas dataframe for a pwm file in probability format
    :param pad_len: Length to pad the pwm on both sides
    :returns: PWM with equiprobable padding as a dictionary
    """
    eqiprobable_column = np.array(([0.25], [0.25], [0.25], [0.25]))
    padding = np.repeat(eqiprobable_column, pad_len, axis=1)
    pwm = np.concatenate((padding, pwm, padding), axis=1)
    return pwm


def pwm_ndarray_to_dict(pwm: np.ndarray) -> dict:
    """Convert a np.ndarray PWM to a dictionary."""
    pwm_dict = {'A': pwm[0],
                'C': pwm[1],
                'G': pwm[2],
                'T': pwm[3]}
    return pwm_dict


def read_pwm_from_meme(pwm_file: str) -> np.ndarray:
    """Read a PWM from a MEME formated file.

    :param pwm_file: PWM file input in MEME format
    :returns: Numpy array of the PWM
    """
    f = open(pwm_file, 'r')
    skip = -1  # default value of not found
    for row_number, line in enumerate(f):
        if line.strip().startswith("letter-probability"):
            pwm_header = line.strip()
            skip = row_number + 1
            row_read_count = int(pwm_header.split(' ')[5])
            break
    f.close()
    if skip == -1:
        raise ValueError("No line with letter-probability found in file")
    pwm = pd.read_csv(pwm_file,
                      delim_whitespace=True,
                      header=None,
                      skiprows=skip,
                      nrows=row_read_count)
    pwm = pwm.T
    pwm_matrix = pwm.to_numpy()
    return pwm_matrix


def read_pwm_from_tabular(pwm_file: str) -> np.ndarray:
    """Read a PWM from a tabular formated text file as an array.

    :param pwm_file: PWM file location
    :returns: PWM as a numpy array
    """
    pwm = pd.read_csv(pwm_file, delim_whitespace=True, header=None)
    pwm = pwm.sort_values(by=0)
    pwm = pwm.to_numpy()
    pwm = np.delete(pwm, 0, 1).astype("float")
    return pwm


def read_pwm_file(pwm_file: str,
                  file_format: str = "Tabular") -> np.ndarray:
    """Read a pwm file and returns a numpy array.

    Given a pwm_file location and an argument specifying the format of
    the file, returns a numpy ndarray of the pwm. Wrapper function for
    read_pwm_from_tabular and read_pwm_from_meme

    :param pwm_file: File location for a PWM
    :param file_format: Format specification (Tabular/MEME) of the PWM
    :returns: Numpy array of the PWM
    """
    if file_format == "Tabular":
        pwm = read_pwm_from_tabular(pwm_file)
    elif file_format == "MEME":
        pwm = read_pwm_from_meme(pwm_file)
    else:
        raise ValueError(f"Agument file_format was {file_format}, "
                         "must be 'Tabular' or 'MEME'")
    return pwm


def trim_pwm_from_kmer_match(pwm_matrix: np.ndarray,
                             kmer: str,
                             position_choice: int = 0) -> np.ndarray:
    """Trim a PWM based on top scoring position from kmer.

    Given a PWM matrix, algins and scores a kmer, returns a trimmed PWM
    matrix at the position and range of the kmer

    :param pwm_matrix: probability matrix of a pwm
    :param kmer: kmer to find the aligned range of in the pwm
    :param position_choice: Given equal scoring positions, choice of the
        index for the position to chose
    """
    pwm_dict = pwm_ndarray_to_dict(pwm_matrix)
    kmer_length = len(kmer)
    score, position = alignment_score_position(kmer, pwm_dict)
    slice_start = position[position_choice]
    slice_end = slice_start + kmer_length
    pwm_trimmed = pwm_matrix[:, slice_start:slice_end]
    return pwm_trimmed


def model_params_from_consensus(consensus: str,
                                pwm_dict: dict,
                                pwm_dict_rc: dict,
                                position_choice: int = 0):
    """Return model parameters from a consensus sequence input."""
    score_f, position = alignment_score_position(consensus, pwm_dict)
    score_rc, position_rc = alignment_score_position(consensus, pwm_dict_rc)
    pwm_reverse_complement = False
    if score_rc > score_f:
        position = position_rc
        pwm_reverse_complement = True
    position = position[0]
    start_param = position + 1
    end_param = position + len(consensus)
    core_gap_iterable = []
    for idx, i in enumerate(consensus):
        if i == '.':
            core_gap_iterable.append(idx + 1)
    return (start_param, end_param, core_gap_iterable, pwm_reverse_complement)


def trim_pwm_by_core(pwm: np.ndarray,
                      core_range,
                      core_gaps):
    """Trim a PWM model by core definitions."""
    start_idx = core_range[0] - 1
    end_idx = core_range[1]
    pwm = pwm[:, start_idx:end_idx]
    if core_gaps:
        for i in core_gaps:
            pwm[:, i-1] = 0.25
    return pwm


def trim_pwm_wrapper(pwm: np.ndarray,
                     core_range: tuple = None,
                     core_gap: Iterable[int] = None,
                     range_consensus: str = None) \
        -> Tuple[np.ndarray, Iterable[int]]:
    """Select core definition based on parameter input.

    :param pwm: Input PWM to trim
    :type pwm: np.ndarray
    :param core_range: Start and end positions to trim the core (1-based)
    :type core_range: tuple
    :param core_gap: Relative position to the core range to create
        equiprobable positions
    :type core_gap: Iterable[int]
    :param range_consensus: Alternative input to core_range and core_gap
        for positions to trim the PWM and add equiprobable positions
    :type range_onsensus: str
    :returns: Tuple of the PWM and the core_gap iterable
    """
    core_gap_iterable = None
    # If a range of positions is given
    if core_range:
        start_idx = core_range[0] - 1
        end_idx = core_range[1]
        pwm = pwm[:, start_idx:end_idx]
    # If an iterable of gaps in the core is given
    if core_gap:
        core_gap_iterable = core_gap
    # If a kmer to align and trim is given
    elif range_consensus:
        pwm = trim_pwm_from_kmer_match(pwm, range_consensus)
        core_gap_iterable = []
        for idx, i in enumerate(range_consensus):
            if i == '.':
                core_gap_iterable.append(idx + 1)
    # If there is a core gap iterable, make the PWM equiprobable
    if core_gap_iterable:
        for i in core_gap_iterable:
            pwm[:, i-1] = 0.25
    return (pwm, core_gap_iterable)


def align_kmers_from_df(kmer_df: pd.DataFrame,
                        pwm_dict: dict,
                        is_palindrome: bool,
                        core_start: int,
                        rank_score_label: str) -> pd.DataFrame:
        """Align kmers from an input kmer dataframe.

        Given a pandas dataframe of kmer data and an argument if the alignment
        is for a palindromic model, returns a dataframe with aligned kmers.
        Wrapper for the alignment_score_position function.

        :param kmer_df: Pandas dataframe of kmers
        :param pwm_length: length of the pwm model
        :param pwm_dict: Dictionary of nucelotides with probabilities
        :param is_palindrome: Indicates if the model is palindromic
        :param return_align_score: Argument to return the alignment score
        :returns: Pandas dataframe with aligned kmer positions
        """
        alignment_result_list = []
        for kmer, rank_score in zip(kmer_df.iloc[:, 0],
                                    kmer_df[rank_score_label]):
            kmer_score, kmer_positions = alignment_score_position(kmer, pwm_dict)
            rc_kmer = ctrlf_tf.str_utils.reverse_complement(kmer)
            rc_score, rc_positions = alignment_score_position(rc_kmer, pwm_dict)
            if is_palindrome:
                # Align forward and reverse complement orientations
                for position in kmer_positions:
                    alignment_result_list.append([kmer,
                                                  position,
                                                  kmer_score,
                                                  rank_score])
                for position in rc_positions:
                    alignment_result_list.append([rc_kmer,
                                                  position,
                                                  rc_score,
                                                  rank_score])
            else:
                # Align only the top scoring orientation
                if kmer_score >= rc_score:
                    for position in kmer_positions:
                        alignment_result_list.append([kmer,
                                                      position,
                                                      kmer_score,
                                                      rank_score])
                if kmer_score <= rc_score:
                    for position in rc_positions:
                        alignment_result_list.append([rc_kmer,
                                                      position,
                                                      rc_score,
                                                      rank_score])
        aligned_df = pd.DataFrame(alignment_result_list,
                                  columns=['Kmer',
                                           "Align_Position",
                                           "Align_Score",
                                           "Rank_Score"])
        aligned_df["Align_Position"] = aligned_df["Align_Position"] - core_start
        return aligned_df.drop_duplicates().reset_index(drop=True)
