#!/usr/bin/env python3

from collections import namedtuple, defaultdict
from typing import Iterable, List

import ahocorasick
import ctrlf_tf.str_utils
import ctrlf_tf.compile_utils
import numpy as np
import pybktree
import networkx as nx
import pandas as pd



# Named tuple outputs
SiteTuple = namedtuple("SiteTuple", ["start",
                                     "end",
                                     "orientation",
                                     "threshold"])
BedTuple = namedtuple("BedTuple", ["chromosome",
                                   "start",
                                   "end",
                                   "name",
                                   "score",
                                   "orientation"])


# Unique for this
def kmer_core_description(kmer: str,
                          align_position: int,
                          relative_core_position_dict: dict) \
                          -> np.ndarray:
    """ Creates an array of 0 and 1 indicating the description of a core

    Given a kmer, align_position, and core, returns the description of the
    core as a numpy array of core indexes and an offset value of the kmer's
    alignment position to the core's position.

    Example:

    A kmer, CACGTG, with an align position of 10 and a core of 10, 11, 12
    with return the array np.array([1, 1, 1]). If the kmer was C.CGTG and
    the same align position it would return np.array([1, 0, 1])

    :param kmer: String of A,C,G,T and '.' characters
    :param align_position: Alignment position of the kmer to a model
    :param relative_core_position_dict: Dictionary of core positions to
        relative positions
    :returns: Numpy array of alignment positions described by the kmer
    """
    core_description_array = np.zeros(len(relative_core_position_dict))
    for idx, i in enumerate(kmer):
        position = align_position + idx
        if position in relative_core_position_dict and i != '.':
            core_description_array[relative_core_position_dict[position]] += 1
    return core_description_array


def automata_from_sites(consensus_sites: Iterable[str]) -> ahocorasick.Automaton:
    """Creates an Aho-Corasick automata from an iterable of sequences."""
    automata = ahocorasick.Automaton()
    for idx, site in enumerate(consensus_sites):
        automata.add_word(site, (idx, site))
    automata.make_automaton()
    return automata


def compiled_dict_from_compiled_sequences(compiled_sequences: Iterable[str],
                                          end_position: Iterable[int],
                                          rank_scores: Iterable[float]) -> dict:
    """Generates a compiled sequence dictionary for fast lookup.

    Given iterables of compiled sequences, end_positions of the non-wildcard string
    relative to the start of the binding site, and rank scores, returns a dictionary with
    keys as the compiled sequences and values as lists of tuples for end_position and
    rank score pairs.
    """
    compiled_dict = {}
    for sequence, position, score in zip(compiled_sequences, end_position, rank_scores):
        if sequence not in compiled_dict:
            compiled_dict[sequence] = []
        compiled_dict[sequence].append((position, score))
    return compiled_dict


def update_site_dict_from_automata_match(automata_match: tuple,
                                         compiled_dict: dict,
                                         site_dict: dict) -> dict:
    match_end_pos = automata_match[0]
    match_str = automata_match[1][1]
    for site in compiled_dict[match_str]:
        site_start = match_end_pos - site[0] + 1
        score = site[1]
        if site_start not in site_dict or site_dict[site_start] < score:
            site_dict.update({site_start: score})


def site_dict_from_sequence(sequence: str,
                            automata: ahocorasick.Automaton,
                            compiled_dict: dict) -> dict:
    results = {}
    for match in automata.iter(sequence):
        update_site_dict_from_automata_match(match, compiled_dict, results)
    return results


def site_dict_to_sitetuples(site_dictionary: dict,
                            sequence: str,
                            orientation: str,
                            site_span: int) -> List[SiteTuple]:
    """Format a site_dictionary to a SiteTuple NamedTuple.

    Given a dictionary of keys as positions and scores as values, returns a
    list of SiteTuples where the site information is adjusted for
    orientation.

    :param site_dictionary: Dictionary of called sites
    :param sequence: DNA sequence used to call the site
    :param orientation: Orientation of the call (+/-/.)
    :returns: List of SiteTuple NamedTuples
    """
    sites = []
    sequence_len = len(sequence)
    for site_dict_start in site_dictionary:
        # Adjust for orientation if needed
        if orientation == '-':
            start = sequence_len - (site_dict_start + site_span + 1)
        else:
            start = site_dict_start
        end = start + site_span + 1
        score = site_dictionary[site_dict_start]
        site = SiteTuple(start, end, orientation, score)
        # Check site in-bounds in case site_span includes wildcard positions
        if site.start >= 0 and site.end <= sequence_len:
            sites.append(site)
    return sites


def site_tuples_to_bed(called_sites: Iterable[SiteTuple],
                       chromosome: str,
                       chromosome_start: int,
                       chromosome_end: int) -> List[BedTuple]:
    """Format SiteTuple as BedTuple.

    Given a dictionary output from call_sites, the name of the sequence,
    the orientation, and optionally the start position of the sequence,
    returns a list of strings in bed format

    :param called_sites: Iterable of SiteTuples
    :param chromosome: Chromosome name
    :param chromosome_start: Start position in the chromosome of the
        sequence
    :returns: List of BedTuples
    """
    bedtuples = []
    for idx, site in enumerate(called_sites):
        start = site.start + chromosome_start
        end = site.end + chromosome_start
        name = (f"{chromosome}:{chromosome_start}-{chromosome_end}"
                f"_{site.orientation}_{idx}")
        bedtuple = BedTuple(chromosome,
                            start,
                            end,
                            name,
                            site.threshold,
                            site.orientation)
        bedtuples.append(bedtuple)
    return bedtuples


def relative_positions_from_core(core_positions: Iterable[int]) -> dict:
    """Core positions to description indicies

    Example:

    relative_positions_from_core([0, 1, 3]) -> {0: 0, 1: 1, 3: 2}

    Given an iterable of core positions, returns a dictionary that
    converts from the alignment position to the index relative to the core.
    """
    core_positions = sorted(core_positions)
    relative_core_position_dict = {}
    for idx, i in enumerate(core_positions):
        relative_core_position_dict[i] = idx
    return relative_core_position_dict

def pad_k(kmer, align_position, left_bound, right_bound):
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

def padded_kmers_from_aligned_df(aligned_df):
    """Return padded k-mers from an aligned k-mer dataframe"""
    kmer_dataframe = aligned_df.drop_duplicates().reset_index(drop=True)
    # Calculate bounds
    kmers = kmer_dataframe.iloc[:, 0]
    max_word_len = max([len(kmer) for kmer in kmer_dataframe.iloc[:, 0]])
    aligned_positions = kmer_dataframe.iloc[:, 1]
    left_bound, right_bound = ctrlf_tf.compile_utils.bounds_from_aligned_positions(aligned_positions, max_word_len)
    # Pad k-mers relative to bounds
    padded_kmers = []
    for row in kmer_dataframe.itertuples():
        padded_kmer = pad_k(row.Kmer, row.Align_Position, left_bound, right_bound)
        padded_kmers.append(padded_kmer)
    return padded_kmers

def create_traverse_graph_from_padded_kmers(padded_kmers):
    """Calculates all-v-all traversals in nlogn."""
    # Create index dictionary
    kmer_idx_dict = {}
    for idx, i in enumerate(padded_kmers):
        kmer_idx_dict[i] = idx
    # Setup Graph
    graph = nx.Graph()
    # Setup BKTree
    bktree = pybktree.BKTree(ctrlf_tf.str_utils.hamming_distance)
    for i in padded_kmers:
        bktree.add(i)
    # Search BKTree
    for i in padded_kmers:
        idx_a = kmer_idx_dict[i]
        r = bktree.find(i, 2)
        for j in r:
            if idx_a != kmer_idx_dict[j[1]]:
                graph.add_edge(idx_a,kmer_idx_dict[j[1]])
    return graph


def kmer_to_index_dict_from_aligned_df(aligned_kmer_df):
    """"""
    kmer_to_index = defaultdict(dict)
    idx = 0
    for kmer, ap in zip(aligned_kmer_df["Kmer"], aligned_kmer_df['Align_Position']):
        kmer_to_index[kmer][ap] = idx
        idx += 1
    return kmer_to_index


def kmer_to_score_dict_from_aligned_df(aligned_df):
    return dict(zip(aligned_df["Kmer"], aligned_df["Rank_Score"]))


def index_to_score_dict_from_aligned_df(aligned_df):
    index_to_score_dict = {}
    for idx, score in enumerate(aligned_df["Rank_Score"]):
        index_to_score_dict[idx] = score
    return index_to_score_dict


def expanded_kmer_to_original_dict_from_aligned_df(aligned_df):
    # Expand k-mers and create mapping
    expanded_kmer_to_kmer = defaultdict(dict)
    for row in aligned_df.itertuples():
        expanded_list = expand_kmer(row.Kmer)
        for i in expanded_list:
            expanded_kmer_to_kmer[i][row.Kmer] = row.Align_Position
    return expanded_kmer_to_kmer

# Get index to core position dictionary
def kmer_idx_to_core_position_dict_from_aligned_df(aligned_df, core_positions):
    relative_core_dict = relative_positions_from_core(core_positions)
    idx = 0
    kmer_idx_core_pos_dict = {}
    for kmer, ap in zip(aligned_df["Kmer"], aligned_df["Align_Position"]):
        kmer_idx_core_pos_dict[idx] = kmer_core_description(kmer, ap, relative_core_dict)
        idx += 1
    return kmer_idx_core_pos_dict


def overlapping_kmers(graph: nx.Graph, subgraph_nodes: set) -> List[nx.Graph]:
    """Return all connected components for subgraphs

    Given a set of kmers, returns a list of networkx subgraphs that
    have 2 or more kmers

    :param subgraph_nodes: Set of kmers to subset the graph by
    :type subgraph_nodes: set
    :returns: List of connected graphs with 2 or more nodes
    """
    subgraph = nx.subgraph(graph, subgraph_nodes)
    connected_groups = []
    for i in nx.connected_components(subgraph):
        if len(i) > 1:
            connected_groups.append(list(i))
    return connected_groups

def score_site_from_candidate_list(candidate_list,
                                   index_to_score_dict,
                                   traversal_graph,
                                   kmer_idx_to_core_position_dict):
    """Current method = N2, could be NlogN"""
    candidate_list = sorted(candidate_list)
    kmer_idx_set = set()
    kmer_idx_set.add(candidate_list[0])
    for idx in range(2, len(candidate_list[1:])):
        candidate_indexes = candidate_list[:idx]
        kmer_idx_set.add(candidate_list[idx - 1])
        # Check for connected groups
        for connected_groups in overlapping_kmers(traversal_graph, kmer_idx_set):
            core_descriptions = [kmer_idx_to_core_position_dict[i] for i in connected_groups]
            if ctrlf_tf.compile_utils.is_core_described(core_descriptions):
                return index_to_score_dict[candidate_indexes[-1]]
    return None

# Search a string for all matches



NonCompilePreprocess = namedtuple("NonCompilePreprocess", ["kmer_automata",
                                                           "expanded_kmer_to_original_dict",
                                                           "kmer_to_index_dict",
                                                           "index_to_score_dict",
                                                           "index_to_core_position_dict",
                                                           "traversal_graph",
                                                           "site_span"])

def noncompile_preprocessing_from_aligned_kmers(aligned_kmer_dataframe, core_positions):
    padded_kmers = padded_kmers_from_aligned_df(aligned_kmer_dataframe)
    traverse_graph = create_traverse_graph_from_padded_kmers(padded_kmers)
    index_to_score_dict = index_to_score_dict_from_aligned_df(aligned_kmer_dataframe)
    kmer_to_index_dict = kmer_to_index_dict_from_aligned_df(aligned_kmer_dataframe)
    index_to_core_position_dict = kmer_idx_to_core_position_dict_from_aligned_df(aligned_kmer_dataframe, core_positions)
    expanded_kmer_to_original_dict = expanded_kmer_to_original_dict_from_aligned_df(aligned_kmer_dataframe)
    automata = automata_from_sites(list(expanded_kmer_to_original_dict.keys()))
    site_span = max(core_positions)
    return NonCompilePreprocess(automata,
                                expanded_kmer_to_original_dict,
                                kmer_to_index_dict,
                                index_to_score_dict,
                                index_to_core_position_dict,
                                traverse_graph,
                                site_span)


def site_dict_noncompiled_from_sequence(sequence,
                                        noncompilepreprocess):
    candidate_site_dict = defaultdict(list)
    for match in noncompilepreprocess.kmer_automata.iter(sequence):
        match_sequence = match[1][1]
        match_position = match[0] - len(match_sequence) + 1
        for original_kmer in noncompilepreprocess.expanded_kmer_to_original_dict[match_sequence]:
            align_position = noncompilepreprocess.expanded_kmer_to_original_dict[match_sequence][original_kmer]
            rel_core = (align_position * -1) + match_position
            candidate_site_dict[rel_core].append(noncompilepreprocess.kmer_to_index_dict[original_kmer][align_position])
    site_dict = {}
    for i in candidate_site_dict:
        score = score_site_from_candidate_list(candidate_site_dict[i],
                                               noncompilepreprocess.index_to_score_dict,
                                               noncompilepreprocess.traversal_graph,
                                               noncompilepreprocess.index_to_core_position_dict)
        if score is not None:
            site_dict[i] = score
    return site_dict


def call_sites_with_kmers(sequence: str, noncompilepreprocess, is_palindrome):
    """Returns a list of SiteTuples from an input sequence.

    Given a sequence, returns a list of SiteTuples for each called site.

    :param sequence: Input DNA sequence
    :type sequence: str
    :param fixed_length: Search mode assumes a fixed model length
    :type fixed_length: bool
    :returns: List of SiteTuples
    """
    # Set sequence to uppercase so match is case insensisitivw
    sequence = sequence.upper()
    # Use appropriate span
    site_span = noncompilepreprocess.site_span
    # Call sites from the input sequence orientation, if palindrome return the results
    orient1 = site_dict_noncompiled_from_sequence(sequence,
                                                  noncompilepreprocess)
    if is_palindrome:
        return site_dict_to_sitetuples(orient1, sequence, '.', site_span)
    # Otherwise call sites on the reverse complement and return results from both orientations
    orient2 = site_dict_noncompiled_from_sequence(ctrlf_tf.str_utils.reverse_complement(sequence),
                                                  noncompilepreprocess)
    pos_sites = ctrlf_tf.site_call_utils.site_dict_to_sitetuples(orient1, sequence, '+', site_span)
    neg_sites = ctrlf_tf.site_call_utils.site_dict_to_sitetuples(orient2, sequence, '-', site_span)
    return pos_sites + neg_sites
