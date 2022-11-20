#!/usr/bin/env python3

from collections import namedtuple
from typing import Iterable, List

import ahocorasick
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


def automata_from_sites(consensus_sites: Iterable[str]) -> ahocorasick.Automaton:
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
