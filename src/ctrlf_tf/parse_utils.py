"""Functions for parsing files."""

from typing import Iterable, Tuple
import re

import pandas as pd


def parse_k(kmer: str) -> int:
    """Return k from a kmer with 0 or more gaps."""
    return len(kmer.replace('.', ''))

def parse_integer_parameters(parameter_str):
    return [int(i) for i in parameter_str.strip().split(": ")[1].split()]


def validate_k_in_kmers(k: int, kmers: Iterable[str]) -> None:
    """Parse k from a list of k-mers.

    Given a list of k-mers, each with any variable number of gaps, return
    the value of k. If this value is not consistant among the input, raise
    an error indicating such.
    """
    invalid_kmers = list(filter(lambda x: parse_k(x) != k, kmers))
    if len(invalid_kmers) > 0:
        formated_output = '\n'.join(invalid_kmers)
        raise ValueError("Given k = {k}, the following k-mers did not match:"
                         f"\n{formated_output}")


def max_length(strings: Iterable[str]) -> int:
    """Return the max length from a list of strings."""
    return max(map(lambda x: len(x), strings))


def parse_core_positions(input_str: str) -> Tuple[int]:
    """Parse a string for core alignment output."""
    if input_str.startswith("#Core Aligned Positions:") is False:
        raise ValueError("String parsed is not a core positions string.")
    # Retrive value from key: value syntax
    positions = input_str.split(": ")[1].strip()
    # Split the positions, deliminated by whitespace
    position_list = positions.split()
    # Cast to integers and return as tuple
    core_positions = [int(i) for i in position_list]
    return tuple(core_positions)


def parse_boolean(input_string: str) -> bool:
    """Parse a string for boolean settings."""
    key, value = input_string.split(": ")
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError((f"Boolean setting string {input_string} does not"
                      " have True or False as a value."))


def parse_orientation_bool(input_string: str) -> bool:
    """Parse a string for boolean settings."""
    key, value = input_string.split(": ")
    if value == ".":
        return True
    if value == "+/-":
        return False
    raise ValueError((f"Boolean setting string {input_string} does not"
                      " have +/- or . as a value."))


def is_ascending_coordinate(coordinate: str) -> bool:
    """Return True if the coordinate is in ascending order, else false."""
    start, end = coordinate.split(":")[1].split("-")
    return int(start) < int(end)


def validate_genome_coordinate_label(coordinate: str) -> bool:
    """Validate if a genomic coordinate label is in an expected format.

    Given a string of genome coordinates, checks that the structure is a
    valid match.

    Valid format: Chromosome:start-end

    :param coordinate: String in the format 'Chromosome:Start-End'
    :returns: True if valid, else raises a ValueError
    """
    match = re.match(".*:\d*-\d*", coordinate)
    if (match and
       match.group(0) == coordinate and
       is_ascending_coordinate(coordinate)):
        return True
    error_message = (f"Genomic coordinate {coordinate} not a valid "
                     "format\nExpected: Chromosome:start-end with "
                     "start and end positions increasing")
    raise ValueError(error_message)


def read_fasta_entry(file_object) -> Tuple[str, str]:
    """Read a fasta entry.

    :param file_object: File object of a fasta file
    :returns: Tuple of header and sequence
    """
    header = file_object.readline().rstrip()
    # If end of file
    if header == '':
        return (False, False)
    # If new sequence has no header, raise error
    if header.startswith('>') is False:
        raise ValueError("Expected sequence header to start with '>")
    sequence = ''
    read_sequence = True
    while read_sequence:
        position_before_read = file_object.tell()
        line = file_object.readline().rstrip()
        # If end of file
        if line == '':
            read_sequence = False
        # If at next entry
        elif line.startswith('>'):
            file_object.seek(position_before_read)
            read_sequence = False
        else:
            sequence = sequence + line
    return (header, sequence)


def parse_fasta_header(header: str,
                       genome_label=True) -> Tuple[str, int]:
    """Parse fasta header information.

    Given the string of a fasta header, returns a tuple with the chromosome
    and the start position of the chromosome. If genome_label is True, then
    the chromosome start position is parsed, otherwise it is set to 0.

    :param header: String that begins with the '>' character
    :param genome_name_parse: Boolean to parse the header in the format
        'Chromosome:Start-End'
    :returns: Tuple of the chromosome and chromosome start position
    """
    # Check that fasta entry has a header and sequence
    if header.startswith('>') is False:
        raise ValueError(f"Fasta entry does not start with '>'\n{header}")
    header = header.lstrip('>')
    # If the label is formated for genomic coordiantes, parse it
    if genome_label:
        validate_genome_coordinate_label(header)
        chromosome, location_string = header.split(":")
        chromosome_start = location_string.split("-")[0]
        chromosome_start = int(chromosome_start)
    else:
        chromosome = header
        chromosome_start = 0
    return (chromosome, chromosome_start)


def validate_align_parameters(core_start,
                              core_end,
                              range_consensus,
                              threshold,
                              threshold_column):
    """Check that the align parameters are valid options."""
    # Is there a specified core and if so is it valid?
    core_specified = False
    # core start cannot be a negative value
    if core_start < 0:
        raise ValueError(f"Core start is {core_start}, has to be 0 or greater.")
    if core_start == core_end and core_start > 0:
        raise ValueError(f"Core end must come after core start unless both are 0.")
    if core_start > core_end:
        raise ValueError(f"Core start, {core_start} specified after core end, {core_end}.")
    if core_start > 0:
        core_specified = True
    # Is there a range consensus and core specification conflict?
    if range_consensus and core_specified:
        raise ValueError(f"Range consensus specified with start and end core positions, must be either or but not both.")


def align_parameter_dict_from_header(meta_data_string: str) -> dict:
    """"""
    rows = meta_data_string.split("\n")
    header_dict = {}
    for row in rows:
        name, data = row.split(":")
        header_dict[name] = data.strip()
    if header_dict["#Core gaps"] == "":
        core_gaps = None
    else:
        core_gaps = [int(i) for i in header_dict["Core gaps"].split()]
    palindrome = header_dict["#Palindrome"] == "True"
    parameter_dict = {"kmer_file": header_dict["#Kmer file"],
                      "pwm_file": header_dict["#PWM file"],
                      "pwm_file_format": header_dict["#PWM file format"],
                      "core_start": header_dict["#Core start"],
                      "core_end": header_dict["#Core end"],
                      "core_gaps": core_gaps,
                      "palindrome": palindrome,
                      "version": header_dict["#Parameter version"]}
    return parameter_dict


def parse_parameter_str(label, value):
    # Convert boolean str to bool
    if value == "True":
        return True
    if value == "False":
        return False
    if label == "threshold":
        return float(value)
    if label in ("core_start", "core_end", "gap_limit"):
        return int(value)
    if value[0] == '[' and value[-1] == ']':
        return [int(i) for i in value[1:-1].split()]
    return value

def parameter_dict_from_strs(strs):
    param_dict = {}
    for line in strs:
        label, value = line.split(": ")
        label = label.lstrip("#")
        value = value.strip()
        param_dict[label] = parse_parameter_str(label, value)
    return param_dict


def parameter_dict_from_file(file_path):
    with open(file_path) as f:
        param_dict = parameter_dict_from_strs(f)
    return param_dict
