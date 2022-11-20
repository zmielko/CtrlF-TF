"""CtrlF_Core: Objects for the main pipeline of calling binding sites.

Module for calling binding sites using kmer data.

Classes:

1) AlignParameters: Dataclass of parameters used to align kmers to a PWM
2) AlignedKmers: Class that aligns kmers to a PWM model
3) CompiledKmers: AlignedKmers child that compiles k-mers into consensus sites
4) CtrlF: CompiledKmers child that calls sites from input sequence

"""

import copy
from dataclasses import dataclass, fields
from io import StringIO
import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import ctrlf_tf.pwm_utils
import ctrlf_tf.str_utils
import ctrlf_tf.parse_utils
import ctrlf_tf.compile_utils
import ctrlf_tf.site_call_utils

__version__ = "1.0b1.post1"
__author__ = "Zachery Mielko"


@dataclass
class AlignParameters:
    """Dataclass for input parameters to align k-mers to a model.

    :param kmer_file: Pathway to a text file of kmers. Must have columns
        representing: k-mer, reverse_complement k-mer, score where the score
        will be used to determine the rank of the k-mer.
    :type kmer_file: str
    :param pwm_file: Pathway to a text file for a position weight matrix to use
        as a model for alignment. Format is specified by the *pwm_file_format*
        argument, which is "Tabular" by default. The pwm file is assumed to
        define probabilities.
    :type pwm_file: str
    :param pwm_file_format: Indicates if the position weight matrix specified
        in the pwm_file param is in a tabular or meme format by specifying
        either *"Tabular"* or *"MEME"*.
    :type pwm_file_format: str
    :param core_start: 1-based, inclusive position for the start of the core.
        The core defines the positions that must be described by kmers to call
        a site.
    :type core_start: int
    :param core_end: 1-based inclusive position for the end of the core. The
        core defines the positions that must be described by kmers to call a
        site.
    :type core_end: int
    :param core_gaps: 1-based positions relative to core start that define
        sections within a core that do not need to be described. These
        positions will be equiprobable during alignment.
    :type core_gaps: Iterable
    :param range_consensus: A kmer input that defines core_start, core_end, and
        core_gaps by aligning the kmer to the PWM based on the maximum score
        and selecting the described positions as a core. Using a '.' in the
        kmer input will make that position a core_gap.
    :type range_consensus: str
    :param gap_limit: Limit on the number of gaps to allow kmers to have. Must
        be 0 or greater.
    :type gap_limit: int
    :param threshold: A score to filter aligned kmers from. The column used to
        filter is determined by *threshold_column*.
    :type threshold: float
    :param threshold_column: Column name in the kmer file to filter for kmers
        above a threshold determined by the *threshold* parameter.
    :type threshold_column: str
    :param palindrome: If *true*, both orientations of kmers are aligned to the
        core pwm model and orientations in called sites will me '.'. If *false*
        then the maximum scoring kmer of the two orientations will be used.
        Sites will be called with a '+' or '-' orientation.
    :type palindrome: bool
    :param version: Version of ctrlf_tf used in alignment
    :type version: str
    :param pwm_reverse_complement: If *true*, the PWM orientation is the
        reverse complement of the input PWM.
    :type pwm_reverse_complement: bool
    """

    pwm_file: str
    pwm_file_format: str = "Tabular"
    core_start: int = 0
    core_end: int = 0
    core_gaps: Iterable[int] = None
    range_consensus: str = None
    gap_limit: int = 0
    threshold: int = None
    threshold_column: str = None
    palindrome: bool = False
    version: str = __version__
    kmer_file: str = None
    pwm_reverse_complement: bool = False

    def __post_init__(self):
        """Parse and validate input parameters."""
        # Validate PWM model parameters
        ctrlf_tf.parse_utils.validate_align_parameters(self.core_start,
                                                        self.core_end,
                                                        self.range_consensus,
                                                        self.threshold,
                                                        self.threshold_column)
        # If a range consensus is specified, update the core and gap parameters
        if self.range_consensus:
            pwm = ctrlf_tf.pwm_utils.read_pwm_file(self.pwm_file,
                                                    self.pwm_file_format)
            full_pwm_dict = ctrlf_tf.pwm_utils.pwm_ndarray_to_dict(pwm)
            full_pwm_dict_rc = ctrlf_tf.pwm_utils.pwm_ndarray_to_dict(pwm[::-1, ::-1])
            parsed_params = ctrlf_tf.pwm_utils.model_params_from_consensus(self.range_consensus,
                                                                            full_pwm_dict,
                                                                            full_pwm_dict_rc)
            self.core_start, self.core_end, self.core_gaps, self.pwm_reverse_complement = parsed_params
        # If no range consensus or core specification, update core as whole PWM
        elif self.core_start == 0 and self.core_end == 0:
            pwm = ctrlf_tf.pwm_utils.read_pwm_file(self.pwm_file,
                                                    self.pwm_file_format)
            self.core_start = 1
            self.core_end = pwm.shape[1]
            self.core_gaps = []

    def save_parameters(self, file_path: str, mode='w'):
        """Saves the parameters to a file."""
        with open(file_path, mode) as file_obj:
            for i in fields(self):
                label = i.name
                if label != "range_consensus":
                    value = getattr(self, label)
                    file_obj.write(f"#{label}: {value}\n")

    @classmethod
    def _from_parameter_dict(cls, param_dict):
        return cls(pwm_file=param_dict["pwm_file"],
                   pwm_file_format=param_dict["pwm_file_format"],
                   core_start=param_dict["core_start"],
                   core_end=param_dict["core_end"],
                   core_gaps=param_dict["core_gaps"],
                   gap_limit=param_dict["gap_limit"],
                   threshold=param_dict["threshold"],
                   threshold_column=param_dict["threshold_column"],
                   palindrome=param_dict["palindrome"],
                   version=param_dict["version"],
                   kmer_file=param_dict["kmer_file"],
                   pwm_reverse_complement=param_dict["pwm_reverse_complement"])

    @classmethod
    def from_parameter_file(cls, file_path: str):
        param_dict = ctrlf_tf.parse_utils.parameter_dict_from_file(file_path)
        return cls._from_parameter_dict(param_dict)

    @classmethod
    def from_str_iterable(cls, iterable: Iterable[str]):
        param_dict = ctrlf_tf.parse_utils.parameter_dict_from_strs(iterable)
        return cls._from_parameter_dict(param_dict)


class AlignedKmers:
    """Aligns kmers to a PWM model."""
    # Initialization and constructors
    def __init__(self,
                 core_positions: Tuple[int] = None,
                 aligned_kmer_dataframe: pd.DataFrame = None,
                 k: int = None,
                 palindrome: bool = None,
                 pwm: np.ndarray = None,
                 version: str = None,
                 kmer_dataframe: pd.DataFrame = None):
        """Class initialization."""
        self.core_positions = core_positions
        self.aligned_kmer_dataframe = aligned_kmer_dataframe
        self.k = k
        self.palindrome = palindrome
        self.pwm = pwm
        self.version = version
        self.kmer_dataframe = kmer_dataframe

    @classmethod
    def from_parameters(cls, parameters: AlignParameters):
        """Construcor using an AlignedParameters object.

        A factory constructor that aligned kmers to a PWM model using
        parameters defined in a *AlignParameters* class.
        """
        # Read kmer data
        if parameters.kmer_file:
            kmer_df = ctrlf_tf.str_utils.read_kmer_data(parameters.kmer_file,
                                                        parameters.threshold,
                                                        parameters.threshold_column,
                                                        parameters.gap_limit)
        else:
            raise ValueError("Parameters do not contain a kmer or selex source.")
        # Adj threshold column
        rank_score_label = parameters.threshold_column
        if rank_score_label is None:
            rank_score_label = kmer_df.columns[2]
        # Read PWM information
        k = ctrlf_tf.str_utils.k_from_kmers(kmer_df.iloc[:, 0])
        pad_length = ctrlf_tf.str_utils.max_length_from_kmers(kmer_df.iloc[:, 0])
        pwm = ctrlf_tf.pwm_utils.read_pwm_file(parameters.pwm_file,
                                          parameters.pwm_file_format)
        if parameters.pwm_reverse_complement:
            pwm = pwm[::-1, ::-1]
        # Specify core range
        core_range = (parameters.core_start, parameters.core_end)
        pwm = ctrlf_tf.pwm_utils.trim_pwm_by_core(pwm, core_range, parameters.core_gaps)
        # Find core absolute start position and relative positions
        core_positions = ctrlf_tf.pwm_utils.core_positions_from_pwm(pwm,
                                                               parameters.core_gaps)
        core_absolute_start = core_positions[0] + pad_length
        # Pad PWM with equiprobable flanks
        pwm_padded = ctrlf_tf.pwm_utils.pad_pwm_equiprobable(pwm, pad_length)
        pwm_dict = ctrlf_tf.pwm_utils.pwm_ndarray_to_dict(pwm_padded)
        # Generate aligned kmers and hamming distance graph
        aligned_kmer_df = ctrlf_tf.pwm_utils.align_kmers_from_df(kmer_df,
                                                                  pwm_dict,
                                                                  parameters.palindrome,
                                                                  core_absolute_start,
                                                                  rank_score_label)
        return cls(core_positions,
                   aligned_kmer_df,
                   k,
                   parameters.palindrome,
                   pwm,
                   parameters.version,
                   kmer_df)

    @classmethod
    def from_alignment_file(cls, file_path: str):
        """Construct class using a previous AlignedKmers output file.

        Parses the information saved from using the method
        *.save_alignment* to make a new class instance.

        :param file_path: Path to saved alignment file
        :type file_path: str
        :returns: Class instance with data from the alignment file
        """
        # Read alignment file
        with open(file_path) as file_obj:
            aligned_kmer_data = file_obj.read()
        # Parse parameters in header
        with StringIO(aligned_kmer_data) as data_obj:
            version = data_obj.readline().split(': ')[1].strip()
            palindrome = ctrlf_tf.parse_utils.parse_boolean(data_obj.readline().strip())
            core_positions = ctrlf_tf.parse_utils.parse_core_positions(data_obj.readline().strip())
            pwm = np.loadtxt(data_obj, delimiter='\t', skiprows=1, max_rows=4)
            # Parse k-mer dataframe
            aligned_kmer_dataframe = pd.read_csv(data_obj, sep='\t', skiprows=1)
        # Parse k and max length
        k = ctrlf_tf.str_utils.k_from_kmers(aligned_kmer_dataframe.iloc[:, 0])
        # Return AlignedKmers object
        return cls(core_positions,
                   aligned_kmer_dataframe,
                   k,
                   palindrome,
                   pwm,
                   version)

    def copy(self):
        """Create a deep copy of the AlignedKmers object."""
        return copy.deepcopy(self)

    # Public instance methods
    def save_alignment(self, location: str = None):
        """Save alignment data to stdout or a file.

        :param location: Output location, if None output will be stdout
        :type location: str
        """
        # Determine if output is stdout or another location
        if location is None:
            output_file_object = sys.stdout
        else:
            output_file_object = open(location, "w")
        # Write version, alignment flag, and core positions
        output_file_object.write((f"#CtrlF Version: {__version__}\n"
                                  f"#Palindrome Alignment: {self.palindrome}\n"
                                  "#Core Aligned Positions:"))
        for i in self.core_positions:
            output_file_object.write(f" {i} ")
        output_file_object.write('\n')
        # Write position weight matrix used in alignment
        output_file_object.write('#Alignment Model\n')
        np.savetxt(output_file_object, self.pwm, delimiter='\t')
        # Write the aligned kmers
        output_file_object.write("#Aligned Kmers\n")
        # Set Align_Score to scientific notation for output
        output_df = self.aligned_kmer_dataframe.copy(deep=True)
        output_df["Align_Score"] = output_df["Align_Score"].apply(lambda x: "{:e}".format(x))
        output_df.to_csv(output_file_object, sep='\t', index=False)
        # Close file object if needed
        if location != sys.stdout:
            output_file_object.close()


class CompiledKmers(AlignedKmers):
    """Compiles aligned k-mers into consensus sites.

    This class is a child class of AlignedKmers that compiles the
    k-mers into consensus sites are part of the initialization
    process. The class also defines how compiled sites are saved
    and loaded to populate a new CompiledKmers object.
    """
    def __init__(self,
                 core_positions: Tuple[int] = None,
                 aligned_kmer_dataframe: pd.DataFrame = None,
                 k: int = None,
                 palindrome: bool = None,
                 pwm: np.ndarray = None,
                 version: str = None,
                 kmer_dataframe: pd.DataFrame = None,
                 compiled_site_dataframe: pd.DataFrame = None,
                 abs_core_start: int = None,
                 abs_core_end: int = None):
        super().__init__(core_positions=core_positions,
                         aligned_kmer_dataframe=aligned_kmer_dataframe,
                         k=k,
                         palindrome=palindrome,
                         pwm=pwm,
                         version=version,
                         kmer_dataframe=kmer_dataframe)
        self.compiled_site_dataframe = compiled_site_dataframe
        self.abs_core_start = abs_core_start
        self.abs_core_end = abs_core_end
        if self.compiled_site_dataframe is None:
            # Generate compiled sequences
            self.compiled_site_dataframe = ctrlf_tf.compile_utils.compile_consensus_sites(self.aligned_kmer_dataframe, self.core_positions)
            # Trim edges to minimal bounds
            left_idx, right_idx = ctrlf_tf.compile_utils.bounds_from_aligned_sequences(self.compiled_site_dataframe[ctrlf_tf.compile_utils.COMPILED_LABEL])
            self.compiled_site_dataframe[ctrlf_tf.compile_utils.COMPILED_LABEL] = self.compiled_site_dataframe[ctrlf_tf.compile_utils.COMPILED_LABEL].apply(lambda x: x[left_idx:right_idx])
        if self.abs_core_start is None and self.abs_core_end is None:
            self.abs_core_start = abs(min(self.aligned_kmer_dataframe["Align_Position"])) + 1 - left_idx
            self.abs_core_end = self.abs_core_start + max(self.core_positions)
        if self.abs_core_start is None or self.abs_core_end is None:
            raise ValueError("CompiledKmers object initialize with 1 of [abs_core_start, abs_core_end]. Must specify neither or both.")
        self.core_span = self.abs_core_end - self.abs_core_start

    def save_compiled_sites(self, output=None, minimal=True):
        """Saves compiled sites as a table to a file or stdout.

        :param output: Output location (default = stdout)
        :type output: str
        :param minimal: If *true*, removes column showing which kmer indexes
                were used to generate the solution.
        :type minimlal: bool
        """
        if output is None:
            output_file_object = sys.stdout
        else:
            output_file_object = open(output, "w")
        search_orientation = "+/-"
        if self.palindrome:
            search_orientation = '.'
        output_file_object.write((f"#CtrlF Version: {__version__}\n"
                                  f"#Search Orientation: {search_orientation}\n"
                                  f"#Core Range: {self.abs_core_start} {self.abs_core_end}\n"))
        if minimal:
            self.compiled_site_dataframe[[ctrlf_tf.compile_utils.COMPILED_LABEL, "Rank_Score"]].to_csv(output_file_object,
                                                                                  sep='\t',
                                                                                  index=False)
        else:
            self.compiled_site_dataframe.to_csv(output_file_object,
                                                sep='\t',
                                                index=False)
        if output is not None:
            output_file_object.close()

    @classmethod
    def from_compiled_sites(cls, file_path):
        """Construct class from *save_compiled_sites()* output.

        :param file_path: File location of compiled sites.
        :type file_path: str
        """
        with open(file_path) as file_obj:
            version = file_obj.readline().strip().split(": ")[1]
            palindrome = ctrlf_tf.parse_utils.parse_orientation_bool(file_obj.readline().rstrip())
            abs_core_start, abs_core_end = ctrlf_tf.parse_utils.parse_integer_parameters(file_obj.readline())
            compiled_site_df = pd.read_csv(file_obj, sep='\t')
        return cls(version=version,
                   palindrome=palindrome,
                   abs_core_start=abs_core_start,
                   abs_core_end=abs_core_end,
                   compiled_site_dataframe=compiled_site_df)


class CtrlF(CompiledKmers):
    """Class used to align, compile, and call sites."""
    def __init__(self,
                 core_positions: Tuple[int] = None,
                 aligned_kmer_dataframe: pd.DataFrame = None,
                 k: int = None,
                 palindrome: bool = None,
                 pwm: np.ndarray = None,
                 version: str = None,
                 kmer_dataframe: pd.DataFrame = None,
                 compiled_site_dataframe: pd.DataFrame = None,
                 abs_core_start: int = None,
                 abs_core_end: int = None):
        super().__init__(core_positions=core_positions,
                         aligned_kmer_dataframe=aligned_kmer_dataframe,
                         k=k,
                         palindrome=palindrome,
                         pwm=pwm,
                         version=version,
                         kmer_dataframe=kmer_dataframe,
                         compiled_site_dataframe=compiled_site_dataframe,
                         abs_core_start=abs_core_start,
                         abs_core_end=abs_core_end)
        # Setup internal copy with end positions for rigid and flexible searches.
        self._site_len = ctrlf_tf.str_utils.total_length_aligned_strs(self.compiled_site_dataframe[ctrlf_tf.compile_utils.COMPILED_LABEL])
        self.site_span = self._site_len - 1
        self._internal_cs_df = self.compiled_site_dataframe.copy(deep=True)
        self._internal_cs_df["Site_End_Pos"] = self._internal_cs_df[ctrlf_tf.compile_utils.COMPILED_LABEL].apply(lambda x: ctrlf_tf.str_utils.relative_end_positions(x))
        self._internal_cs_df["Core_End_Pos"] = self._internal_cs_df[ctrlf_tf.compile_utils.COMPILED_LABEL].apply(lambda x: ctrlf_tf.str_utils.relative_end_positions(x, start_position=self.abs_core_start - 1))
        self._internal_cs_df["Search_Sites"] = self._internal_cs_df[ctrlf_tf.compile_utils.COMPILED_LABEL].apply(lambda x: x.strip('.'))
        self.automata = ctrlf_tf.site_call_utils.automata_from_sites(self._internal_cs_df["Search_Sites"])
        self.fixed_length_search_dict = ctrlf_tf.site_call_utils.compiled_dict_from_compiled_sequences(self._internal_cs_df["Search_Sites"],
                                                                                                self._internal_cs_df["Site_End_Pos"],
                                                                                                self._internal_cs_df["Rank_Score"])
        self.variable_length_search_dict = ctrlf_tf.site_call_utils.compiled_dict_from_compiled_sequences(self._internal_cs_df["Search_Sites"],
                                                                                                   self._internal_cs_df["Core_End_Pos"],
                                                                                                   self._internal_cs_df["Rank_Score"])

    def call_sites(self, sequence: str, fixed_length=True):
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
        # Use appropriate span and compiled site dictionary for the mode given
        compiled_site_dict = self.fixed_length_search_dict
        site_span = self.site_span
        if not fixed_length:
            compiled_site_dict = self.variable_length_search_dict
            site_span = self.core_span
        # Call sites from the input sequence orientation, if palindrome return the results
        orient1 = ctrlf_tf.site_call_utils.site_dict_from_sequence(sequence,
                                                                   self.automata,
                                                                   compiled_site_dict)
        if self.palindrome:
            return ctrlf_tf.site_call_utils.site_dict_to_sitetuples(orient1, sequence, '.', site_span)
        # Otherwise call sites on the reverse complement and return results from both orientations
        orient2 = ctrlf_tf.site_call_utils.site_dict_from_sequence(ctrlf_tf.str_utils.reverse_complement(sequence),
                                                                   self.automata,
                                                                   compiled_site_dict)
        pos_sites = ctrlf_tf.site_call_utils.site_dict_to_sitetuples(orient1, sequence, '+', site_span)
        neg_sites = ctrlf_tf.site_call_utils.site_dict_to_sitetuples(orient2, sequence, '-', site_span)
        return pos_sites + neg_sites

    def call_sites_as_bed(self,
                          sequence: str,
                          chromosome: str,
                          chromosome_start: int,
                          fixed_length=True):
        """Call sites in BED format.

        Given a seuqence, chromosome, and chromosome_start information, returns
        called sites as a list of BedTuples

        :param sequence: Input DNA sequence
        :type sequence: str
        :param chromosome: Chromosome label
        :type chromosome: str
        :param chromosome_start: Start position of the input sequence
        :type chromosome_start: int
        :returns: List of BedTuples
        """
        sites = self.call_sites(sequence, fixed_length)
        chromosome_end = chromosome_start + len(sequence)
        bedtuple_result = ctrlf_tf.site_call_utils.site_tuples_to_bed(sites,
                                                                      chromosome,
                                                                      chromosome_start,
                                                                      chromosome_end)
        return bedtuple_result

    def call_sites_from_fasta(self,
                              fasta_file: str,
                              genomic_label: bool,
                              to_file: str = None):
        """Given a fasta file input, calls sites in BED format.

        With a fasta file as input, calls sites from every sequence. Returns
        output in BED format. By default, the chromosome is the full header and
        the start and end are relatve to sequence length. With the genomic
        label parameter set to True, the header is parsed if given in the
        format: Chromosome:start-end. By default the method returns a pandas
        DataFrame, but with the to_file parameter set to a file location, the
        function will write output to the location. This differs in that called
        site data is not kept in memory and is written in groups of sites
        called per sequence.

        :param fasta_file: A fasta file input of sequences
        :type fasta_file: str
        :param genomic_label: Flag for if the headers are genomic coordinates
        :type genomic_label: bool
        :param to_file: File location output. If specified, outputs sites
            for each sequence one set at a time.
        :type to_file: str
        :returns: By default a Pandas DataFrame of called sites in bed format but if to_file is specified, writes the results to a file in bed format
        """
        fasta_file_object = open(fasta_file)
        # If to_file, open output location, else initialize empty list
        if to_file is None:
            results = []
            output_file_object = None
        elif isinstance(to_file, str):
            output_file_object = open(to_file, 'w')
        elif to_file == sys.stdout:
            output_file_object = sys.stdout
        # For each pair of rows in the fasta file
        reading = True
        while reading:
            header, sequence = ctrlf_tf.parse_utils.read_fasta_entry(fasta_file_object)
            if header:
                # Parse the chromosome, start position, and sequence
                chromosome, start = ctrlf_tf.parse_utils.parse_fasta_header(header.rstrip(), genomic_label)
                sequence = sequence.rstrip().upper()
                # Call sites and convert them to bed format
                bed_list = self.call_sites_as_bed(sequence, chromosome, start)
                # If outputing to a file or stdout, write output
                if output_file_object:
                    for site in bed_list:
                        formated_site = "\t".join([str(i) for i in site])
                        output_file_object.write(formated_site + '\n')
                # Otherwise add to result list
                else:
                    results += bed_list
            else:
                reading = False
        fasta_file_object.close()
        # If output is a file, close it
        if isinstance(to_file, str):
            output_file_object.close()
        elif to_file is None:
            return pd.DataFrame(results)
