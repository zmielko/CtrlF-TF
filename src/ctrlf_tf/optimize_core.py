"""Optimize AlignParameters based on de Bruijn classification.

Set of classes to optimize parameter choice based on performance
on the classification of de Bruijn sequences.

1) ClassifiedDeBruijn; Classifies de Bruijn sequences into positive,
    negative, and ambiguous groups based on a specified method.
2) Optimize: Optimizes AlignParameters based on performance on
    classification of de Bruijn sequences at a specified false
    positive rate.
3) ThresholdTuple: NamedTuple for a threshold description and value
4) IterationTuple: NamedTuple used internally by Optimize for iterations
    of parameters.

ClassifiedDeBruijn is used to format de bruijn sequence data into input
for Optimize, though it is not strictly required as long as the dataframe
has the same format. The initialization process of Optimize creates an
optimal_parameters attribute in the form of an AlignedParameters instance.
It also has a parameters_dataframe which shows all of the parameters that
were benchmarked.
"""

from collections import namedtuple
from io import StringIO
import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import ctrlf_tf.optimize_utils
import ctrlf_tf.threshold_utils
import ctrlf_tf.ctrlf_core

VERSION = ctrlf_tf.ctrlf_core.__version__

ThresholdTuple = namedtuple("ThresholdTuple", ["definition", "value"])


class ClassifiedSequences:
    """Classify de Bruijn sequences.

    Takes an input dataframe in the format (with or without headers):

    ===== =========
    Value Sequence
    ===== =========
    50000 CAATCG...
    49554 ATCGAT...
    ...   ...
    ===== =========

    and creates a classified dataframe in the format:

    ===== ========= =====
    Value Sequence  Group
    ===== ========= =====
    50000 CAATCG... '+'
    49554 ATCGAT... '-'
    ...   ...       ...
    ===== ========= =====

    The Group column indicates if it is a bound sequence, not bound or
    ambiguous (+/-/.).

    Meta data is accessed with the following attributes:

    1) negative_threshold: ThresholdTuple that describes the definition and
        value of the negative group "-"
    2) positive_threshold: ThresholdTuple that described the definition and
        value of the positive group "+"
    3) version: Version of ctrlf_tf used to generate the classification
    """

    def __init__(self,
                 classified_dataframe: pd.DataFrame,
                 negative_threshold: ThresholdTuple,
                 positive_threshold: ThresholdTuple,
                 version=VERSION,
                 classification_params: dict = None):
        """Initialize the class."""
        self.dataframe: pd.DataFrame = classified_dataframe
        self.negative_threshold: ThresholdTuple = negative_threshold
        self.positive_threshold: ThresholdTuple = positive_threshold
        self.version: str = version
        # Store all classification parameters to avoid redundancy in optimize function
        self.classification_params: dict = classification_params or {}

    @classmethod
    def classify_from_dataframe(cls,
                                df: pd.DataFrame,
                                method: str = "kde",
                                z_negative: int = 3,
                                z_positive: int = 4,
                                sequence_start: int = None,
                                sequence_end: int = None,
                                ln_transform: bool = False,
                                kde_positive_ratio = 1):
        """Classifies a set of de Bruijn sequences using a given method.

        Factory method to create a ClassifiedDeBruijn object from a pandas
        DataFrame. The dataframe must have columns in the following format
        with or without headers:

        ===== =========
        Value Sequence
        ===== =========
        50000 CAATCG...
        49554 ATCGAT...
        ...   ...
        ===== =========

        By default, classification is done using the kde method. This
        calculates the Kernal Density Estimate (Gaussian) of the
        distribution and selects the maximum result of each value as an
        input. Then the distance between the minimum value and this value
        is multiplied by 2 and that is the kde threshold. This is compared
        to a percent multiplier of the total distance.

        Otherwise, classification can be done using the "z-score" method.
        This calculated the modified z-score for a given negative and
        positive value as the negative and positive thresholds respectively.
        

        :param dataframe: Dataframe of the de Bruijn sequence information
        :type dataframe: Pandas DataFrame
        :param method: Method of classification (kde_z4 or z-score)
        :type method: str
        :param z_negative: Negative z-score value if using z-score method
            (default = 3)
        :type z_negative: float
        :param z_positive: Positive z-score value if using z-score method
            (default = 4)
        :type z_positive: float
        :param sequence_start: Optional start index for the de Bruijn sequence
            (default None)
        :type sequence_start: int
        :param sequence_end: Optional end index for the de Bruijn sequence
            (default None)
        :type sequence_end: int
        :param kde_positive_ratio: Positive threshold from multiplying value by negative threshold.
        :type kde_positive_ratio: float
        :returns: ClassifiedDeBruijn Object
        """
        print(f"Classifying {len(df)} sequences using {method} method...")
        df = df.rename(columns={df.columns[0]: "Values",
                                df.columns[1]: "Sequence"})
        if ln_transform:
            print("Applying log transformation to values...")
            df["Values"] = df["Values"].apply(lambda x: np.log(x))
        if method == "kde_z4":
            print("Computing KDE z-score thresholds...")
            negative, positive = ctrlf_tf.threshold_utils.thresholds_kde_zscore(df["Values"])
        elif method == "z-score":
            print(f"Computing z-score thresholds (negative: {z_negative}, positive: {z_positive})...")
            negative = ctrlf_tf.threshold_utils.threshold_from_zscore(df['Values'],
                                                     z_negative)
            positive = ctrlf_tf.threshold_utils.threshold_from_zscore(df["Values"],
                                                     z_positive)
        elif method == "kde":
            print(f"Computing KDE thresholds (positive ratio: {kde_positive_ratio})...")
            negative, positive = ctrlf_tf.threshold_utils.threshold_from_kde(df["Values"], kde_positive_ratio)
        else:
            raise ValueError("Method must be 'kde_z4' or 'z-score'")
        
        print(f"Thresholds: negative={negative.value:.3f}, positive={positive.value:.3f}")
        group_tuple = ctrlf_tf.threshold_utils.classify_values(df["Values"],
                                              negative.value,
                                              positive.value)
        df["Group"] = group_tuple
        df["Sequence"] = df["Sequence"].apply(lambda x:
                                              x[sequence_start:sequence_end])
        
        # Count classifications
        group_counts = df['Group'].value_counts()
        print(f"Classification results: {group_counts.to_dict()}")
        print(f"PBM classification completed: {len(df)} sequences ready for optimization")
        
        # Store classification parameters
        classification_params = {
            'data_type': 'pbm',
            'method': method,
            'z_negative': z_negative,
            'z_positive': z_positive,
            'sequence_start': sequence_start,
            'sequence_end': sequence_end,
            'ln_transform': ln_transform,
            'kde_positive_ratio': kde_positive_ratio
        }
        return cls(df, negative, positive, classification_params=classification_params)

    @classmethod
    def classify_selex_from_dataframe(cls,
                                     sequences: list,
                                     scores: list,
                                     buffer_zone: float = 0.05,
                                     sample_size: int = 100000,
                                     sample_method: str = "balanced"):
        """Factory method for SELEX data classification (sample only).
        
        Similar to PBM classify - only generates classified sequences for optimization,
        no k-mer generation (that happens in optimize step).
        
        :param sequences: List of DNA sequences
        :param scores: List of sequence scores
        :param buffer_zone: SELEX buffer zone around zero (default: 0.05)
        :param sample_size: Sample size for optimization
        :param sample_method: Sampling method ("balanced" or "random")
        :returns: ClassifiedSequences object with sampled SELEX data
        """
        import ctrlf_tf.selex_utils
        import ctrlf_tf.threshold_utils
        
        print(f"Classifying {len(sequences)} SELEX sequences (buffer zone: {buffer_zone})...")
        # Classify sequence scores for sampling
        seq_groups, negative_thresh, positive_thresh = ctrlf_tf.threshold_utils.classify_selex_values(
            scores, buffer_zone
        )
        
        # Count classifications
        group_counts = {'+': 0, '-': 0, '.': 0}
        for group in seq_groups:
            group_counts[group] += 1
        print(f"Classification: positive={group_counts['+']}, negative={group_counts['-']}, ambiguous={group_counts['.']}")
        
        # Get optimization sample (this is the main output like PBM classify)
        print(f"Sampling {sample_size} sequences for optimization using {sample_method} method...")
        sample_seqs, sample_scores, sample_groups = ctrlf_tf.selex_utils.get_optimization_sample(
            sequences, scores, list(seq_groups), sample_size, sample_method
        )
        print(f"Optimization sample: {len(sample_seqs)} sequences")
        
        # Create DataFrame in standard format for classification (no k-mers yet)
        df = pd.DataFrame({
            "Values": sample_scores,
            "Sequence": sample_seqs,
            "Group": sample_groups
        })
        
        print(f"SELEX classification completed: {len(df)} sequences ready for optimization")
        
        # Store classification parameters
        classification_params = {
            'data_type': 'selex',
            'buffer_zone': buffer_zone,
            'sample_size': sample_size,
            'sample_method': sample_method
        }
        return cls(df, negative_thresh, positive_thresh, classification_params=classification_params)

    @classmethod
    def load_from_file(cls, file_path: str):
        """Initialize a ClassifiedDeBruijn object from a file.

        Initialize a CLassifiedDeBruijn object from a text file. The file
        has to be in the same format as the output from the save_to_file
        method.

        :param file_path: File path to load the ClassifiedDeBruijn from
        :returns: ClassifiedDeBruijn object
        """
        file_object = open(file_path, 'r')
        meta_data_string, dataframe_string = file_object.read().split("Dataframe:\n")
        file_object.close()
        meta_data = meta_data_string.split("\n")
        
        # Parse version
        version = meta_data[1].split(": ")[1]
        
        # Parse threshold information
        negative_definition = meta_data[2].split(": ")[1]
        negative_threshold = float(meta_data[3].split(": ")[1])
        positive_definition = meta_data[4].split(": ")[1]
        positive_threshold = float(meta_data[5].split(": ")[1])
        negative_tuple = ThresholdTuple(negative_definition, negative_threshold)
        positive_tuple = ThresholdTuple(positive_definition, positive_threshold)
        
        # Parse classification parameters (if present)
        classification_params = {}
        for line in meta_data:
            if line.startswith('#') and ':' in line and not line.startswith('#Classified') and not line.startswith('#Version') and not line.startswith('#Negative') and not line.startswith('#Positive') and not line.startswith('#Dataframe'):
                try:
                    key = line.split('#')[1].split(':')[0].strip()
                    value_str = line.split(':', 1)[1].strip()
                    # Try to convert to appropriate type
                    if value_str.lower() in ['true', 'false']:
                        value = value_str.lower() == 'true'
                    elif value_str.replace('.', '').replace('-', '').isdigit():
                        value = float(value_str) if '.' in value_str else int(value_str)
                    else:
                        value = value_str
                    classification_params[key] = value
                except:
                    continue  # Skip lines that can't be parsed
        
        # Parse dataframe
        dataframe = pd.read_csv(StringIO(dataframe_string), sep='\t')
        return cls(dataframe, negative_tuple, positive_tuple, version, classification_params)

    def _save(self, output_obj):
        headers = ["#Classified Sequences\n",
                   f"#Version: {self.version}\n",
                   f"#Negative Definition: {self.negative_threshold.definition}\n",
                   f"#Negative Threshold: {self.negative_threshold.value}\n",
                   f"#Positive Definition: {self.positive_threshold.definition}\n",
                   f"#Positive Threshold: {self.positive_threshold.value}\n"]
        
        # Add classification parameters to avoid redundancy in optimize
        if self.classification_params:
            headers.append("#Classification Parameters:\n")
            for key, value in self.classification_params.items():
                headers.append(f"#{key}: {value}\n")
        
        headers.append("#Dataframe:\n")
        
        for header in headers:
            output_obj.write(header)
        # Save dataframe
        self.dataframe.to_csv(output_obj, sep='\t', index=False, mode='a')

    def save_to_file(self, file_path: str):
        """Save classification data to a file in tabular format with meta data.

        Saves the information from the class attributes to a text file output.
        This file can be loaded by load_from_file to initialize a
        ClassifiedDeBruijn object.

        :param file_path: Relative path to save the attribute information to
        """
        # Save header meta data
        with open(file_path, 'w') as file_object:
            self._save(file_object)

    def save_to_stdout(self):
        self._save(sys.stdout)


class Optimize:
    """Optimize AlignParameters using classified de Bruijn sequences.

    Optimizes AlignParameters using de Bruijn sequence data classified
    into positive and negative probes. Optimization based on AUROC at a target
    false positive rate (default = 0.01).
    """

    def __init__(self,
                 align_parameters: ctrlf_tf.ctrlf_core.AlignParameters,
                 classified_df: pd.DataFrame,
                 fpr_threshold: float = 0.01,
                 version: str = VERSION,
                 parameter_dataframe: pd.DataFrame = None,
                 tpr_fpr_dictionary: dict = None,
                 optimal_parameters: ctrlf_tf.ctrlf_core.AlignParameters = None,
                 gap_thresholds={0: 0.35, 1: 0.35, 2: 0.35},
                 selex_metadata: dict = None,
                 kmer_length: int = None):
        """Initialize Optimize class.

        Takes as input AlignParameters, a classified_df sequence,
        and an optional FPR threshold. The classified_df input is
        a dataframe with or without headers that has the following columns:

        ===== ========= =====
        Value Sequence  Group
        ===== ========= =====
        50000 CAATCG... +
        49554 ATCGAT... -
        ...   ...       ...
        ===== ========= =====

        The optimiztion assigns a score to every sequence based on the maximum
        threshold score of called sites. An AUROC is generated for initial
        parameters. If the alignment is palindrome, expands the dimension of
        the PWM in both directions and picks the maximum AUROC among the new
        parameters and the previous. If a new parameter is chosen, the
        dimensions are increased again and the loop repeats until the previous
        parameter is the maximum. If non-palindromic, compares left, right, and
        both directions for each iteration.

        :param align_parameters: Alignment parameters to optimize
        :type align_parameters: ctrlf_tf.AlignParameters
        :param classified_df: Classified debruijn sequence into positive,
            negative, and ambiguous binding sequences
        :type classified_df: pandas.DataFrame
        :param fpr_threshold: Threshold false positive rate (default = 0.01)
        :type fpr_threshold: float
        """
        # Input arguments
        self.init_parameters = align_parameters
        self.classified_df = classified_df
        self.gap_thresholds = gap_thresholds
        self.fpr_threshold = fpr_threshold
        self.version = version
        self.selex_metadata = selex_metadata
        self.kmer_length = kmer_length
        if parameter_dataframe is not None and tpr_fpr_dictionary:
            self.parameter_dataframe = parameter_dataframe
            self.tpr_fpr_dictionary = tpr_fpr_dictionary
            self.optimal_parameters = ctrlf_tf.optimize_utils.optimal_parameters_from_df(self.parameter_dataframe, self.init_parameters)
        elif parameter_dataframe is not None or tpr_fpr_dictionary:
            raise ValueError(("paramater_dataframe and tpr_fpr_dictionary must"
                              "be specified together"))
        else:
            # Run Optimization
            gap_limit = self.init_parameters.gap_limit
            if gap_limit is None:
                gap_limit = 0
            self.parameter_dataframe, self.tpr_fpr_dictionary = ctrlf_tf.optimize_utils.optimize_parameters(gap_limit,
                                                                                            self.fpr_threshold,
                                                                                            self.gap_thresholds,
                                                                                            self.classified_df,
                                                                                            self.init_parameters,
                                                                                            self.kmer_length)
            self.optimal_parameters = ctrlf_tf.optimize_utils.optimal_parameters_from_df(self.parameter_dataframe, self.init_parameters)


    @classmethod
    def load_from_file(cls, file_path: str):
        """Load an Optimize class from a saved text file with SELEX support."""
        # Read whole file as string
        with open(file_path) as file_obj:
            file_string = file_obj.read()
        
        # Parse SELEX metadata from header if present
        selex_metadata = {}
        lines = file_string.split('\n')
        for line in lines:
            if line.startswith('#SELEX'):
                key_value = line[1:].split(': ', 1)
                if len(key_value) == 2:
                    key = key_value[0].replace('SELEX ', '').lower().replace(' ', '_')
                    selex_metadata[key] = key_value[1]
        
        # Seperate into groups
        meta_data_string, dataframe_strings = file_string.split("#Parameter DataFrame:\n")
        fpr_string, init_params_string = meta_data_string.split("#Initial Parameters:")
        # Read init params
        init_parameters = ctrlf_tf.ctrlf_core.AlignParameters.from_str_iterable(init_params_string.strip().split('\n'))
        
        # Extract FPR threshold from first line (handles both PBM and SELEX formats)
        fpr_line = fpr_string.split('\n')[0]
        fpr_threshold = float(fpr_line.split(': ')[1].strip())
        def parse_section_until_next_header(content):
            """Parse section content until the next header (line starting with #)."""
            lines = content.split('\n')
            data_lines = []
            for line in lines:
                if line.startswith('#'):
                    break
                data_lines.append(line)
            return '\n'.join(data_lines)
        
        parameters_string, dataframe_strings = \
            dataframe_strings.split("#Classified_Dataframe:\n")
        classified_df_string, tpr_fpr_full_string = \
            dataframe_strings.split("#TPR_FPR_Dataframe:\n")
        
        # Parse each section robustly by stopping at next header
        parameters_string = parse_section_until_next_header(parameters_string)
        classified_df_string = parse_section_until_next_header(classified_df_string)
        tpr_fpr_string = parse_section_until_next_header(tpr_fpr_full_string)
        
        # Parse classified debruijn and parameter dataframes
        classified_dataframe = pd.read_csv(StringIO(classified_df_string), sep='\t')
        parameter_dataframe = pd.read_csv(StringIO(parameters_string),
                                          sep='\t')
        # Convert empty list into list instead of a string
        parameter_dataframe["Core_Gaps"] = parameter_dataframe["Core_Gaps"].apply(lambda x: ctrlf_tf.parse_utils.parse_parameter_str("Core_Gaps", x))
        # Parse tpr_fpr_dictionary from dataframe
        tpr_fpr_dataframe = pd.read_csv(StringIO(tpr_fpr_string), sep='\t')
        tpr_fpr_dictionary = {}
        for key, dataframe in tpr_fpr_dataframe.groupby(by="ID"):
            tpr_fpr_dictionary[key] = dataframe
        # Return class instance
        instance = cls(align_parameters=init_parameters,
                       classified_df=classified_dataframe,
                       fpr_threshold=fpr_threshold,
                       parameter_dataframe=parameter_dataframe,
                       tpr_fpr_dictionary=tpr_fpr_dictionary,
                       optimal_parameters=ctrlf_tf.optimize_utils.optimal_parameters_from_df(parameter_dataframe, init_parameters))
        
        if selex_metadata:
            instance.selex_metadata = selex_metadata
        
        return instance

    def save_to_file(self, file_path: str, input_files: dict = None):
        """Save optimized parameter information to a text file using unified format.

        Uses the unified optimization format that works with both PBM and SELEX workflows
        and is compatible with the align function's parsing logic.

        :param file_path: File path to save the attribute information
        :param input_files: Dictionary of input file paths (optional, for better documentation)
        """
        # Import here to avoid circular imports
        import ctrlf_tf.cli_prgm
        
        # Determine data type from metadata or classification params
        data_type = 'PBM'  # Default
        if hasattr(self, 'selex_metadata') and self.selex_metadata:
            data_type = 'SELEX'
        elif (hasattr(self, 'classified_df') and 
              hasattr(self.classified_df, 'classification_params') and
              self.classified_df.classification_params.get('data_type') == 'selex'):
            data_type = 'SELEX'
        
        # Prepare input files dictionary
        if input_files is None:
            input_files = {}
            if hasattr(self, 'init_parameters'):
                params = self.init_parameters
                input_files['PWM File'] = getattr(params, 'pwm_file', 'N/A')
                input_files['K-mer File'] = getattr(params, 'kmer_file', 'N/A')
                input_files['Classified File'] = 'N/A'  # This would be set by calling function
        
        # Call unified save function
        ctrlf_tf.cli_prgm._save_unified_optimization(
            output_file=file_path,
            opt_obj=self,
            data_type=data_type,
            input_files=input_files
        )

    def distance_based_optimal_threshold(self):
        return ctrlf_tf.optimize_utils.distance_adjusted_threshold(self.parameter_dataframe,
                                                                   self.tpr_fpr_dictionary,
                                                                   self.fpr_threshold)
