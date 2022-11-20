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
                 version=VERSION):
        """Initialize the class."""
        self.dataframe: pd.DataFrame = classified_dataframe
        self.negative_threshold: ThresholdTuple = negative_threshold
        self.positive_threshold: ThresholdTuple = positive_threshold
        self.version: str = version

    @classmethod
    def classify_from_dataframe(cls,
                                df: pd.DataFrame,
                                method: str = "kde_z4",
                                z_negative: int = 3,
                                z_positive: int = 4,
                                sequence_start: int = None,
                                sequence_end: int = None,
                                ln_transform: bool = False):
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

        By default, classification is done using the kde_z4 method. This
        calculates the Kernal Density Estimate (Gaussian) of the
        distribution and selects the maximum result of each value as an
        input. Then the distance between the minimum value and this value
        is multiplied by 2 and that is the kde threshold. This is compared
        to a modified z-score based on medians of 4 (assuming a Gaussian
        distribution). The negative and positive thresholds are determined
        by the minimum and maximum of these results respectively.

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
        :returns: ClassifiedDeBruijn Object
        """
        df = df.rename(columns={df.columns[0]: "Values",
                                df.columns[1]: "Sequence"})
        if ln_transform:
            df["Values"] = df["Values"].apply(lambda x: np.log(x))
        if method == "kde_z4":
            negative, positive = ctrlf_tf.threshold_utils.thresholds_kde_zscore(df["Values"])
        elif method == "z-score":
            negative = ctrlf_tf.threshold_utils.threshold_from_zscore(df['Values'],
                                                     z_negative)
            positive = ctrlf_tf.threshold_utils.threshold_from_zscore(df["Values"],
                                                     z_positive)
        else:
            raise ValueError("Method must be 'kde_z4' or 'z-score'")
        group_tuple = ctrlf_tf.threshold_utils.classify_values(df["Values"],
                                              negative.value,
                                              positive.value)
        df["Group"] = group_tuple
        df["Sequence"] = df["Sequence"].apply(lambda x:
                                              x[sequence_start:sequence_end])
        return cls(df, negative, positive)

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
        # Parse dataframe
        dataframe = pd.read_csv(StringIO(dataframe_string), sep='\t')
        return cls(dataframe, negative_tuple, positive_tuple, version)

    def _save(self, output_obj):
        headers = ("#Classified Sequences\n",
                   f"#Version: {self.version}\n",
                   f"#Negative Definition: {self.negative_threshold.definition}\n",
                   f"#Negative Threshold: {self.negative_threshold.value}\n",
                   f"#Positive Definition: {self.positive_threshold.definition}\n",
                   f"#Positive Threshold: {self.positive_threshold.value}\n",
                   "#Dataframe:\n")
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
                 gap_thresholds={0: 0.35, 1: 0.40, 2: 0.43}):
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
                                                                                            self.init_parameters)
            self.optimal_parameters = ctrlf_tf.optimize_utils.optimal_parameters_from_df(self.parameter_dataframe, self.init_parameters)


    @classmethod
    def load_from_file(cls, file_path: str):
        """Load an Optimize class from a saved text file."""
        # Read whole file as string
        with open(file_path) as file_obj:
            file_string = file_obj.read()
        # Seperate into groups
        meta_data_string, dataframe_strings = file_string.split("#Parameter DataFrame:\n")
        fpr_string, init_params_string = meta_data_string.split("#Initial Parameters:")
        # Read init params
        init_parameters = ctrlf_tf.ctrlf_core.AlignParameters.from_str_iterable(init_params_string.strip().split('\n'))
        fpr_threshold = float(fpr_string.split(': ')[1].strip())
        parameters_string, dataframe_strings = \
            dataframe_strings.split("#Classified_Dataframe:\n")
        classified_df_string, tpr_fpr_string = \
            dataframe_strings.split("#TPR_FPR_Dataframe:\n")
        # Parse classified debruijn and parameter dataframes
        classified_dataframe = pd.read_csv(StringIO(classified_df_string), sep='\t')
        parameter_dataframe = pd.read_csv(StringIO(parameters_string),
                                          sep='\t')
        # Parse tpr_fpr_dictionary from dataframe
        tpr_fpr_dataframe = pd.read_csv(StringIO(tpr_fpr_string), sep='\t')
        tpr_fpr_dictionary = {}
        for key, dataframe in tpr_fpr_dataframe.groupby(by="ID"):
            tpr_fpr_dictionary[key] = dataframe
        # Return class instance
        return cls(align_parameters=init_parameters,
                   classified_df=classified_dataframe,
                   fpr_threshold=fpr_threshold,
                   parameter_dataframe=parameter_dataframe,
                   tpr_fpr_dictionary=tpr_fpr_dictionary)

    def save_to_file(self, file_path: str):
        """Save optimized parameter information to a text file.

        The saved file contains the attributes of the Optimize object,
        including:
        1) Inital AlignParameter
        2) Input Classified deBruijn DataFrame
        3) DataFrame of all benchmarked parameters up to the optimal one
        4) DataFrame of all TPR and FPR results

        The save file generated from this method can be used by the
        load_from_file method to create a new Optimize object.

        :param file_path: File path to save the attribute information
        """
        with open(file_path, 'w') as file_obj:
            file_obj.write(f"#FPR threshold: {self.fpr_threshold}\n#Initial Parameters:\n")
        self.init_parameters.save_parameters(file_path, mode='a')
        with open(file_path, 'a') as file_obj:
            file_obj.write("#Parameter DataFrame:\n")
        self.parameter_dataframe.to_csv(file_path,
                                        sep='\t',
                                        index=False,
                                        mode='a')
        with open(file_path, 'a') as file_obj:
            file_obj.write("#Classified_Dataframe:\n")
        self.classified_df.to_csv(file_path,
                                  sep='\t',
                                  index=False,
                                  mode='a')
        with open(file_path, 'a') as file_obj:
            file_obj.write("#TPR_FPR_Dataframe:\n")
        tpr_fpr_dataframe = ctrlf_tf.optimize_utils.meta_tpr_fpr_dataframe(self.tpr_fpr_dictionary)
        tpr_fpr_dataframe.to_csv(file_path,
                                 sep='\t',
                                 index=False,
                                 mode='a')
