"""CtrlF-TF Command Line Interface."""


import argparse
import sys

import pandas as pd
import ctrlf_tf as cftf


MAIN_DESCRIPTION = ("CtrlF-TF: TF Binding Site Search via Aligned Sequences.")


def _config_compile_parser(parser):
    """Configure the arguments for the align subprogram."""
    required = parser.add_argument_group("required arguments")
    optimization = parser.add_argument_group("optimization arguments")
    required.add_argument("-a",
                          "--align_model",
                          required=True,
                          type=str,
                          help="Alignment model")
    required.add_argument("-k",
                          "--kmer_file",
                          required=True,
                          type=str,
                          help="Kmer file in a")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=None,
                        help="Output file, stdout by default")
    parser.add_argument("-p",
                        "--palindrome",
                        action="store_true",
                        help="Boolean flag if the model is palindromic")
    parser.add_argument("-m",
                        "--meme",
                        action="store_true",
                        help="Boolean flag if the model is in MEME format")
    parser.add_argument("-g",
                        "--gap_limit",
                        type=int,
                        default=0,
                        help="""Filters the kmer dataframe for kmers with a
                                 max count of gaps. Must be 0 or a positive
                                 integer (default is 0).""")
    parser.add_argument("-t",
                        "--threshold",
                        type=float,
                        help=("Threshold score for kmers to align, if no "
                              "-tc argument provided, uses 3rd column."))
    parser.add_argument("-tc",
                        "--threshold_column",
                        type=str,
                        help=("Column in the kmer dataframe to use for the "
                              "rank score (Default is 3rd column)."))
    parser.add_argument("-r",
                        "--range",
                        nargs=2,
                        type=int,
                        default=(0, 0),
                        help="""Core range for PWM model (1-based), default is
                                 the whole input""")
    parser.add_argument("-cg",
                        "--core_gap",
                        nargs='*',
                        type=int,
                        default=None,
                        help="""Positions within the core range (1-based,
                        relative to core range '-r') that are not part of the
                        kmer description of a site. Must be given with the '-r'
                        argument""")
    parser.add_argument("-rc",
                        "--range_consensus",
                        type=str,
                        default=None,
                        help="""Definition of -r and -cg using the alignment of a
                        consensus site instead. A '.' character in the
                        consensus string indicates a -cg position.""")
    optimization.add_argument("-opt",
                              "--optimize",
                              action="store_true",
                              help="Boolean flag to perform optimization on classified sequences.")
    optimization.add_argument("-c",
                              "--classify_file",
                              type=str,
                              help="Output file from 'ctrlf classify'")
    optimization.add_argument("-fpr",
                              "--fpr_threshold",
                              type=float,
                              default=0.01,
                              help="FPR target for optimization on de bruijn data.")
    optimization.add_argument("-orep",
                              "--output_report",
                              type=str,
                              default=None,
                              help="Output file location for Optimization Report (Default is no output)")
    optimization.add_argument("-gthr",
                              "--gap_thresholds",
                              nargs='*',
                              default=[0.35, 0.40, 0.43],
                              type=float,
                              help="Rank score thresholds for optimizing gaps (Default is E-score based)")
    return parser


def _config_call_parser(parser):
    """Configure the arguments for the call subprogram parser."""
    required = parser.add_argument_group("required arguments")
    required.add_argument("-c",
                          "--consensus_sites",
                          required=True,
                          type=str,
                          help="Compiled consensus sites file.")
    required.add_argument("-f",
                          "--fasta_file",
                          required=True,
                          type=str,
                          help="Fasta file of DNA sequences")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=None,
                        help="Output file, stdout by default")
    parser.add_argument("-gc",
                        "--genomic_coordinates",
                        action="store_true",
                        help="Parse fasta input for genomic coordinates")
    return parser


def _config_classify_parser(parser):
    """Configure the arguments for the classify subprogram parser."""
    required = parser.add_argument_group("required arguments")
    required.add_argument("-i",
                          "--input_file",
                          required=True,
                          type=str,
                          help="Input file for classification.")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=None,
                        help="Output file, stdout by default.")
    parser.add_argument("-m",
                        "--method",
                        type=str,
                        choices=["kde_z4", "z-score"],
                        default='kde_z4',
                        help="Classification method, default = kde_z4.")
    parser.add_argument("-z",
                        "--z_scores",
                        nargs=2,
                        type=float,
                        default=(3, 4),
                        help="Z-scores to use if classifying by 'z-score'")
    parser.add_argument("-sr",
                        "--sequence_range",
                        nargs=2,
                        type=int,
                        help="Sequence position range to use.")
    parser.add_argument("-ln",
                        "--ln_transform",
                        action="store_true",
                        help="Natural log transform the values prior to classification.")
    return parser


def _cli_parser():
    """Parse arguments for ctrlf_cli."""
    # Define main parser
    main_parser = argparse.ArgumentParser(description=MAIN_DESCRIPTION)
    main_parser.add_argument('-v', '--version',action="store_true", help="Return version.")
    subparsers = main_parser.add_subparsers(dest="program",
                                            help="Available subcommands:")
    # Align program parser definition
    align_parser = subparsers.add_parser("compile",
                                         help="Align and compile k-mers into aligned sequences containing sites.")
    align_parser = _config_compile_parser(align_parser)
    # Call program parser definition
    call_parser = subparsers.add_parser("sitecall",
                                        help="Call sites using the aligned sequences containing sites.")
    call_parser = _config_call_parser(call_parser)
    # Classify program parser
    classify_parser = subparsers.add_parser("classify",
                                            help="Classify a table of values and sequences for optimization.")
    classify_parser = _config_classify_parser(classify_parser)
    return main_parser


def _args_to_align_parameters(args) -> cftf.AlignParameters:
    """Convert output from argument parser to AlignParameters."""
    # Convert meme boolean flag to format choice
    if args.meme:
        file_format = "MEME"
    else:
        file_format = "Tabular"
    # Convert range argument to AlignParameters format
    if args.range:
        start_parameter = args.range[0]
        end_parameter = args.range[1]
    result = cftf.AlignParameters(kmer_file=args.kmer_file,
                                  pwm_file=args.align_model,
                                  pwm_file_format=file_format,
                                  core_start=start_parameter,
                                  core_end=end_parameter,
                                  core_gaps=args.core_gap,
                                  range_consensus=args.range_consensus,
                                  gap_limit=args.gap_limit,
                                  threshold=args.threshold,
                                  threshold_column=args.threshold_column,
                                  palindrome=args.palindrome)
    return result


def _align_parser_validation(parser, args) -> bool:
    """Validate argument inputs, raise parser.error if invalid."""
    if args.gap_limit is not None and args.gap_limit < 0:
        parser.error("'-g' given a negative integer, needs to be 0 or more")
    if args.core_gap is not None and args.range is None:
        parser.error("'-cg' was given without specifying '-r'")
    if args.range_consensus and args.range != (0, 0):
        parser.error("-r must be specified with either -r or -rc, not both.")
    return True


def _compile_program(args):
    parameters = _args_to_align_parameters(args)
    if args.optimize:
        gap_thresholds = {}
        for idx, i in enumerate(args.gap_thresholds):
            gap_thresholds[idx] = i
        classified_seqs = cftf.ClassifiedSequences.load_from_file(args.classify_file)
        opt_obj = cftf.Optimize(align_parameters=parameters,
                                classified_df=classified_seqs.dataframe,
                                fpr_threshold=args.fpr_threshold,
                                gap_thresholds=gap_thresholds)
        parameters = opt_obj.optimal_parameters
        if args.output_report:
            opt_obj.save_to_file(args.output_report)
    compiled_kmers = cftf.CompiledKmers.from_parameters(parameters)
    compiled_kmers.save_compiled_sites(args.output)


def _call_program(args):
    ctrlf_object = cftf.CtrlF.from_compiled_sites(args.consensus_sites)
    if args.output:
        output = args.output
    else:
        output = sys.stdout
    ctrlf_object.call_sites_from_fasta(args.fasta_file,
                                       args.genomic_coordinates,
                                       output)


def _classify_program(args):
    input_df = pd.read_csv(args.input_file,
                           sep='\t',
                           header=None)
    if args.sequence_range:
        string_start = args.sequence_range[0]
        string_end = args.sequence_range[1]
    else:
        string_start = None
        string_end = None
    results = cftf.ClassifiedSequences.classify_from_dataframe(df=input_df,
                                                               method=args.method,
                                                               z_negative=args.z_scores[0],
                                                               z_positive=args.z_scores[1],
                                                               sequence_start=string_start,
                                                               sequence_end=string_end,
                                                               ln_transform=args.ln_transform)
    if args.output:
        results.save_to_file(args.output)
    else:
        results.save_to_stdout()


def main():
    """CtrlF-TF CLI logic."""
    parser = _cli_parser()
    arguments = parser.parse_args()
    # If the main program is run
    if arguments.program == "compile":
        _align_parser_validation(parser, arguments)
        _compile_program(arguments)
    elif arguments.program == "classify":
        _classify_program(arguments)
    elif arguments.version:
        print(cftf.__version__)
    elif arguments.program == "sitecall":
        _call_program(arguments)


if __name__ == "__main__":
    main()
