"""CtrlF-TF Command Line Interface."""


import argparse
import sys

import pandas as pd
import ctrlf_tf as cftf


MAIN_DESCRIPTION = ("CtrlF-TF: Transcription Factor Binding Site Search via Aligned Sequences.")


def _add_output_to_parser(parser):
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=None,
                        help="Output file, stdout by default")
    return parser


def _add_alignparams_to_parser(parser):
    alignment = parser.add_argument_group("Alignment Parameters")
    alignment.add_argument("-a",
                          "--align_model",
                          required=True,
                          type=str,
                          help="Alignment model")
    alignment.add_argument("-k",
                          "--kmer_file",
                          required=True,
                          type=str,
                          help="File with k-mers and a ranked score.")
    alignment.add_argument("-p",
                        "--palindrome",
                        action="store_true",
                        help="Boolean flag if the model is palindromic")
    alignment.add_argument("-m",
                        "--meme",
                        action="store_true",
                        help="Boolean flag if the model is in MEME format")
    alignment.add_argument("-g",
                        "--gap_limit",
                        type=int,
                        default=0,
                        help="""Filters the kmer dataframe for kmers with a
                                 max count of gaps. Must be 0 or a positive
                                 integer (default is 0).""")
    alignment.add_argument("-t",
                        "--threshold",
                        type=float,
                        help=("Threshold score for kmers to align, if no "
                              "-tc argument provided, uses 3rd column."))
    alignment.add_argument("-tc",
                        "--threshold_column",
                        type=str,
                        help=("Column in the kmer dataframe to use for the "
                              "rank score (Default is 3rd column)."))
    alignment.add_argument("-r",
                        "--range",
                        nargs=2,
                        type=int,
                        default=(0, 0),
                        help="""Core range for PWM model (1-based), default is
                                 the whole input""")
    alignment.add_argument("-cg",
                        "--core_gap",
                        nargs='*',
                        type=int,
                        default=None,
                        help="""Positions within the core range (1-based,
                        relative to core range '-r') that are not part of the
                        kmer description of a site. Must be given with the '-r'
                        argument""")
    alignment.add_argument("-rc",
                        "--range_consensus",
                        type=str,
                        default=None,
                        help="""Definition of -r and -cg using the alignment of a
                        consensus site instead. A '.' character in the
                        consensus string indicates a -cg position.""")
    return parser


def _config_optimize_parser(parser):
    parser = _add_alignparams_to_parser(parser)
    parser = _add_output_to_parser(parser)
    optimization = parser.add_argument_group("Optimization Parameters")
    optimization.add_argument("-c",
                              "--classify_file",
                              type=str,
                              help="Output file from 'ctrlf classify'")
    optimization.add_argument("-fpr",
                              "--fpr_threshold",
                              type=float,
                              default=0.01,
                              help="FPR target for optimization on de bruijn data.")
    optimization.add_argument("-gthr",
                              "--gap_thresholds",
                              nargs='*',
                              default=[0.35, 0.35, 0.38],
                              type=float,
                              help="Rank score thresholds for optimizing gaps (Default is E-score based)")
    return parser


def _config_align_compile_parser(parser):
    parser = _add_alignparams_to_parser(parser)
    parser = _add_output_to_parser(parser)
    parser.add_argument("-oi",
                        "--optimize_input",
                        type=str,
                        default=None,
                        help="Use parameters from an optimization report as input in addition to -a and -k.")
    return parser


def _config_call_parser(parser):
    """Configure the arguments for the call subprogram parser."""
    required = parser.add_argument_group("required arguments")
    required.add_argument("-i",
                          "--input_model",
                          required=True,
                          type=str,
                          help="Input of Aligned k-mers or Compiled Solutions.")
    required.add_argument("-f",
                          "--fasta_file",
                          required=True,
                          type=str,
                          help="Fasta file of DNA sequences")
    parser.add_argument("-gc",
                        "--genomic_coordinates",
                        action="store_true",
                        help="Parse fasta input for genomic coordinates")
    parser = _add_output_to_parser(parser)
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
                        choices=["kde", "z-score", "kde_z4"],
                        default='kde',
                        help="Classification method, default = kde.")
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
    parser.add_argument("-kde_p",
                        "--kde_positive_ratio",
                        type=float,
                        default=1,
                        help="Multiplier of kde negative threshold to obtain positive threshold.")
    return parser


def _cli_parser():
    """Parse arguments for ctrlf_cli."""
    # Define main parser
    main_parser = argparse.ArgumentParser(description=MAIN_DESCRIPTION)
    main_parser.add_argument('-v', '--version',action="store_true", help="Return version.")
    subparsers = main_parser.add_subparsers(dest="program",
                                            help="Available subcommands:")
    # Align program parser definition
    align_parser = subparsers.add_parser("align",
                                         help="Align k-mers to a model.")
    align_parser = _config_align_compile_parser(align_parser)
    # Compile program parser definition
    compile_parser = subparsers.add_parser("compile",
                                         help="Compile k-mers into aligned consensus sites by generating all possible solutions.")
    compile_parser = _config_align_compile_parser(compile_parser)
    # Optimization
    optimize_parser = subparsers.add_parser("optimize",
                                            help="Optimize alignment parameters based on de Bruijn sequence classification.")
    optimize_parser = _config_optimize_parser(optimize_parser)
    # Call program parser definition
    call_parser = subparsers.add_parser("callsites",
                                        help="Call sites using aligned k-mers or compiled solutions.")
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


def _align_program(args):
    parameters = _args_to_align_parameters(args)
    if args.optimize_input:
        opt_obj = cftf.Optimize.load_from_file(args.optimize_input)
        parameters = opt_obj.optimal_parameters
        parameters.pwm_file = args.align_model
        parameters.kmer_file = args.kmer_file
    aligned_kmers = cftf.AlignedKmers.from_parameters(parameters)
    aligned_kmers.save_alignment(args.output)


def _compile_program(args):
    parameters = _args_to_align_parameters(args)
    if args.optimize_input:
        opt_obj = cftf.Optimize.load_from_file(args.optimize_input)
        parameters = opt_obj.optimal_parameters
        parameters.pwm_file = args.align_model
        parameters.kmer_file = args.kmer_file
    compiled_kmers = cftf.CtrlF.from_parameters(parameters)
    compiled_kmers.compile_all_solutions()
    compiled_kmers.save_compiled_sites(args.output)


def _optimize_program(args):
    parameters = _args_to_align_parameters(args)
    gap_thresholds = {}
    for idx, i in enumerate(args.gap_thresholds):
        gap_thresholds[idx] = i
    classified_seqs = cftf.ClassifiedSequences.load_from_file(args.classify_file)
    opt_obj = cftf.Optimize(align_parameters=parameters,
                            classified_df=classified_seqs.dataframe,
                            fpr_threshold=args.fpr_threshold,
                            gap_thresholds=gap_thresholds)
    parameters = opt_obj.optimal_parameters
    opt_obj.save_to_file(args.output)


def _call_program(args):
    # Determine if to init CtrlF from k-mers or solutions
    with open(args.input_model) as read_obj:
        lines = read_obj.readlines()
        if lines[1].startswith("#Palindrome"):
            ctrlf_object = cftf.CtrlF.from_alignment_file(args.input_model)
        else:
            ctrlf_object = cftf.CtrlF.from_compiled_sites(args.input_model)
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
                                                               ln_transform=args.ln_transform,
                                                               kde_positive_ratio=args.kde_positive_ratio)
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
    elif arguments.program == "align":
        _align_parser_validation(parser, arguments)
        _align_program(arguments)
    elif arguments.program == "optimize":
        _align_parser_validation(parser, arguments)
        _optimize_program(arguments)
    elif arguments.program == "classify":
        _classify_program(arguments)
    elif arguments.version:
        print(cftf.__version__)
    elif arguments.program == "callsites":
        _call_program(arguments)


if __name__ == "__main__":
    main()
