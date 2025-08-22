"""CtrlF-TF Command Line Interface."""


import argparse
import sys

import pandas as pd
import ctrlf_tf as cftf


MAIN_DESCRIPTION = """CtrlF-TF: Transcription Factor Binding Site Search via Aligned Sequences.

Supports two data workflows:
  • PBM (Protein Binding Microarray): Pre-computed k-mer scores
  • HT-SELEX: Bias-corrected logits output with sequences from QBiC-SELEX

Use --data parameter with 'classify' and 'optimize' commands to specify workflow type."""


def _add_output_to_parser(parser):
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=None,
                        help="Output file, stdout by default")
    return parser


def _add_alignparams_to_parser(parser):
    alignment = parser.add_argument_group("Input-Output Parameters")
    alignment.add_argument("-a",
                           "--align_model",
                           required=True,
                           type=str,
                           help="PWM model the k-mers are aligned to.")
    alignment.add_argument("-k",
                           "--kmer_file",
                           required=False,  # Not required for SELEX data or when using SELEX optimization input
                           type=str,
                           help="K-mer file. PBM: Required. HT-SELEX: Auto-selected from optimization output when using --optimize_input (-oi).")
    alignment.add_argument("-oi",
                           "--optimize_input",
                           type=str,
                           default=None,
                           help="""Load parameters from 'ctrlf optimize' output file.
                           HT-SELEX workflows: Automatically selects optimal k-mer file (e.g., results_kmers_k8.txt).
                           PBM workflows: Still requires manual --kmer_file specification.""")
    alignment.add_argument("-m",
                           "--meme",
                           action="store_true",
                           help="Boolean flag if the model is in MEME format")
    alignment.add_argument("-o",
                           "--output",
                           type=str,
                           default=None,
                           help="Output file location, standard output by default")
    alignment_settings = parser.add_argument_group("Alignment Settings.")
    alignment_settings.add_argument("-rc",
                                    "--range_consensus",
                                    type=str,
                                    default=None,
                                    help="""Representation of the orientation and spatial
                                    position of a binding site as a string to automatically
                                    define the advanced parameters: '-r', '-p' and  '-cg'.
                                    Can use {A, C, G, T, N, and .} characters where {N, .}
                                    represent wildcard positions with the '.' indicating that
                                    the k-mers do not need to describe that position for calling
                                    sites. Example: The string of TTCCNGGAA will find the top scoring
                                    position in the PWM as the '-r' and define the model as
                                    palindromic and use the '-p' flag. Using TTCC.GGAA will do
                                    the same but define the '.' position as a core gap (-cg argument).""")
    alignment_settings.add_argument("-g",
                                    "--gap_limit",
                                    type=int,
                                    default=0,
                                    help="""The number of allowed gaps in a k-mer. Must be
                                    greater than or equal to zero. By default this value is
                                    zero and aligns all non-gapped k-mers.""")
    alignment_settings.add_argument("-t",
                                    "--threshold",
                                    type=float,
                                    help="""Convenience argument to only align k-mers equal to
                                    or above a given threshold score. By default all k-mers provided
                                    are aligned.""")
    alignment_settings.add_argument("-tc",
                                    "--threshold_column",
                                    type=str,
                                    help="""Convenience argument to select which column to use for the
                                    '-t' argument. By default uses the third column.""")
    alignment_adv_settings = parser.add_argument_group("Alignment Advanced Settings")
    alignment_adv_settings.add_argument("-p",
                                        "--palindrome",
                                        action="store_true",
                                        help="""Flag for if the alignment should be in 'palindrome mode', where both
                                        orientations for each k-mer are aligned to the model instead of choosing an
                                        orientation based on the alignment score. This flag will override the automatic
                                        chosen setting from the '-oi' argument if used in conjunction.""")
    alignment_adv_settings.add_argument("-r",
                                        "--range",
                                        nargs=2,
                                        type=int,
                                        default=(0, 0),
                                        help="""Description position range within the input PWM model (1-based),
                                        default is the whole size of the input PWM. This argument will override the
                                        automatic chosen setting from the '-oi' argument if used in conjunction.""")
    alignment_adv_settings.add_argument("-cg",
                                        "--core_gap",
                                        nargs='*',
                                        type=int,
                                        default=None,
                                        help="""Positions within the core range (1-based, relative to core range '-r')
                                        that are not part of the kmer description of a site. Must be given with the '-r'
                                        argument. This argument will override the automatic chosen setting from the '-oi'
                                        argument if used in conjunction.""")
    # Experimental setting.
    alignment_adv_settings.add_argument("-oit", "--opt_threshold_type", type=str, default="Distance", choices=("Distance", "AtFPR"))
    return parser


def _config_optimize_parser(parser):
    parser = _add_alignparams_to_parser(parser)
    
    # Data type selection
    data_group = parser.add_argument_group("Data Type")
    data_group.add_argument("--data",
                           type=str,
                           choices=["pbm", "selex"],
                           required=True,
                           help="Data type (REQUIRED): 'pbm' for pre-computed k-mer scores, 'selex' for HT-SELEX bias-corrected scores from QBiC-SELEX")
    
    optimization = parser.add_argument_group("Optimization Parameters",
                                         "Parameters for both PBM and HT-SELEX optimization workflows")
    optimization.add_argument("-c",
                              "--classify_file",
                              type=str,
                              help="Output file from 'ctrlf classify' (required for both PBM and HT-SELEX workflows)")
    optimization.add_argument("-fpr",
                              "--fpr_threshold",
                              type=float,
                              default=0.01,
                              help="FPR target for optimization on de Bruijn sequences.")
    optimization.add_argument("-gthr",
                              "--gap_thresholds",
                              nargs='*',
                              default=[0.35, 0.35, 0.38],
                              type=float,
                              help="Rank score thresholds for optimizing gaps (Default is E-score based)")
    
    # SELEX-specific parameters  
    selex_group = parser.add_argument_group("HT-SELEX Parameters", 
                                        "Parameters specific to HT-SELEX workflow (bias-corrected scores from QBiC-SELEX)")
    selex_group.add_argument("-i",
                            "--input_file", 
                            type=str,
                            help="HT-SELEX input file with sequence and score columns")
    selex_group.add_argument("--buffer_zone",
                            type=float,
                            default=0.1,
                            help="Classification buffer zone around zero. Scores > buffer_zone = positive, < -buffer_zone = negative (default: 0.1)")
    selex_group.add_argument("--kmer_length_range",
                            type=str,
                            default="7-10",
                            help="K-mer lengths to test during optimization, format: 'min-max' (default: '7-10')")
    selex_group.add_argument("--scoring_method",
                            type=str,
                            choices=["median", "average"],
                            default="average",
                            help="K-mer scoring method (default: average)")
    selex_group.add_argument("--sample_size",
                            type=int,
                            default=100000,
                            help="Sample size for optimization (default: 100000)")
    selex_group.add_argument("--sample_method",
                            type=str,
                            choices=["balanced", "stratified", "random"],
                            default="balanced",
                            help="Sampling method (default: balanced)")
    selex_group.add_argument("--kmer_threshold",
                            type=float,
                            default=0.0,
                            help="K-mer score threshold - only k-mers with scores above this value are used (default: 0.0)")
    selex_group.add_argument("--kmer_threshold_column",
                            type=str,
                            default="score",
                            help="Column name for k-mer thresholding (default: score)")
    selex_group.add_argument("--keep_all_kmers",
                            action="store_true",
                            default=False,
                            help="Keep all generated k-mer files. Default: False (auto-cleanup, only optimal k-mer file + _best.txt remain)")
    return parser


def _config_align_compile_parser(parser):
    parser = _add_alignparams_to_parser(parser)
    return parser


def _config_call_parser(parser):
    """Configure the arguments for the call subprogram parser."""
    call_io = parser.add_argument_group("Input-Output Arguments")
    call_io.add_argument("-i",
                         "--input_model",
                         required=True,
                         type=str,
                         help="Input of Aligned k-mers or Compiled Solutions.")
    call_io.add_argument("-f",
                         "--fasta_file",
                         required=True,
                         type=str,
                         help="Fasta file of DNA sequences")
    call_io.add_argument("-o",
                         "--output",
                         type=str,
                         default=None,
                         help="Output file location, standard output by default.")
    call_settings = parser.add_argument_group("Settings")
    call_settings.add_argument("-gc",
                               "--genomic_coordinates",
                               action="store_true",
                               help="Parse fasta input for genomic coordinates.")
    return parser


def _config_classify_parser(parser):
    """Configure the arguments for the classify subprogram parser."""
    classify_io = parser.add_argument_group("Input-Output Arguments")
    classify_io.add_argument("--data",
                             type=str,
                             choices=["pbm", "selex"],
                             required=True,
                             help="Data type (REQUIRED): 'pbm' for pre-computed k-mer scores, 'selex' for HT-SELEX bias-corrected scores from QBiC-SELEX")
    classify_io.add_argument("-i",
                             "--input_file",
                             required=True,
                             type=str,
                             help="""Input file format depends on --data type:
• PBM: Tab-separated, no header, columns: [score, kmer, reverse_complement]  
• HT-SELEX: CSV with header, required columns: 'sequence', 'score'""")
    classify_io.add_argument("-o",
                             "--output",
                             type=str,
                             default=None,
                             help="Output file, stdout by default.")
    classify_settings = parser.add_argument_group("Settings")
    classify_settings.add_argument("-m",
                                   "--method",
                                   type=str,
                                   choices=["kde", "z-score", "kde_z4"],
                                   default='kde',
                                   help="Classification method, default = kde.")
    classify_settings.add_argument("-z",
                                   "--z_scores",
                                   nargs=2,
                                   type=float,
                                   default=(3, 4),
                                   help="Z-scores to use if classifying by 'z-score'")
    classify_settings.add_argument("-kde_p",
                                   "--kde_positive_ratio",
                                   type=float,
                                   default=1,
                                   help="Multiplier of kde negative threshold to obtain positive threshold.")
    classify_conv = parser.add_argument_group("Convenience Settings")
    classify_conv.add_argument("-sr",
                               "--sequence_range",
                               nargs=2,
                               type=int,
                               help="""Subsets all sequences in the input file by the given range (1-base).""")
    classify_conv.add_argument("-ln",
                               "--ln_transform",
                               action="store_true",
                               help="Natural log transform the values prior to classification.")

    classify_conv.add_argument("-pv",
                               "--positive_values_only",
                               action="store_true",
                               help="Convenience argument that subsets the input values for positive values only prior to classification.")
    
    # HT-SELEX-specific parameters
    selex_settings = parser.add_argument_group("HT-SELEX Settings",
                                              "Parameters specific to HT-SELEX classification workflow")
    selex_settings.add_argument("--buffer_zone",
                               type=float,
                               default=0.05,
                               help="Classification buffer zone around zero. Scores > buffer_zone = positive, < -buffer_zone = negative (default: 0.1)")
    selex_settings.add_argument("--sample_size",
                               type=int,
                               default=100000,
                               help="Sample size for classification (default: 100000)")
    selex_settings.add_argument("--sample_method",
                               type=str,
                               choices=["balanced", "stratified", "random"],
                               default="balanced",
                               help="Sampling method (default: balanced)")
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
                                         help="Align k-mers to PWM model (works with both PBM and HT-SELEX outputs).")
    align_parser = _config_align_compile_parser(align_parser)
    # Compile program parser definition
    compile_parser = subparsers.add_parser("compile",
                                           help="Compile k-mers into aligned consensus sites (works with both PBM and HT-SELEX outputs).")
    compile_parser = _config_align_compile_parser(compile_parser)
    # Optimization
    optimize_parser = subparsers.add_parser("optimize",
                                            help="Optimize alignment parameters. PBM: uses existing k-mers. HT-SELEX: tests multiple k-mer lengths (7-10).")
    optimize_parser = _config_optimize_parser(optimize_parser)
    # Call program parser definition
    call_parser = subparsers.add_parser("callsites",
                                        help="Call binding sites in sequences (works with both PBM and HT-SELEX outputs).")
    call_parser = _config_call_parser(call_parser)
    # Classify program parser
    classify_parser = subparsers.add_parser("classify",
                                            help="Classify sequences for optimization. PBM: KDE/z-score methods. HT-SELEX: buffer zone classification.")
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
    else:
        start_parameter = 0
        end_parameter = 0
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

def _init_alignparameters_from_args(args):
    if args.optimize_input:
        opt_obj = cftf.Optimize.load_from_file(args.optimize_input)
        parameters = opt_obj.optimal_parameters
        parameters.pwm_file = args.align_model
        
        # Handle SELEX k-mer file selection automatically
        if hasattr(parameters, 'optimal_kmer_length') and hasattr(opt_obj, 'selex_metadata'):
            # This is SELEX optimization output - auto-select k-mer file
            if args.kmer_file:
                # User provided k-mer file - use it (manual override)
                parameters.kmer_file = args.kmer_file
            else:
                # Auto-generate k-mer file path based on optimal length
                import os
                optimization_dir = os.path.dirname(args.optimize_input)
                base_name = os.path.splitext(os.path.basename(args.optimize_input))[0]
                optimal_kmer_file = os.path.join(optimization_dir, f"{base_name}_kmers_k{parameters.optimal_kmer_length}.txt")
                
                if os.path.exists(optimal_kmer_file):
                    parameters.kmer_file = optimal_kmer_file
                    print(f"Auto-selected optimal k-mer file: {optimal_kmer_file} (k={parameters.optimal_kmer_length})")
                else:
                    raise FileNotFoundError(f"Expected k-mer file not found: {optimal_kmer_file}. "
                                          f"Please provide k-mer file with --kmer_file (-k)")
        else:
            # PBM optimization or manual k-mer file required
            if not args.kmer_file:
                raise ValueError("--kmer_file (-k) is required when using PBM optimization output")
            parameters.kmer_file = args.kmer_file
            
        # Apply other parameter overrides
        if args.opt_threshold_type == "Distance":
            parameters.threshold = opt_obj.distance_based_optimal_threshold()
        if args.threshold:
            parameters.threshold = args.threshold
        if args.threshold_column:
            parameters.threshold_column = args.threshold_column
        if args.palindrome:
            parameters.palindrome = args.palindrome  # Fix typo
        if args.meme:
            parameters.pwm_file_format = "MEME"
    else:
        parameters = _args_to_align_parameters(args)
    return parameters


def _align_program(args):
    parameters = _init_alignparameters_from_args(args)
    aligned_kmers = cftf.AlignedKmers.from_parameters(parameters)
    aligned_kmers.save_alignment(args.output)


def _compile_program(args):
    parameters = _init_alignparameters_from_args(args)
    compiled_kmers = cftf.CtrlF.from_parameters(parameters)
    compiled_kmers.compile_all_solutions()
    compiled_kmers.save_compiled_sites(args.output)


def _save_combined_selex_optimization(output_file: str, all_opt_objs: dict, 
                                      performance_summary: list, best_opt_obj):
    """Save combined SELEX optimization results for all k-mer lengths.
    
    Creates a comprehensive optimization output that includes:
    1. Performance summary for all k-mer lengths tested
    2. Detailed parameter dataframes for all k-mer lengths
    3. Best optimization result marked clearly
    
    :param output_file: Output file path
    :param all_opt_objs: Dictionary mapping k_length -> optimization object
    :param performance_summary: List of performance dictionaries for each k-length
    :param best_opt_obj: Best performing optimization object
    """
    import pandas as pd
    
    with open(output_file, 'w') as file_obj:
        # Write header and metadata from best optimization
        file_obj.write(f"#FPR threshold: {best_opt_obj.fpr_threshold}\n")
        
        # Add SELEX preprocessing metadata if present
        if hasattr(best_opt_obj, 'selex_metadata') and best_opt_obj.selex_metadata:
            file_obj.write(f"#SELEX scoring method: {best_opt_obj.selex_metadata.get('scoring_method', 'N/A')}\n")
            file_obj.write(f"#SELEX buffer zone: {best_opt_obj.selex_metadata.get('buffer_zone', 'N/A')}\n")
            file_obj.write(f"#SELEX sample size: {best_opt_obj.selex_metadata.get('sample_size', 'N/A')}\n")
            file_obj.write(f"#SELEX sample method: {best_opt_obj.selex_metadata.get('sample_method', 'N/A')}\n")
        
        # Write K-mer Performance Summary
        file_obj.write("#K-mer Length Performance Summary:\n")
        summary_df = pd.DataFrame(performance_summary)
        summary_df.to_csv(file_obj, sep='\t', index=False)
        file_obj.write("\n")
        
        # Write best k-mer length info
        best_entry = max(performance_summary, key=lambda x: x['performance'])
        file_obj.write(f"#Best K-mer Length: {best_entry['k_length']} "
                      f"({best_entry['metric_name']} = {best_entry['performance']:.6f})\n")
        
        # Write initial parameters from best optimization
        file_obj.write("#Initial Parameters:\n")
        
    # Save best optimization parameters (reuse existing method)
    best_opt_obj.init_parameters.save_parameters(output_file, mode='a')
    
    # Add standard Parameter DataFrame section for align compatibility
    with open(output_file, 'a') as file_obj:
        file_obj.write("#Parameter DataFrame:\n")
    
    # Save best k-mer parameter dataframe in standard format
    if hasattr(best_opt_obj, 'parameter_dataframe') and not best_opt_obj.parameter_dataframe.empty:
        best_opt_obj.parameter_dataframe.to_csv(output_file, sep='\t', index=False, mode='a')
    
    # Add classified dataframe section for align compatibility
    with open(output_file, 'a') as file_obj:
        file_obj.write("#Classified_Dataframe:\n")
    
    # Save best k-mer classified dataframe
    if hasattr(best_opt_obj, 'classified_df') and not best_opt_obj.classified_df.empty:
        best_opt_obj.classified_df.to_csv(output_file, sep='\t', index=False, mode='a')
    
    # Add TPR/FPR dataframe section for align compatibility  
    with open(output_file, 'a') as file_obj:
        file_obj.write("#TPR_FPR_Dataframe:\n")
    
    # Save best k-mer TPR/FPR data in standard format
    if hasattr(best_opt_obj, 'tpr_fpr_dictionary') and best_opt_obj.tpr_fpr_dictionary:
        for key, dataframe in best_opt_obj.tpr_fpr_dictionary.items():
            dataframe.to_csv(output_file, sep='\t', index=False, header=True, mode='a')
    
    with open(output_file, 'a') as file_obj:
        file_obj.write("#Combined Parameter DataFrame (All K-mer Lengths):\n")
    
    # Combine all parameter dataframes with k-length identification
    combined_dfs = []
    for k_length in sorted(all_opt_objs.keys()):
        opt_obj = all_opt_objs[k_length]
        if hasattr(opt_obj, 'parameter_dataframe') and not opt_obj.parameter_dataframe.empty:
            df_copy = opt_obj.parameter_dataframe.copy()
            # Ensure Kmer_Length column exists and is set correctly
            df_copy['Kmer_Length'] = k_length
            # Add prefix to ID to distinguish between k-lengths
            df_copy['ID'] = df_copy['ID'].apply(lambda x: f"k{k_length}_{x}")
            combined_dfs.append(df_copy)
    
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_df.to_csv(output_file, sep='\t', index=False, mode='a')
    
    # Add TPR/FPR data for all k-lengths
    with open(output_file, 'a') as file_obj:
        file_obj.write("#TPR FPR DataFrame (All K-mer Lengths):\n")
        
        for k_length in sorted(all_opt_objs.keys()):
            opt_obj = all_opt_objs[k_length]
            if hasattr(opt_obj, 'tpr_fpr_dictionary') and opt_obj.tpr_fpr_dictionary:
                file_obj.write(f"#K-mer Length {k_length} TPR/FPR Data:\n")
                for key, dataframe in opt_obj.tpr_fpr_dictionary.items():
                    df_copy = dataframe.copy()
                    # Prefix ID to distinguish k-lengths
                    df_copy['ID'] = f"k{k_length}_{key}"
                    df_copy.to_csv(file_obj, sep='\t', index=False, header=True)


def _optimize_program(args):
    """Optimize CtrlF-TF alginment parameters.

    Optimize CtrlF-TF alignment parameters, returning
    an optimization report file containing the performance
    information for each set of hyperparameters attempted.
    """
    # Validate data type and required parameters
    if args.data == "pbm" and not args.kmer_file:
        raise ValueError("PBM data requires --kmer_file (-k) parameter")
    
    if args.data == "selex":
        # SELEX optimization workflow - similar to PBM but with k-mer length optimization
        if not args.classify_file:
            raise ValueError("SELEX optimization requires --classify_file (-c) parameter (run 'ctrlf classify --data selex' first)")
        if not args.input_file:
            raise ValueError("SELEX optimization requires --input_file (-i) parameter (raw sequences + scores)")
        
        # Set SELEX-specific defaults: use single threshold since only gap=0 is used
        if args.kmer_threshold == 0.0:  # Default value, user didn't override
            args.kmer_threshold = 0.0  # Keep permissive default for SELEX
        # For SELEX, use kmer_threshold as the single gap threshold (gap=0 only)
        args.gap_thresholds = [args.kmer_threshold]  # Single threshold for gap=0
        
        # Load classified sequences (from classify step)
        classified_seqs = cftf.ClassifiedSequences.load_from_file(args.classify_file)
        
        # Read raw SELEX data for k-mer generation
        selex_df = pd.read_csv(args.input_file)
        sequences = selex_df['sequence'].tolist()
        scores = selex_df['score'].tolist()
        
        # Parse k-mer length range
        k_min, k_max = map(int, args.kmer_length_range.split('-'))
        k_lengths = list(range(k_min, k_max + 1))
        
        best_performance = -1
        best_k_length = None
        best_kmer_file = None
        
        import os
        base_output = os.path.splitext(args.output)[0]
        
        # Stage 1: Pre-generate k-mer files for all k-lengths (time-saving approach)
        print("STAGE 1: Generating k-mer files for all k-lengths...")
        kmer_files = {}
        import ctrlf_tf.selex_utils
        for k_length in k_lengths:
            print(f"Generating k-mers from {len(sequences)} sequences (k={k_length})...")
            kmer_df = ctrlf_tf.selex_utils.generate_kmers_from_sequences(
                sequences, scores, k_length, args.scoring_method
            )
            
            # Save k-mer file for this length
            kmer_output = f"{base_output}_kmers_k{k_length}.txt"
            kmer_export_df = pd.DataFrame({
                'kmer': kmer_df['kmer'],
                'rev_comp': kmer_df['rev_comp'],
                'score': kmer_df['score']
            })
            kmer_export_df.to_csv(kmer_output, sep='\t', index=False, header=True)
            kmer_files[k_length] = kmer_output
            print(f"Saved {len(kmer_df)} k-mers to {kmer_output}")
        
        # Stage 2: Optimize parameters using pre-generated k-mer files  
        print("\nSTAGE 2: Optimizing parameters for each k-length...")
        best_opt_obj = None  # Track best optimization object
        all_opt_objs = {}  # Track all optimization objects by k-length
        kmer_performance_summary = []  # Track performance summary for all k-lengths
        for k_length in k_lengths:
            print(f"Testing k-mer length: {k_length}")
            kmer_output = kmer_files[k_length]
            
            # Create parameters for this k-length (similar to PBM)
            parameters = _args_to_align_parameters(args)
            parameters.kmer_file = kmer_output
            parameters.gap_limit = 0  # Force ungapped for SELEX
            # Set SELEX-specific threshold parameters
            parameters.threshold = args.kmer_threshold
            parameters.threshold_column = args.kmer_threshold_column
            
            # Sample classified sequences for faster optimization (parameter testing only)
            if len(classified_seqs.dataframe) > args.sample_size:
                print(f"Sampling {args.sample_size} sequences from {len(classified_seqs.dataframe)} for parameter optimization...")
                sampled_classified_df = classified_seqs.dataframe.sample(n=args.sample_size, random_state=42)
            else:
                print(f"Using all {len(classified_seqs.dataframe)} sequences for parameter optimization...")
                sampled_classified_df = classified_seqs.dataframe
            
            # Run optimization for this k-length (SELEX ungapped, so simplified gap_thresholds)
            gap_thresholds = {0: args.kmer_threshold}  # Single threshold for gap=0 only
            
            # Create SELEX metadata for this optimization
            selex_metadata = {
                'scoring_method': args.scoring_method,
                'buffer_zone': args.buffer_zone,
                'sample_size': args.sample_size,
                'sample_method': args.sample_method
            }
                
            opt_obj = cftf.Optimize(align_parameters=parameters,
                                   classified_df=sampled_classified_df,
                                   fpr_threshold=args.fpr_threshold,
                                   gap_thresholds=gap_thresholds,
                                   selex_metadata=selex_metadata,
                                   kmer_length=k_length)
            
            # Use standard optimization performance metric (same as PBM)
            # The optimal parameters are determined by optimize_utils.optimal_parameters_from_df
            try:
                # Get the performance metric that was used for optimization
                if hasattr(opt_obj, 'parameter_dataframe') and not opt_obj.parameter_dataframe.empty:
                    # Use the same metric that determines optimal_parameters
                    if 'pAUROC' in opt_obj.parameter_dataframe.columns:
                        performance = opt_obj.parameter_dataframe.iloc[-1]['pAUROC']
                        metric_name = "pAUROC"
                    elif 'AUROC' in opt_obj.parameter_dataframe.columns:
                        performance = opt_obj.parameter_dataframe.iloc[-1]['AUROC']
                        metric_name = "AUROC"
                    else:
                        # Fallback to first numeric column
                        numeric_cols = opt_obj.parameter_dataframe.select_dtypes(include=[float, int]).columns
                        if len(numeric_cols) > 0:
                            performance = opt_obj.parameter_dataframe.iloc[-1][numeric_cols[0]]
                            metric_name = numeric_cols[0]
                        else:
                            performance = 0.0
                            metric_name = "Unknown"
                else:
                    performance = 0.0
                    metric_name = "No optimization data"
            except Exception as e:
                print(f"Warning: Could not extract performance metric for k={k_length}: {e}")
                performance = 0.0
                metric_name = "Error"
            
            print(f"K-mer length {k_length}: {metric_name} = {performance:.4f}")
            
            # Store all optimization objects and performance summary
            all_opt_objs[k_length] = opt_obj
            kmer_performance_summary.append({
                'k_length': k_length,
                'performance': performance,
                'metric_name': metric_name,
                'kmer_file': kmer_output
            })
            
            # Keep track of best performing k-length
            if performance > best_performance:
                best_performance = performance
                best_k_length = k_length
                best_kmer_file = kmer_output
                best_opt_obj = opt_obj  # Keep the best optimization object
        
        # Save the best optimization result and k-mer file (consistent with PBM workflow)
        if best_kmer_file and best_opt_obj is not None:
            final_kmer_output = f"{base_output}_kmers_best.txt"
            import shutil
            shutil.copy(best_kmer_file, final_kmer_output)
            print(f"Best k-mer length: {best_k_length} ({metric_name} = {best_performance:.4f})")
            print(f"Saved best k-mer file: {final_kmer_output}")
            
            # Clean up non-optimal k-mer files unless --keep_all_kmers is specified
            if args.keep_all_kmers:
                print(f"Kept all k-mer files: {', '.join([f'k{k}' for k in kmer_files.keys()])}")
            else:
                import os
                cleaned_files = []
                kept_files = []
                for k_length, kmer_file in kmer_files.items():
                    if k_length != best_k_length:
                        try:
                            os.remove(kmer_file)
                            cleaned_files.append(f"k{k_length}")
                        except OSError as e:
                            print(f"Warning: Could not remove {kmer_file}: {e}")
                    else:
                        kept_files.append(f"k{k_length}")
                
                if cleaned_files:
                    print(f"Cleaned up non-optimal k-mer files: {', '.join(cleaned_files)}")
                if kept_files:
                    print(f"Kept optimal k-mer file: {', '.join(kept_files)} (also saved as _kmers_best.txt)")
            
            # Save combined optimization results for all k-mer lengths
            _save_combined_selex_optimization(args.output, all_opt_objs, kmer_performance_summary, best_opt_obj)
            print(f"Saved SELEX optimization results (all k-lengths) to: {args.output}")
        
        # Continue with standard workflow (don't exit early) - SELEX now follows PBM pattern
        # The saved optimization file can be used by align function with --optimize_input
        return
        
    else:
        # PBM optimization workflow (original)
        parameters = _args_to_align_parameters(args)
        classified_seqs = cftf.ClassifiedSequences.load_from_file(args.classify_file)
    
    gap_thresholds = {}
    for idx, i in enumerate(args.gap_thresholds):
        gap_thresholds[idx] = i
        
    opt_obj = cftf.Optimize(align_parameters=parameters,
                            classified_df=classified_seqs.dataframe,
                            fpr_threshold=args.fpr_threshold,
                            gap_thresholds=gap_thresholds)
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
    """Runs the CtrlF-TF classification task."""
    
    if args.data == "selex":
        # SELEX data processing: expects columns named "sequence" and "score"
        input_df = pd.read_csv(args.input_file)
        
        # Validate required columns
        if 'sequence' not in input_df.columns or 'score' not in input_df.columns:
            raise ValueError("SELEX data must have 'sequence' and 'score' columns")
        
        sequences = input_df['sequence'].tolist()
        scores = input_df['score'].tolist()
        
        # Apply sequence range if specified
        if args.sequence_range:
            string_start = args.sequence_range[0] - 1  # Convert to 0-based
            string_end = args.sequence_range[1]
            sequences = [seq[string_start:string_end] for seq in sequences]
        
        # Use SELEX classification method (sample for optimization)
        results = cftf.ClassifiedSequences.classify_selex_from_dataframe(
            sequences=sequences,
            scores=scores,
            buffer_zone=args.buffer_zone,
            sample_size=args.sample_size,
            sample_method=args.sample_method
        )
    else:
        # PBM data processing (original behavior) - no header
        input_df = pd.read_csv(args.input_file,
                               sep='\t',
                               header=None)
        # if positive argument, subset by positive values
        if args.positive_values_only:
            input_df = input_df[input_df[0] >= 0].reset_index(drop=True)
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
