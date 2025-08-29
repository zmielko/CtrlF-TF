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
                            default=0.05,
                            help="Classification buffer zone around zero. Scores > buffer_zone = positive, < -buffer_zone = negative (default: 0.05)")
    selex_group.add_argument("--kmer_length_range",
                            type=str,
                            default="7-10",
                            help="K-mer lengths to test during optimization, format: 'min-max' (default: '7-10')")
    selex_group.add_argument("--scoring_method",
                            type=str,
                            choices=["median", "average"],
                            default="average",
                            help="K-mer scoring method (default: average)")
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
                            help="Keep all generated k-mer files. Default: False (auto-cleanup, only optimal k-mer file remains)")
    selex_group.add_argument("--use_kde_threshold",
                            action="store_true",
                            default=False,
                            help="Use KDE (Kernel Density Estimation) to automatically determine k-mer score threshold instead of fixed --kmer_threshold")
    selex_group.add_argument("--kde_positive_ratio",
                            type=float,
                            default=1.0,
                            help="Multiplier for KDE threshold calculation (default: 1.0)")
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
                               help="Classification buffer zone around zero. Scores > buffer_zone = positive, < -buffer_zone = negative (default: 0.05)")
    selex_settings.add_argument("--sample_size",
                               type=int,
                               default=100000,
                               help="Sample size for classification (default: 100000)")
    selex_settings.add_argument("--sample_method",
                               type=str,
                               choices=["balanced", "random"],
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

def _parse_unified_optimization_file(file_path: str):
    """Parse unified optimization file to detect data type and extract key information.
    
    :param file_path: Path to optimization file
    :returns: Tuple of (data_type, optimization_info dict)
    """
    optimization_info = {
        'input_files': {},
        'best_kmer_file': None,
        'performance_summary': {},
        'optimal_parameters': {},
        'optimization_parameters': {},
        'parameter_dataframe': None,
        'best_parameter_row': None
    }
    
    data_type = 'PBM'  # Default fallback
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect data type from unified header
            if line.startswith('#Data Type:'):
                data_type = line.split(':', 1)[1].strip()
            
            # Extract input files
            elif line.startswith('#PWM File:'):
                optimization_info['input_files']['PWM File'] = line.split(':', 1)[1].strip()
            elif line.startswith('#K-mer File:'):
                optimization_info['input_files']['K-mer File'] = line.split(':', 1)[1].strip()
            elif line.startswith('#Sequence Data:'):
                optimization_info['input_files']['Sequence Data'] = line.split(':', 1)[1].strip()
            elif line.startswith('#Classified File:'):
                optimization_info['input_files']['Classified File'] = line.split(':', 1)[1].strip()
                
            # Extract best k-mer file for SELEX
            elif line.startswith('#Best K-mer File:'):
                optimization_info['best_kmer_file'] = line.split(':', 1)[1].strip()
                
            # Extract performance info
            elif line.startswith('#Best pAUROC:'):
                optimization_info['performance_summary']['best_pauroc'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('#Best K-mer Length:'):
                # Extract k-mer length for SELEX
                parts = line.split(':', 1)[1].strip().split()
                if parts:
                    optimization_info['performance_summary']['best_kmer_length'] = int(parts[0])
                    
            # Extract optimal parameters from Performance Summary
            elif line.startswith('#Optimal Core Start:'):
                optimization_info['performance_summary']['optimal_core_start'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#Optimal Core End:'):
                optimization_info['performance_summary']['optimal_core_end'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#Optimal Gap Limit:'):
                optimization_info['performance_summary']['optimal_gap_limit'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#Optimal Threshold:'):
                optimization_info['performance_summary']['optimal_threshold'] = float(line.split(':', 1)[1].strip())
                
            # Extract optimization parameters
            elif line.startswith('#Range Consensus:'):
                optimization_info['optimization_parameters']['range_consensus'] = line.split(':', 1)[1].strip()
            elif line.startswith('#Palindrome:'):
                optimization_info['optimization_parameters']['palindrome'] = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('#PWM Reverse Complement:'):
                optimization_info['optimization_parameters']['pwm_reverse_complement'] = line.split(':', 1)[1].strip().lower() == 'true'
                
            # Extract initial parameters (from original parameters section)
            elif line.startswith('#pwm_file_format:'):
                optimization_info['optimal_parameters']['pwm_file_format'] = line.split(':', 1)[1].strip()
            elif line.startswith('#core_start:'):
                optimization_info['optimal_parameters']['core_start'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#core_end:'):
                optimization_info['optimal_parameters']['core_end'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#core_gaps:'):
                # Parse list format: [] or [1, 2, 3]
                gaps_str = line.split(':', 1)[1].strip()
                if gaps_str == '[]':
                    optimization_info['optimal_parameters']['core_gaps'] = []
                else:
                    # Simple parsing - could be enhanced if needed
                    optimization_info['optimal_parameters']['core_gaps'] = []
            elif line.startswith('#gap_limit:'):
                optimization_info['optimal_parameters']['gap_limit'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('#threshold:'):
                threshold_str = line.split(':', 1)[1].strip()
                if threshold_str != 'None':
                    optimization_info['optimal_parameters']['threshold'] = float(threshold_str)
                else:
                    optimization_info['optimal_parameters']['threshold'] = None
            elif line.startswith('#threshold_column:'):
                threshold_col = line.split(':', 1)[1].strip()
                optimization_info['optimal_parameters']['threshold_column'] = threshold_col if threshold_col != 'None' else None
            elif line.startswith('#palindrome:'):
                optimization_info['optimal_parameters']['palindrome'] = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('#pwm_reverse_complement:'):
                optimization_info['optimal_parameters']['pwm_reverse_complement'] = line.split(':', 1)[1].strip().lower() == 'true'
    
    # Parse Parameter DataFrame to find best pAUROC row
    _parse_parameter_dataframe(file_path, optimization_info)
    
    return data_type, optimization_info

def _parse_parameter_dataframe(file_path: str, optimization_info: dict):
    """Parse Parameter DataFrame section and find the best pAUROC row."""
    import pandas as pd
    from io import StringIO
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find Parameter DataFrame section
    if '#Parameter DataFrame:' not in content:
        return
    
    # Extract Parameter DataFrame section
    param_start = content.find('#Parameter DataFrame:')
    param_content = content[param_start:]
    
    # Find the end of Parameter DataFrame (next section starting with #)
    lines = param_content.split('\n')
    param_lines = [lines[1]]  # Header line (skip #Parameter DataFrame:)
    
    for line in lines[2:]:  # Data lines
        if line.strip().startswith('#') or line.strip() == '':
            break
        param_lines.append(line)
    
    if len(param_lines) < 2:  # Need at least header + one data row
        return
    
    # Parse as DataFrame
    param_df_str = '\n'.join(param_lines)
    try:
        param_df = pd.read_csv(StringIO(param_df_str), sep='\t')
        optimization_info['parameter_dataframe'] = param_df
        
        # Find row with best pAUROC
        if 'pAUROC' in param_df.columns and not param_df.empty:
            best_idx = param_df['pAUROC'].idxmax()
            best_row = param_df.loc[best_idx]
            optimization_info['best_parameter_row'] = best_row.to_dict()
            
            print(f"Found best pAUROC row: ID={best_row.get('ID', 'N/A')}, "
                  f"pAUROC={best_row.get('pAUROC', 'N/A'):.6f}, "
                  f"Core={best_row.get('Core_Start', 'N/A')}-{best_row.get('Core_End', 'N/A')}")
                  
    except Exception as e:
        print(f"Warning: Could not parse Parameter DataFrame: {e}")
        return

def _init_alignparameters_from_args(args):
    if args.optimize_input:
        # Parse unified optimization format to detect data type and extract information
        data_type, optimization_info = _parse_unified_optimization_file(args.optimize_input)
        print(f"Detected optimization format: {data_type}")
        
        # Extract optimal parameters from unified format instead of loading old format
        optimal_params = optimization_info.get('optimal_parameters', {})
        performance_summary = optimization_info.get('performance_summary', {})
        
        # Priority: Use optimal parameters from best Parameter DataFrame row
        best_row = optimization_info.get('best_parameter_row')
        range_consensus = optimization_info.get('optimization_parameters', {}).get('range_consensus')
        
        if best_row and 'Core_Start' in best_row and 'Core_End' in best_row:
            # Use optimal core positions from best pAUROC row in Parameter DataFrame
            core_start = int(best_row['Core_Start'])
            core_end = int(best_row['Core_End'])
            # Use the optimal threshold from the same row
            optimal_threshold = best_row.get('Score_Threshold')
            range_consensus = None  # Override range consensus with explicit optimal positions
            print(f"Using optimal core positions from best pAUROC row: {core_start}-{core_end}")
            if optimal_threshold is not None:
                print(f"Using optimal threshold from best row: {optimal_threshold}")
        elif range_consensus and range_consensus != 'None':
            # Fallback to range consensus if no Parameter DataFrame available
            core_start = 0
            core_end = 0
            optimal_threshold = performance_summary.get('optimal_threshold')
            print(f"Using range consensus '{range_consensus}' to auto-determine core positions")
        else:
            # Last fallback to explicit performance summary positions
            core_start = performance_summary.get('optimal_core_start', optimal_params.get('core_start', 0))
            core_end = performance_summary.get('optimal_core_end', optimal_params.get('core_end', 0))
            optimal_threshold = performance_summary.get('optimal_threshold', optimal_params.get('threshold'))
            range_consensus = None
            print(f"Using fallback core positions: {core_start}-{core_end}")
        
        # Create AlignParameters from unified format data
        parameters = cftf.AlignParameters(
            pwm_file=args.align_model,
            pwm_file_format=optimal_params.get('pwm_file_format', 'Tabular'),
            core_start=core_start,
            core_end=core_end,
            core_gaps=optimal_params.get('core_gaps', []),
            range_consensus=range_consensus,
            gap_limit=performance_summary.get('optimal_gap_limit', optimal_params.get('gap_limit', 0)),
            threshold=optimal_threshold,
            threshold_column=optimal_params.get('threshold_column'),
            palindrome=optimal_params.get('palindrome', False),
            pwm_reverse_complement=optimal_params.get('pwm_reverse_complement', True)
        )
        print(f"Constructed AlignParameters from unified format (core: {parameters.core_start}-{parameters.core_end})")
        
        # Handle k-mer file selection automatically for both PBM and SELEX
        if args.kmer_file:
            # User provided k-mer file - use it (manual override)
            parameters.kmer_file = args.kmer_file
        else:
            # Auto-detect k-mer file from unified optimization output
            best_kmer_file = optimization_info.get('best_kmer_file')
            
            if best_kmer_file:
                import os
                if os.path.exists(best_kmer_file):
                    parameters.kmer_file = best_kmer_file
                    print(f"Auto-selected k-mer file from optimization: {best_kmer_file}")
                else:
                    raise FileNotFoundError(f"K-mer file specified in optimization not found: {best_kmer_file}")
            else:
                # Use k-mer file from input files section
                input_kmer_file = optimization_info.get('input_files', {}).get('K-mer File')
                if input_kmer_file and input_kmer_file != 'N/A':
                    import os
                    if os.path.exists(input_kmer_file):
                        parameters.kmer_file = input_kmer_file
                        print(f"Using k-mer file from input files: {input_kmer_file}")
                    else:
                        raise FileNotFoundError(f"K-mer file from input files not found: {input_kmer_file}")
                else:
                    raise ValueError("--kmer_file (-k) is required when k-mer file cannot be auto-detected from optimization output")
            
        # Apply other parameter overrides
        if args.opt_threshold_type == "Distance":
            # We already applied the optimal threshold above, just confirm
            if parameters.threshold is not None:
                print(f"Applied distance-based optimal threshold: {parameters.threshold}")
            else:
                print("Warning: Distance threshold requested but not available in optimization results")
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


def _save_unified_optimization(output_file: str, opt_obj, data_type: str, 
                              input_files: dict = None, selex_data: dict = None):
    """Save unified optimization results for both PBM and SELEX workflows.
    
    Creates a comprehensive optimization output with clear data type identification
    that works with both PBM and SELEX workflows, and is compatible with align function.
    
    :param output_file: Output file path
    :param opt_obj: Optimization object (single for PBM, best for SELEX)
    :param data_type: 'PBM' or 'SELEX'
    :param input_files: Dictionary of input file paths
    :param selex_data: Dictionary of SELEX-specific data (performance_summary, all_opt_objs, etc.)
    """
    import pandas as pd
    import os
    
    with open(output_file, 'w') as file_obj:
        # Unified header structure
        file_obj.write("#CtrlF-TF Optimization Results\n")
        file_obj.write(f"#Version: {opt_obj.version}\n")
        file_obj.write(f"#Data Type: {data_type}\n")
        file_obj.write("\n")
        
        # Input files section
        file_obj.write("#Input Files:\n")
        if input_files:
            for key, path in input_files.items():
                # Convert to absolute path
                abs_path = os.path.abspath(path) if path else 'N/A'
                file_obj.write(f"#{key}: {abs_path}\n")
        file_obj.write("\n")
        
        # Optimization parameters section
        file_obj.write("#Optimization Parameters:\n")
        file_obj.write(f"#FPR Threshold: {opt_obj.fpr_threshold}\n")
        
        # Extract range consensus and other parameters from init_parameters
        if hasattr(opt_obj, 'init_parameters'):
            params = opt_obj.init_parameters
            if hasattr(params, 'range_consensus') and params.range_consensus:
                file_obj.write(f"#Range Consensus: {params.range_consensus}\n")
            file_obj.write(f"#Palindrome: {getattr(params, 'palindrome', False)}\n")
            file_obj.write(f"#PWM Reverse Complement: {getattr(params, 'pwm_reverse_comp', True)}\n")
        
        # Add workflow-specific parameters from classification
        if hasattr(opt_obj, 'classified_df') and hasattr(opt_obj.classified_df, 'classification_params'):
            params = opt_obj.classified_df.classification_params
            if params.get('data_type') == 'pbm':
                file_obj.write(f"#Classification Method: {params.get('method', 'N/A')}\n")
                file_obj.write(f"#Z Negative: {params.get('z_negative', 'N/A')}\n")
                file_obj.write(f"#Z Positive: {params.get('z_positive', 'N/A')}\n")
                file_obj.write(f"#KDE Positive Ratio: {params.get('kde_positive_ratio', 'N/A')}\n")
            elif params.get('data_type') == 'selex':
                file_obj.write(f"#Buffer Zone: {params.get('buffer_zone', 'N/A')}\n")
                file_obj.write(f"#Sample Size: {params.get('sample_size', 'N/A')}\n")
                file_obj.write(f"#Sample Method: {params.get('sample_method', 'N/A')}\n")
        
        # SELEX-specific sections
        if data_type == 'SELEX' and selex_data:
            # K-mer performance summary for SELEX
            performance_summary = selex_data.get('performance_summary', [])
            if performance_summary:
                file_obj.write("#K-mer Length Performance Summary:\n")
                summary_df = pd.DataFrame(performance_summary)
                summary_df.to_csv(file_obj, sep='\t', index=False)
                file_obj.write("\n")
                
                # Best k-mer length info
                best_entry = max(performance_summary, key=lambda x: x['performance'])
                file_obj.write(f"#Best K-mer Length: {best_entry['k_length']} "
                              f"({best_entry['metric_name']} = {best_entry['performance']:.6f})\n")
                file_obj.write(f"#Best K-mer File: {os.path.abspath(best_entry['kmer_file'])}\n")
                file_obj.write("\n")
        
        # Performance summary for optimal parameters
        file_obj.write("#Performance Summary:\n")
        if hasattr(opt_obj, 'parameter_dataframe') and not opt_obj.parameter_dataframe.empty:
            best_row = opt_obj.parameter_dataframe.loc[opt_obj.parameter_dataframe['pAUROC'].idxmax()]
            file_obj.write(f"#Best pAUROC: {best_row['pAUROC']:.6f}\n")
            file_obj.write(f"#Optimal Core Start: {best_row['Core_Start']}\n")
            file_obj.write(f"#Optimal Core End: {best_row['Core_End']}\n")
            file_obj.write(f"#Optimal Gap Limit: {best_row['Kmer_Gap_Limit']}\n")
            file_obj.write(f"#Optimal Threshold: {best_row['Score_Threshold']}\n")
        file_obj.write("\n")
        
    # Save initial parameters (for align compatibility)
    opt_obj.init_parameters.save_parameters(output_file, mode='a')
    
    # Save standard sections for align compatibility
    with open(output_file, 'a') as file_obj:
        file_obj.write("#Parameter DataFrame:\n")
    if hasattr(opt_obj, 'parameter_dataframe') and not opt_obj.parameter_dataframe.empty:
        opt_obj.parameter_dataframe.to_csv(output_file, sep='\t', index=False, mode='a')
    
    with open(output_file, 'a') as file_obj:
        file_obj.write("#Classified_Dataframe:\n")
    if hasattr(opt_obj, 'classified_df') and not opt_obj.classified_df.empty:
        opt_obj.classified_df.to_csv(output_file, sep='\t', index=False, mode='a')
    
    with open(output_file, 'a') as file_obj:
        file_obj.write("#TPR_FPR_Dataframe:\n")
    # Debug TPR_FPR dictionary status
    has_attr = hasattr(opt_obj, 'tpr_fpr_dictionary')
    has_data = has_attr and opt_obj.tpr_fpr_dictionary is not None
    data_size = len(opt_obj.tpr_fpr_dictionary) if has_data else 0
    
    with open(output_file, 'a') as file_obj:
        file_obj.write(f"# TPR_FPR Debug: has_attr={has_attr}, has_data={has_data}, data_size={data_size}\n")
    
    if has_attr and opt_obj.tpr_fpr_dictionary:
        # Use the same format as original optimize_core.py
        import ctrlf_tf.optimize_utils
        tpr_fpr_dataframe = ctrlf_tf.optimize_utils.meta_tpr_fpr_dataframe(opt_obj.tpr_fpr_dictionary)
        tpr_fpr_dataframe.to_csv(output_file, sep='\t', index=False, mode='a')
    else:
        # TPR_FPR dictionary is missing or empty - this should not happen in a proper optimization
        with open(output_file, 'a') as file_obj:
            file_obj.write("# TPR_FPR data not available - optimization object missing performance data\n")
    
    # SELEX-specific combined data sections
    if data_type == 'SELEX' and selex_data:
        all_opt_objs = selex_data.get('all_opt_objs', {})
        if all_opt_objs:
            with open(output_file, 'a') as file_obj:
                file_obj.write("#Combined Parameter DataFrame (All K-mer Lengths):\n")
            
            # Combine all parameter dataframes with k-length identification
            combined_dfs = []
            for k_length in sorted(all_opt_objs.keys()):
                k_opt_obj = all_opt_objs[k_length]
                if hasattr(k_opt_obj, 'parameter_dataframe') and not k_opt_obj.parameter_dataframe.empty:
                    df_copy = k_opt_obj.parameter_dataframe.copy()
                    df_copy['Kmer_Length'] = k_length
                    df_copy['ID'] = df_copy['ID'].apply(lambda x: f"k{k_length}_{x}")
                    combined_dfs.append(df_copy)
            
            if combined_dfs:
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                combined_df.to_csv(output_file, sep='\t', index=False, mode='a')


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
        
        # Handle KDE vs fixed threshold
        if args.use_kde_threshold and args.kmer_threshold != 0.0:
            print("Warning: Both --use_kde_threshold and --kmer_threshold specified.")
            print("         Using KDE threshold (ignoring fixed --kmer_threshold)")
        
        # Set SELEX-specific defaults: use single threshold since only gap=0 is used
        if not args.use_kde_threshold and args.kmer_threshold == 0.0:  # Default value, user didn't override
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
            
            # Apply KDE threshold if requested
            if args.use_kde_threshold:
                print(f"Applying KDE threshold to k={k_length} k-mers...")
                kde_threshold = ctrlf_tf.selex_utils.determine_kde_threshold(
                    kmer_df['score'].tolist(), args.kde_positive_ratio
                )
                # Filter k-mers based on KDE threshold
                original_count = len(kmer_df)
                kmer_df = kmer_df[kmer_df['score'] >= kde_threshold]
                print(f"KDE filtering: {original_count} -> {len(kmer_df)} k-mers (threshold: {kde_threshold:.6f})")
                
                # Update the threshold for optimization stage
                effective_threshold = kde_threshold
            else:
                # Use fixed threshold
                if args.kmer_threshold != 0.0:
                    original_count = len(kmer_df)
                    kmer_df = kmer_df[kmer_df['score'] >= args.kmer_threshold]
                    print(f"Fixed threshold filtering: {original_count} -> {len(kmer_df)} k-mers (threshold: {args.kmer_threshold})")
                effective_threshold = args.kmer_threshold
            
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
            
            # Store the effective threshold for use in optimization
            if not hasattr(args, '_effective_thresholds'):
                args._effective_thresholds = {}
            args._effective_thresholds[k_length] = effective_threshold
        
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
            effective_threshold = args._effective_thresholds.get(k_length, args.kmer_threshold)
            parameters.threshold = effective_threshold
            parameters.threshold_column = args.kmer_threshold_column
            
            # Use classified sequences directly (already pre-sampled during classify step)
            print(f"Using {len(classified_seqs.dataframe)} pre-sampled sequences for parameter optimization...")
            sampled_classified_df = classified_seqs.dataframe
            
            # Run optimization for this k-length (SELEX ungapped, so simplified gap_thresholds)
            gap_thresholds = {0: effective_threshold}  # Single threshold for gap=0 only
            
            # Create SELEX metadata using parameters from classified file and current args
            classification_params = classified_seqs.classification_params
            selex_metadata = {
                'scoring_method': args.scoring_method,
                'buffer_zone': classification_params.get('buffer_zone', 'N/A'),
                'sample_size': classification_params.get('sample_size', 'N/A'),
                'sample_method': classification_params.get('sample_method', 'N/A'),
                'use_kde_threshold': args.use_kde_threshold,
                'kde_positive_ratio': args.kde_positive_ratio if args.use_kde_threshold else None,
                'effective_threshold': effective_threshold
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
                    # Use the BEST performance metric (not the last one) for k-mer length comparison
                    if 'pAUROC' in opt_obj.parameter_dataframe.columns:
                        performance = opt_obj.parameter_dataframe['pAUROC'].max()
                        metric_name = "pAUROC"
                    elif 'AUROC' in opt_obj.parameter_dataframe.columns:
                        performance = opt_obj.parameter_dataframe['AUROC'].max()
                        metric_name = "AUROC"
                    else:
                        # Fallback to first numeric column (use max for best performance)
                        numeric_cols = opt_obj.parameter_dataframe.select_dtypes(include=[float, int]).columns
                        if len(numeric_cols) > 0:
                            performance = opt_obj.parameter_dataframe[numeric_cols[0]].max()
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
        
        # Clean up non-optimal k-mer files unless --keep_all_kmers is specified
        if best_kmer_file and best_opt_obj is not None:
            print(f"Best k-mer length: {best_k_length} ({metric_name} = {best_performance:.4f})")
            print(f"Optimal k-mer file: {best_kmer_file}")
            
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
                    print(f"Kept optimal k-mer file: {', '.join(kept_files)}")
            
            # Prepare input files and SELEX data for unified save
            input_files = {
                'PWM File': args.align_model,
                'Sequence Data': args.input_file,
                'K-mer File': best_kmer_file,
                'Classified File': args.classify_file
            }
            
            selex_data = {
                'performance_summary': kmer_performance_summary,
                'all_opt_objs': all_opt_objs
            }
            
            # Save using unified optimization format
            _save_unified_optimization(
                output_file=args.output,
                opt_obj=best_opt_obj,
                data_type='SELEX',
                input_files=input_files,
                selex_data=selex_data
            )
            print(f"Saved SELEX optimization results (unified format) to: {args.output}")
        
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
    
    # Prepare input files for PBM unified format
    pbm_input_files = {
        'PWM File': args.align_model,
        'K-mer File': args.kmer_file,
        'Classified File': args.classify_file
    }
    
    opt_obj.save_to_file(args.output, input_files=pbm_input_files)


def _call_program(args):
    # Determine file type and initialize CtrlF appropriately
    with open(args.input_model) as read_obj:
        lines = read_obj.readlines()
        
        # Check if it's an optimization file (unified format)
        if len(lines) > 2 and lines[0].startswith("#CtrlF-TF Optimization Results"):
            raise ValueError(f"Input file appears to be an optimization file. "
                           f"For callsites, use either: "
                           f"1) Alignment file (from 'ctrlf align'), or "
                           f"2) Compiled sites file (from 'ctrlf compile')")
        
        # Check for alignment file format (has #Palindrome header)
        is_alignment_file = False
        for line in lines[:10]:  # Check first 10 lines to be safe
            if line.startswith("#Palindrome"):
                is_alignment_file = True
                break
        
        # Initialize CtrlF object based on file type
        if is_alignment_file:
            ctrlf_object = cftf.CtrlF.from_alignment_file(args.input_model)
        else:
            # Assume it's a compiled sites file
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
