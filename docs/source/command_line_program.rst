.. _CLI:

Command Line Interface
======================

CtrlF-TF comes with a command line program, `ctrlf`, which comes with
three subprograms:

1) `ctrlf compile`: Aligns k-mers and compiles them into searchable sites.
2) `ctrlf sitecall`: Calls binding sites from FASTA files using the output of `ctrlf compile`.
3) `ctrlf classify`: Classifies a table of values and sequences for use in optimization. Used only when the `-optimize` argument is given in `ctrlf compile`.

Using `ctrlf -v` will return the version number.

A typical workflow would be to generate searchable sites once with `ctrlf compile` and then use `ctrlf sitecall` to call sites as needed.


Calling TF binding sites: ctrlf sitecall
----------------------------------------

The `ctrlf sitecall` program uses the following arguments:

1) --consensus_sites: Output file from `ctrlf compile`
2) --fasta_file: File of DNA sequences to call sites
3) --output: Output file location, stdout by default. Output is in BED format.
4) --genomic_coordiantes: Parses the fasta input for genomic coordinates to use in the BED output.

Below is an example of the syntax:

.. code:: bash

    ctrlf sitecall --consensus_sites ctrlf_compile_output.txt --fasta_file genomic_sequences.fasta --genomic_coordinates --output binding_sites.bed


Creating searchable sites: ctrlf compile
----------------------------------------

Conceptually, you will need to:

1) Have a PWM model and know a minimal subset of it that constitutes a core region,
2) Know if the TF binding is palindromic

For optimization of the compilation parameters, you will also need to have a table with values, sequences, and a classification of +/./- which can be generated with the `ctrlf classify` program. More details on how CtrlF conceptually works can be found here REF. The syntax for `ctrlf classify` follows the syntax for AlignParameters in the python library which can be found here REF.

To use ctrlf compile, you will need the following **data files**:

1) A PWM model to align k-mers to
2) A tabular files of k-mers

The PWM file is used to align the k-mers and its width provides the definition of a
core binding site. The PWM file format is tabular by default but can be specified as a MEME format with the `-m` argument. The default core definition is the entire PWM as given, but several optional
parameters are available to indicate a subset of the PWM is the core region.

Manual definiton:

1) --range: 1-based inclusive range of the binding site core.
2) --core_gap: Positions within the core range that are not part of the core definition

Automatic definition (recommended):

1) --range_consensus: A consensus site of the core with '.' characters representing core gaps.

For example, given a --range-consensus of TTCC.GGAA will align the sequence to the PWM, use that
subset at the core definition, and have the 5th position as a core gap.

Helper arguments also exist for working with the k-mer data:

1) --threshold: A threshold value for the k-mers.
2) --threshold_column: A column in the k-mer file to use values from.
3) --gap_limit: The number of gaps allowed in k-mers.

Output is to stdout by default, but can be specified to save to a file location with the `-output` argument.

.. code:: bash

   ctrlf compile -a pwm_file.txt -k kmer_file.txt --range-consensus TTCC.GGAA --palindrome -o searchable_sites.txt

One can also run an optimization step, which uses the output from `ctrlf classify` to optimize for the core definition, gap limit, and threshold. To do this, the `--optimize` argument needs to be given and if so the following argument is required:

1) --classify_file: output file form `ctrlf classify`

Some optional parameters only relating to optimization include:

1) --fpr_threshold: A fpr threshold to benchmark performance on (default = 0.01)
2) --gap_thresholds: k-mer score thresholds for optimizing gaps
3) --output-report: Output file location for a report on the optimization (default is no report)

These arguments correpsond to the ones in the Optimize class REF. Here is an example of the syntax from before but with optimization:

.. code:: bash

    ctrlf compile -a pwm_file.txt -k kmer_file.txt --range-consensus TTCC.GGAA --palindrome -o searchable_sites.txt --optimize --classify_file ctrlf_classify_output.txt


Classification of sequences for optimization: ctrlf classify
------------------------------------------------------------

The optimization step involves benchmarking a classification task of sequences
with binding sites and those without. The `ctrlf classify` program offers a
convienence feature of taking tables with TF binding measurements and sequences
and returning a table in the format used by `ctrlf compile` (see REF).

The only required argument is the input file:

1) --input_file: Input tabular file with measurements in the fist column and sequences in the second

Optional arguments include:

1) --output: Output file location, stdout by default.
2) --method: Classification method, kde_z4 or z-score, kde_z4 by default.
3) --z_scores: Z-scores to use if classifying by z-score.
4) --sequence-range: Sequence subset positions to use (default is the whole sequence).
5) --ln_transform: Performs a natural log transformation of values prior to classification.

Example syntax would be:

.. code:: bash

    ctrlf classify --input_file input_file --output classified_values.txt
