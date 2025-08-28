#!/bin/bash
#SBATCH --job-name=selex_kde_test
#SBATCH --output=selex_kde_test_%j.out
#SBATCH --error=selex_kde_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Load necessary modules
module load python/3.10

# Change to working directory
cd /usr/project/xtmp/sl548/CtrlF-TF

# Run SELEX optimization with KDE threshold
echo "Starting SELEX optimization with KDE threshold..."
echo "Parameters:"
echo "  Core: TTCC"  
echo "  K-mer range: 7-10"
echo "  Method: KDE threshold"
echo "  Data: selex_run/GABPA_SELEX_example_data.csv"
echo "  PWM: selex_run/GABPA.H12CORE.0.PSM.A.pfm.tab"

ctrlf optimize --data selex \
    -a selex_run/GABPA.H12CORE.0.PSM.A.pfm.tab \
    -c selex_run/classify/classified_selex_100k.txt \
    -i selex_run/GABPA_SELEX_example_data.csv \
    -rc TTCC \
    --kmer_length_range 7-10 \
    --scoring_method average \
    --buffer_zone 0.05 \
    --sample_size 100000 \
    -o selex_run/optimize/selex_kde_test.txt

echo "SELEX optimization completed!"
echo "Output saved to: selex_run/optimize/selex_kde_test.txt"

# List k-mer files generated
echo ""
echo "Generated k-mer files:"
ls -la selex_run/optimize/selex_kde_test_kmers_k*.txt 2>/dev/null || echo "No k-mer files found"