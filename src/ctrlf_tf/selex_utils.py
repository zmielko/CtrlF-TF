"""SELEX data processing utilities."""

import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import ctrlf_tf.str_utils


def generate_kmers_from_sequences(sequences: List[str], 
                                 scores: List[float], 
                                 k_length: int, 
                                 method: str = "median") -> pd.DataFrame:
    """Generate k-mers from sequences with combined scores.
    
    Extracts all k-mers using sliding window, combines with reverse complement
    scores, and returns PBM-compatible format.
    
    :param sequences: List of DNA sequences
    :param scores: List of sequence scores  
    :param k_length: Length of k-mers to extract
    :param method: Scoring method ("median" or "average")
    :returns: DataFrame with columns [kmer, rev_comp, combined_score]
    """
    kmer_scores = defaultdict(list)
    
    for seq, score in zip(sequences, scores):
        seq = seq.upper()
        # Extract all k-mers from sequence
        for i in range(len(seq) - k_length + 1):
            kmer = seq[i:i + k_length]
            if 'N' not in kmer:  # Skip ambiguous nucleotides
                kmer_scores[kmer].append(score)
                # Add reverse complement
                rev_comp = ctrlf_tf.str_utils.reverse_complement(kmer)
                kmer_scores[rev_comp].append(score)
    
    # Combine scores for each k-mer
    kmer_data = []
    processed_kmers = set()
    
    for kmer in kmer_scores:
        if kmer in processed_kmers:
            continue
            
        rev_comp = ctrlf_tf.str_utils.reverse_complement(kmer)
        
        # Combine k-mer and reverse complement scores
        all_scores = kmer_scores[kmer] + kmer_scores.get(rev_comp, [])
        
        if method == "median":
            combined_score = smart_median(all_scores)
        else:  # average
            combined_score = np.mean(all_scores)
        
        kmer_data.append([kmer, rev_comp, combined_score])
        processed_kmers.add(kmer)
        processed_kmers.add(rev_comp)
    
    return pd.DataFrame(kmer_data, columns=['kmer', 'rev_comp', 'score'])


def stratified_sample(sequences: List[str], 
                     scores: List[float], 
                     groups: List[str], 
                     sample_size: int) -> Tuple[List[str], List[float], List[str]]:
    """Balanced sampling for optimization with fixed negative minimum.
    
    Logic:
    - If positives < sample_size/2: use all positives + sample_size/2 negatives
    - If positives >= sample_size/2: sample sample_size/2 positives + sample_size/2 negatives
    - Ambiguous sequences (.) are excluded from optimization sampling
    
    :param sequences: List of sequences
    :param scores: List of scores
    :param groups: List of group classifications (+/-/.)
    :param sample_size: Target sample size (total will be positives + sample_size/2)
    :returns: Tuple of (sampled_sequences, sampled_scores, sampled_groups)
    """
    # Group indices by classification
    group_indices = defaultdict(list)
    for i, group in enumerate(groups):
        group_indices[group].append(i)
    
    sampled_indices = []
    
    # Get positive and negative indices
    positive_indices = group_indices.get('+', [])
    negative_indices = group_indices.get('-', [])
    
    # Calculate target sizes
    max_positives = sample_size // 2
    target_negatives = sample_size // 2
    
    # Sample positives
    if len(positive_indices) <= max_positives:
        # Use all positives if less than half of sample_size
        sampled_indices.extend(positive_indices)
        actual_positives = len(positive_indices)
    else:
        # Sample half of sample_size positives
        sampled_indices.extend(random.sample(positive_indices, max_positives))
        actual_positives = max_positives
    
    # Sample negatives (always target_negatives = sample_size/2)
    if len(negative_indices) <= target_negatives:
        # Use all negatives if we don't have enough
        sampled_indices.extend(negative_indices)
    else:
        # Sample exactly target_negatives negatives
        sampled_indices.extend(random.sample(negative_indices, target_negatives))
    
    # Extract sampled data
    sampled_sequences = [sequences[i] for i in sampled_indices]
    sampled_scores = [scores[i] for i in sampled_indices]
    sampled_groups = [groups[i] for i in sampled_indices]
    
    actual_sample_size = len(sampled_sequences)
    pos_sampled = sum(1 for g in sampled_groups if g == '+')
    neg_sampled = sum(1 for g in sampled_groups if g == '-')
    print(f"      Sample result: {actual_sample_size} total ({pos_sampled} positive, {neg_sampled} negative)")
    return sampled_sequences, sampled_scores, sampled_groups


def _legacy_stratified_sample(sequences: List[str], 
                             scores: List[float], 
                             groups: List[str], 
                             sample_size: int) -> Tuple[List[str], List[float], List[str]]:
    """Legacy stratified sampling maintaining group proportions.
    
    :param sequences: List of sequences
    :param scores: List of scores
    :param groups: List of group classifications (+/-/.)
    :param sample_size: Target sample size
    :returns: Tuple of (sampled_sequences, sampled_scores, sampled_groups)
    """
    # Group indices by classification
    group_indices = defaultdict(list)
    for i, group in enumerate(groups):
        group_indices[group].append(i)
    
    # Calculate proportional sample sizes
    total = len(sequences)
    sampled_indices = []
    
    for group, indices in group_indices.items():
        group_proportion = len(indices) / total
        group_sample_size = max(1, int(sample_size * group_proportion))
        
        if len(indices) <= group_sample_size:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(random.sample(indices, group_sample_size))
    
    # Extract sampled data
    sampled_sequences = [sequences[i] for i in sampled_indices]
    sampled_scores = [scores[i] for i in sampled_indices]
    sampled_groups = [groups[i] for i in sampled_indices]
    
    return sampled_sequences, sampled_scores, sampled_groups


def get_optimization_sample(sequences: List[str], 
                          scores: List[float], 
                          groups: List[str],
                          sample_size: int = 100000, 
                          sample_method: str = "balanced") -> Tuple[List[str], List[float], List[str]]:
    """Get sample for optimization with balanced sampling strategy.
    
    Default balanced sampling:
    - If positives < sample_size/2: use all positives + sample_size/2 negatives
    - If positives >= sample_size/2: sample sample_size/2 positives + sample_size/2 negatives
    
    :param sequences: All classified sequences
    :param scores: All scores
    :param groups: All group classifications  
    :param sample_size: Target sample size for balanced approach (default: 100000)
    :param sample_method: Sampling method ("balanced", "stratified", or "random")
    :returns: Tuple of (sample_sequences, sample_scores, sample_groups)
    """
    # Count positive and negative sequences
    positive_count = groups.count('+')
    negative_count = groups.count('-')
    total_sequences = len(sequences)
    print(f"    Available for sampling: {positive_count} positive, {negative_count} negative, {total_sequences - positive_count - negative_count} ambiguous")
    
    # If very small dataset, use all sequences
    if positive_count + negative_count <= sample_size // 4:
        print(f"    Using all {total_sequences} sequences (small dataset)")
        return sequences, scores, groups
    
    # Sample subset for optimization
    if sample_method == "balanced":
        return stratified_sample(sequences, scores, groups, sample_size)
    elif sample_method == "stratified":
        # Legacy proportional sampling (kept for compatibility)
        return _legacy_stratified_sample(sequences, scores, groups, sample_size)
    else:
        # Simple random sampling
        indices = random.sample(range(total_sequences), min(sample_size, total_sequences))
        sampled_sequences = [sequences[i] for i in indices]
        sampled_scores = [scores[i] for i in indices]
        sampled_groups = [groups[i] for i in indices]
        return sampled_sequences, sampled_scores, sampled_groups




def smart_median(scores: List[float]) -> float:
    """Optimized median computation with smart handling for different sizes.
    
    :param scores: List of score values
    :returns: Median value
    """
    n = len(scores)
    if n == 1:
        return scores[0]
    elif n == 2:
        return (scores[0] + scores[1]) / 2.0
    elif n <= 5:
        # Use insertion sort for small arrays (faster than numpy overhead)
        sorted_scores = sorted(scores)
        if n % 2 == 1:
            return sorted_scores[n // 2]
        else:
            mid = n // 2
            return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    else:
        # Use numpy's optimized median for larger arrays
        return np.median(scores)



