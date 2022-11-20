"""Unit tests for the str_utils module."""

import ctrlf_tf.str_utils
import pytest



def test_reverse_complement():
    """Tests the reverse_complement function."""
    query_list = ["TTCCGG", "", "CACGTG", "CA.TTC", "TTccGG", "TTYANN"]
    expected_output = ["CCGGAA", "", "CACGTG", "GAA.TG", "CCGGAA", "NNTYAA"]
    for query, expected in zip(query_list, expected_output):
        assert ctrlf_tf.str_utils.reverse_complement(query) == expected


def test_compatible_descripton():
    """ Tests the compatible_description function."""
    comparison_list_a = ["TTCCGGAA", "...CTAGCTAG..", "ACGTCA", ".CACGTG.."]
    comparison_list_b = ["TTCCGGAA", "..ACTAGCTA...", "AG....", "..ACG...."]
    result_list = [True, True, False]
    for a, b, r in zip(comparison_list_a, comparison_list_b, result_list):
        assert ctrlf_tf.str_utils.compatible_description(a, b) == r
    with pytest.raises(ValueError):
        ctrlf_tf.str_utils.compatible_description("AAAAA", "A")


def test_hamming_distance():
    """Tests the hamming_distance function."""
    comparison_list_a = ["TTCCGGAA", "...CTAGCTAG..", "ACGTCA", ".CACGTG.."]
    comparison_list_b = ["TTCCGGAA", "..ACTAGCTA...", "AG....", "..ACG...."]
    result_list = [0, 2, 5, 3]
    for a, b, r in zip(comparison_list_a, comparison_list_b, result_list):
        assert ctrlf_tf.str_utils.hamming_distance(a, b) == r
    with pytest.raises(ValueError):
        ctrlf_tf.str_utils.hamming_distance("AAAAA", "A")


def test_is_within_gap_limit():
    test_iterable = ["ACGTTCC", "CAC.TGAT", ".ACGTA.TAC"]
    result_0 = (True, False, False)
    result_1 = (True, True, False)
    assert result_0 == tuple(ctrlf_tf.str_utils.is_within_gap_limit(test_iterable, 0))
    assert result_1 == tuple(ctrlf_tf.str_utils.is_within_gap_limit(test_iterable, 1))
    with pytest.raises(ValueError):
        ctrlf_tf.str_utils.is_within_gap_limit(test_iterable, -4)


def test_k_from_kmers():
    kmers = [["TTC.GGC", "AGTCTA", "C.A.C.G.T.G"], ["GATAA"]]
    results = [6, 5]
    for k, r in zip(kmers, results):
        assert ctrlf_tf.str_utils.k_from_kmers(k) == r
    with pytest.raises(ValueError):
        bad_input = ["TTCCGGA", "TT.CGGA"]
        ctrlf_tf.str_utils.k_from_kmers(bad_input)
