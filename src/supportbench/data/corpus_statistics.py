from dataclasses import dataclass
from math import ceil, floor
from statistics import fmean, median, pstdev

from supportbench.retrieval.inverted_index import InvertedIndex


@dataclass(frozen=True, slots=True)
class DocumentLengthStats:
    minimum: int
    median: float
    mean: float
    p90: float
    maximum: int
    standard_deviation: float
    coefficient_of_variation: float


@dataclass(frozen=True, slots=True)
class PostingFrequencyStats:
    posting_count: int
    share_tf_1: float  # share of postings with term frequency = 1
    share_tf_2: float
    share_tf_3_or_more: float
    mean: float
    maximum: float


@dataclass(frozen=True, slots=True)
class FullCorpusStats:
    document_lengths: DocumentLengthStats
    posting_frequencies: PostingFrequencyStats


def compute_full_corpus_statistics(index: InvertedIndex) -> FullCorpusStats:
    document_lengths = [index.document_length(doc_id) for doc_id in index.document_ids]

    if not document_lengths:
        raise ValueError("cannot compute statistics for an empty index")

    posting_frequencies = [
        frequency for term in index.terms for frequency in index.postings_for(term).values()
    ]

    if not posting_frequencies:
        raise ValueError("index contains no postings")

    mean_document_length = fmean(document_lengths)
    std_document_length = pstdev(document_lengths)

    length_statistics = DocumentLengthStats(
        minimum=min(document_lengths),
        median=float(median(document_lengths)),
        mean=mean_document_length,
        p90=_percentile(document_lengths, p=0.9),
        maximum=max(document_lengths),
        standard_deviation=std_document_length,
        coefficient_of_variation=(std_document_length / mean_document_length),
    )

    posting_count = len(posting_frequencies)

    frequency_statistics = PostingFrequencyStats(
        posting_count=posting_count,
        share_tf_1=(sum(freq == 1 for freq in posting_frequencies) / posting_count),
        share_tf_2=(sum(freq == 2 for freq in posting_frequencies) / posting_count),
        share_tf_3_or_more=(sum(freq >= 3 for freq in posting_frequencies) / posting_count),
        mean=fmean(posting_frequencies),
        maximum=max(posting_frequencies),
    )

    return FullCorpusStats(
        document_lengths=length_statistics,
        posting_frequencies=frequency_statistics,
    )


def _percentile(values: list[int], *, p: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")

    if not 0.0 <= p <= 1.0:
        raise ValueError("percentile 'p' must be between 0 and 1")

    ordered_values = sorted(values)

    if len(ordered_values) == 1:
        return float(ordered_values[0])

    position = p * (len(ordered_values) - 1)
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return float(ordered_values[lower_index])

    integer_part = int(position)
    fractional_part = position - integer_part
    return ordered_values[integer_part] + fractional_part * (
        ordered_values[integer_part + 1] - ordered_values[integer_part]
    )
