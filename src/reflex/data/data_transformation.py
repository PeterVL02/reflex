"""
Sklearn-compatible text feature transformers for prompt-injection detection.

Every factory returns an unfitted transformer: fit it on training texts only (e.g. as the
first step of a Pipeline) and transform any texts, so no vocabulary or scaling leaks
from validation/test data.
"""

import re
import string
import unicodedata
from enum import StrEnum, auto
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Soft hyphen, zero-width space/joiners, LRM/RLM, word joiner, invisible operators, BOM.
ZERO_WIDTH_CHARS: dict[int, None] = dict.fromkeys(
    [0xAD, 0xFEFF, *range(0x200B, 0x2010), *range(0x2060, 0x2065)]
)
PUNCTUATION: frozenset[str] = frozenset(string.punctuation)

# ponytail: hand-written regex heuristics; extend from false negatives found in evaluation.
ROLE_TAG_PATTERN: re.Pattern[str] = re.compile(
    r"</?\s*(?:system|user|assistant|instructions?|prompt|im_start|im_end)\s*>"
    r"|\[/?(?:inst|sys)\]|<<\s*/?sys\s*>>|<\|[a-z_]+\|>"
    r"|^\s*#*\s*(?:system|assistant|user|developer)\s*:"
    r"|^\s*(?:-{3,}|={3,}|#{3,})",
    re.IGNORECASE | re.MULTILINE,
)
OVERRIDE_PHRASE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|skip)\b.{0,40}?"
    r"\b(?:instructions?|rules|prompts?|directions|guidelines|guardrails|restrictions)\b"
    r"|\byou are now\b|\bdeveloper mode\b|\bjailbr[eo]ak|\bdo anything now\b|\bdan\b"
    r"|\bpretend (?:you are|to be)\b|\bact as\b|\bno (?:restrictions|limits|filters)\b"
    r"|\b(?:system|initial|hidden) prompt\b",
    re.IGNORECASE,
)
BASE64_RUN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


class TokenLevel(StrEnum):
    WORD = auto()
    CHAR = auto()


class VectorizerKwargs(TypedDict):
    analyzer: Literal["word", "char_wb"]
    ngram_range: tuple[int, int]


def normalize_text(text: str) -> str:
    """Apply NFKC normalisation, strip zero-width/invisible characters and casefold."""
    return unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH_CHARS).casefold()


def _build_vectorizer_kwargs(
    level: TokenLevel, ngram_range: tuple[int, int] | None
) -> VectorizerKwargs:
    """Return the vectorizer arguments shared by the bag-of-words and TF-IDF factories."""
    match level:
        case TokenLevel.WORD:
            return {"analyzer": "word", "ngram_range": ngram_range or (1, 2)}
        case TokenLevel.CHAR:
            return {"analyzer": "char_wb", "ngram_range": ngram_range or (2, 5)}


def make_bag_of_words(
    level: TokenLevel = TokenLevel.WORD,
    ngram_range: tuple[int, int] | None = None,
    min_df: int = 2,
) -> CountVectorizer:
    """
    Create an unfitted bag-of-words (count) vectorizer over normalised text.

    Args:
        level (TokenLevel): Word n-grams or character n-grams (within word boundaries).
        ngram_range (tuple[int, int] | None): N-gram sizes; defaults to (1, 2) for words, (2, 5) for chars.
        min_df (int): Drop n-grams seen in fewer training documents than this.

    Returns:
        CountVectorizer: Transformer producing a sparse count matrix.
    """
    return CountVectorizer(
        preprocessor=normalize_text,
        min_df=min_df,
        **_build_vectorizer_kwargs(level, ngram_range),
    )


def make_tfidf(
    level: TokenLevel = TokenLevel.WORD,
    ngram_range: tuple[int, int] | None = None,
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Create an unfitted TF-IDF vectorizer (sublinear tf) over normalised text.

    Args:
        level (TokenLevel): Word n-grams or character n-grams (within word boundaries).
        ngram_range (tuple[int, int] | None): N-gram sizes; defaults to (1, 2) for words, (2, 5) for chars.
        min_df (int): Drop n-grams seen in fewer training documents than this.

    Returns:
        TfidfVectorizer: Transformer producing a sparse, L2-normalised TF-IDF matrix.
    """
    return TfidfVectorizer(
        preprocessor=normalize_text,
        min_df=min_df,
        sublinear_tf=True,
        **_build_vectorizer_kwargs(level, ngram_range),
    )


type VectorizerName = Literal["bow_word", "bow_char", "tfidf_word", "tfidf_char"]
VECTORIZER_NAMES: tuple[VectorizerName, ...] = (
    "bow_word",
    "bow_char",
    "tfidf_word",
    "tfidf_char",
)


def make_vectorizer(name: VectorizerName) -> CountVectorizer:
    """Create an unfitted vectorizer by name: bag-of-words or TF-IDF, word or char n-grams."""
    kind, level = name.split("_")
    factory = make_tfidf if kind == "tfidf" else make_bag_of_words
    return factory(TokenLevel(level))


def extract_handcrafted_features(
    texts: list[str] | pd.Series[str],
) -> npt.NDArray[np.float64]:
    """
    Compute stateless injection-signal features per text.

    Columns: log length, uppercase ratio, punctuation ratio, non-ASCII ratio, digit ratio,
    has zero-width chars, has base64-like run, role-tag/delimiter count, override-phrase count.
    Shape/obfuscation features use the raw text; pattern counts use normalised text.

    Args:
        texts (list[str] | pd.Series[str]): Raw prompts.

    Returns:
        npt.NDArray[np.float64]: Array of shape (n_texts, 9).
    """
    rows: list[list[float]] = []
    for text in texts:
        n = max(len(text), 1)
        normalized = normalize_text(text)
        rows.append(
            [
                np.log1p(len(text)),
                sum(c.isupper() for c in text) / n,
                sum(c in PUNCTUATION for c in text) / n,
                sum(not c.isascii() for c in text) / n,
                sum(c.isdigit() for c in text) / n,
                float(len(text.translate(ZERO_WIDTH_CHARS)) != len(text)),
                float(BASE64_RUN_PATTERN.search(text) is not None),
                len(ROLE_TAG_PATTERN.findall(normalized)),
                len(OVERRIDE_PHRASE_PATTERN.findall(normalized)),
            ]
        )
    return np.array(rows, dtype=np.float64)


def make_handcrafted_features() -> Pipeline:
    """Create an unfitted transformer of standardised handcrafted features (dense, 9 columns)."""
    return make_pipeline(
        FunctionTransformer(extract_handcrafted_features), StandardScaler()
    )


def make_embedder(batch_size: int = 32) -> FunctionTransformer:
    """
    Create a stateless transformer embedding texts with Qwen3-Embedding (dense, 1024 columns).

    The embedding model is imported lazily so the other transformers stay lightweight.

    Args:
        batch_size (int): Prompts per forward pass; lower it if the GPU runs out of memory.

    Returns:
        FunctionTransformer: Transformer calling `embed_prompts`.
    """
    from reflex.model.embeddings import embed_prompts

    # ponytail: re-embeds on every fit/transform; precompute once and slice per fold if k-fold is slow.
    return FunctionTransformer(embed_prompts, kw_args={"batch_size": batch_size})
