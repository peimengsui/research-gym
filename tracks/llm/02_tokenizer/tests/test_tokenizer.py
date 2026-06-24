import pytest

from implementation import BPETokenizer, UNK_ID, UNK_TOKEN, count_pairs, merge_pair


def test_count_pairs_counts_adjacent_occurrences() -> None:
    counts = count_pairs([1, 2, 1, 2])

    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1


def test_merge_pair_replaces_non_overlapping_occurrences() -> None:
    merged = merge_pair([1, 1, 1, 1, 2], pair=(1, 1), new_token_id=9)

    assert merged == [9, 9, 2]


def test_train_builds_character_vocabulary_and_merges() -> None:
    tokenizer = BPETokenizer.train("banana banana", num_merges=4)

    assert tokenizer.vocab[UNK_ID] == UNK_TOKEN
    assert {"b", "a", "n", " "}.issubset(tokenizer.char_to_id)
    assert 0 < len(tokenizer.merges) <= 4


def test_known_text_round_trips() -> None:
    tokenizer = BPETokenizer.train("banana bandana", num_merges=5)
    text = "banana bandana"

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text


def test_learned_merges_reduce_token_count() -> None:
    tokenizer = BPETokenizer.train("banana banana banana", num_merges=6)

    token_ids = tokenizer.encode("banana")

    assert len(token_ids) < len("banana")


def test_unseen_characters_map_to_unknown_token() -> None:
    tokenizer = BPETokenizer.train("abc abc", num_merges=2)

    token_ids = tokenizer.encode("a?c")

    assert UNK_ID in token_ids
    assert tokenizer.decode(token_ids) == f"a{UNK_TOKEN}c"


def test_unknown_token_id_decodes_safely() -> None:
    tokenizer = BPETokenizer.train("abc", num_merges=0)

    assert tokenizer.decode([999]) == UNK_TOKEN


def test_training_is_deterministic() -> None:
    first = BPETokenizer.train("abab baba", num_merges=4)
    second = BPETokenizer.train("abab baba", num_merges=4)

    assert first.vocab == second.vocab
    assert first.merges == second.merges


def test_negative_merge_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BPETokenizer.train("abc", num_merges=-1)


def test_empty_training_text_is_supported() -> None:
    tokenizer = BPETokenizer.train("", num_merges=3)

    assert tokenizer.vocab == {UNK_ID: UNK_TOKEN}
    assert tokenizer.merges == []
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
