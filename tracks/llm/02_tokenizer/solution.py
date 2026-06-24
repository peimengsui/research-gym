"""Reference solution for the Tokenizer Fundamentals lesson."""

from collections import Counter
from collections.abc import Iterable

Pair = tuple[int, int]
Merge = tuple[Pair, int]

UNK_TOKEN = "<unk>"
UNK_ID = 0


def count_pairs(token_ids: list[int]) -> Counter[Pair]:
    return Counter(zip(token_ids, token_ids[1:]))


def merge_pair(token_ids: list[int], pair: Pair, new_token_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(token_ids):
        if (
            index + 1 < len(token_ids)
            and (
                token_ids[index],
                token_ids[index + 1],
            )
            == pair
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


class BPETokenizer:
    """A tiny character-initialized byte-pair-style tokenizer."""

    def __init__(self, vocab: dict[int, str], merges: list[Merge]):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.unk_token = UNK_TOKEN
        self.unk_id = UNK_ID
        self.char_to_id = {
            token: token_id for token_id, token in self.vocab.items() if len(token) == 1
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, num_merges: int) -> "BPETokenizer":
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")

        characters = sorted(set(text))
        vocab = {UNK_ID: UNK_TOKEN}
        vocab.update(
            {token_id: character for token_id, character in enumerate(characters, 1)}
        )
        char_to_id = {token: token_id for token_id, token in vocab.items()}
        token_ids = [char_to_id[character] for character in text]
        merges: list[Merge] = []

        for _ in range(num_merges):
            pair_counts = count_pairs(token_ids)
            if not pair_counts:
                break

            best_frequency = max(pair_counts.values())
            if best_frequency < 2:
                break
            best_pair = min(
                pair
                for pair, frequency in pair_counts.items()
                if frequency == best_frequency
            )

            new_token_id = len(vocab)
            left, right = best_pair
            vocab[new_token_id] = vocab[left] + vocab[right]
            token_ids = merge_pair(token_ids, best_pair, new_token_id)
            merges.append((best_pair, new_token_id))

        return cls(vocab=vocab, merges=merges)

    def encode(self, text: str) -> list[int]:
        token_ids = [self.char_to_id.get(character, self.unk_id) for character in text]
        for pair, new_token_id in self.merges:
            token_ids = merge_pair(token_ids, pair, new_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(
            self.vocab.get(token_id, self.unk_token) for token_id in token_ids
        )
