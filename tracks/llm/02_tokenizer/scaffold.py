"""Learner scaffold for the Tokenizer Fundamentals lesson."""

from collections import Counter
from collections.abc import Iterable

Pair = tuple[int, int]
Merge = tuple[Pair, int]

UNK_TOKEN = "<unk>"
UNK_ID = 0


def count_pairs(token_ids: list[int]) -> Counter[Pair]:
    """Count adjacent token pairs in one token sequence."""

    # TODO: Return counts for every adjacent pair.
    # Example: [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}
    raise NotImplementedError("TODO: count adjacent token pairs")


def merge_pair(token_ids: list[int], pair: Pair, new_token_id: int) -> list[int]:
    """Replace non-overlapping occurrences of ``pair`` from left to right."""

    # TODO: Scan once from left to right. When pair appears, append
    # new_token_id and advance by two positions. Otherwise copy one token.
    # Example: [1, 1, 1], pair=(1, 1) -> [new_token_id, 1]
    raise NotImplementedError("TODO: merge one adjacent token pair")


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
        """Return the number of tokens in the learned vocabulary."""

        return len(self.vocab)

    @classmethod
    def train(cls, text: str, num_merges: int) -> "BPETokenizer":
        """Learn up to ``num_merges`` frequent adjacent-token merges."""

        # TODO:
        # 1. Build a deterministic character vocabulary. Reserve ID 0 for
        #    UNK_TOKEN and assign sorted characters IDs starting at 1.
        # 2. Convert the training text to character IDs.
        # 3. Repeatedly count adjacent pairs and select the most frequent pair.
        #    Break ties lexicographically so training is reproducible.
        # 4. Stop if there are no pairs or the best pair occurs fewer than
        #    two times.
        # 5. Add the concatenated pair string to vocab, merge the training
        #    sequence, and remember (pair, new_token_id).
        raise NotImplementedError("TODO: learn BPE merges")

    def encode(self, text: str) -> list[int]:
        """Encode text, mapping unseen characters to ``unk_id``."""

        # TODO: Start with character IDs, then apply learned merges in their
        # training order using merge_pair.
        raise NotImplementedError("TODO: encode text")

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode IDs, rendering unknown IDs as ``<unk>``."""

        # TODO: Look up each token string and concatenate them.
        raise NotImplementedError("TODO: decode token IDs")
