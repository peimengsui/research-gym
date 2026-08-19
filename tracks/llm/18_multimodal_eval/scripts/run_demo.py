"""Train briefly, then inspect generation and candidate-ranking metrics."""

import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import evaluate_multimodal_model, make_toy_evaluation_examples
    from provided import TinyCachedVLM, TinyVocabulary
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import evaluate_multimodal_model, make_toy_evaluation_examples
    from provided import TinyCachedVLM, TinyVocabulary


def train_on_toy_examples(
    model: TinyCachedVLM,
    examples: list,
    steps: int = 100,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        losses = []
        for example in examples:
            eos = example.candidate_answer_token_ids[0][-1:]
            answer = torch.cat((example.reference_answer_token_ids, eos))
            text = torch.cat((example.prompt_token_ids, answer))
            logits = model.forward_full(example.image.unsqueeze(0), text.unsqueeze(0))[
                0
            ]
            start = example.prompt_token_ids.numel() - 1
            answer_logits = logits[start : start + answer.numel()]
            losses.append(F.cross_entropy(answer_logits, answer))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def print_summary(label: str, summary, vocabulary: TinyVocabulary) -> None:
    print(label)
    print(f"  generation exact match: {summary.generation_exact_match:.2f}")
    print(f"  candidate accuracy:     {summary.candidate_accuracy:.2f}")
    for index, record in enumerate(summary.records):
        generated = vocabulary.decode(list(record.generated_answer))
        reference = vocabulary.decode(list(record.reference_answer))
        rounded_scores = [round(score, 3) for score in record.candidate_scores]
        print(
            f"  row {index}: generated='{generated}' reference='{reference}' "
            f"scores={rounded_scores}"
        )


def main() -> None:
    torch.manual_seed(18)
    vocabulary = TinyVocabulary()
    examples = make_toy_evaluation_examples(vocabulary)
    model = TinyCachedVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=len(vocabulary.tokens),
        max_text_tokens=9,
        embed_dim=12,
        num_heads=3,
        num_layers=1,
    )

    before = evaluate_multimodal_model(
        model, examples, max_new_tokens=3, eos_token_id=vocabulary.eos_token_id
    )
    train_on_toy_examples(model, examples)
    after = evaluate_multimodal_model(
        model, examples, max_new_tokens=3, eos_token_id=vocabulary.eos_token_id
    )

    print_summary("Before tiny training:", before, vocabulary)
    print_summary("After tiny training:", after, vocabulary)
    print("Inspect both aggregate metrics and per-example records when debugging.")


if __name__ == "__main__":
    main()
