"""Briefly train on two moving-square clips, then generate and evaluate."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyVideoLanguageModel,
        VideoEvalExample,
        evaluate_video_language_model,
        generate_video_text,
        make_next_token_targets,
    )
    from provided import TinyVideoVocabulary, make_moving_square_videos
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        TinyVideoLanguageModel,
        VideoEvalExample,
        evaluate_video_language_model,
        generate_video_text,
        make_next_token_targets,
    )
    from provided import TinyVideoVocabulary, make_moving_square_videos


def main() -> None:
    torch.manual_seed(24)
    vocabulary = TinyVideoVocabulary()
    videos = make_moving_square_videos()
    prompt = torch.tensor(
        vocabulary.encode(["<bos>", "<user>", "what", "moves", "<assistant>"])
    )
    left_answer = torch.tensor(vocabulary.encode(["left", "<eos>"]))
    right_answer = torch.tensor(vocabulary.encode(["right", "<eos>"]))
    text = torch.stack(
        (torch.cat((prompt, left_answer)), torch.cat((prompt, right_answer)))
    )

    model = TinyVideoLanguageModel(
        frames=4,
        image_size=4,
        tubelet_size=2,
        patch_size=2,
        in_channels=1,
        vocab_size=len(vocabulary.tokens),
        max_text_tokens=9,
        embed_dim=8,
        num_heads=1,
        num_video_layers=1,
        num_multimodal_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    targets = make_next_token_targets(text)
    initial_loss = None
    # A few updates demonstrate gradient flow without turning the lesson demo
    # into a training job. The tiny metrics are a pipeline check, not a benchmark.
    for step in range(8):
        optimizer.zero_grad()
        output = model(videos, text, targets)
        assert output.loss is not None
        if step == 0:
            initial_loss = output.loss.item()
        output.loss.backward()
        optimizer.step()

    generated = generate_video_text(
        model,
        videos,
        prompt.unsqueeze(0).expand(2, -1),
        max_new_tokens=2,
        eos_token_id=vocabulary.eos_token_id,
    )
    examples = (
        VideoEvalExample(
            videos[0], prompt, left_answer, (left_answer, right_answer), 0
        ),
        VideoEvalExample(
            videos[1], prompt, right_answer, (left_answer, right_answer), 1
        ),
    )
    summary = evaluate_video_language_model(
        model, examples, max_new_tokens=2, eos_token_id=vocabulary.eos_token_id
    )

    print(f"video shape:                  {tuple(videos.shape)}")
    print(f"video prefix token count:     {model.video_token_count}")
    print(f"initial training loss:        {initial_loss:.4f}")
    print(f"final training loss:          {output.loss.item():.4f}")
    print(f"generated text:               {vocabulary.decode(generated)}")
    print(f"generation exact match:       {summary.generation_exact_match:.3f}")
    print(f"candidate ranking accuracy:   {summary.candidate_accuracy:.3f}")
    print(
        "Generation recomputes the tiny sequence; KV-cache reuse is a separate lesson."
    )


if __name__ == "__main__":
    main()
