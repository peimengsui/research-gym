"""Briefly train a tiny audio LM, then run generation and evaluation."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        AudioEvalExample,
        TinyAudioLanguageModel,
        evaluate_audio_language_model,
        generate_audio_text,
        make_next_token_targets,
    )
    from provided import TinyAudioVocabulary, make_toy_tone_batch
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        AudioEvalExample,
        TinyAudioLanguageModel,
        evaluate_audio_language_model,
        generate_audio_text,
        make_next_token_targets,
    )
    from provided import TinyAudioVocabulary, make_toy_tone_batch


def main() -> None:
    torch.manual_seed(27)
    vocabulary = TinyAudioVocabulary()
    waveforms, lengths = make_toy_tone_batch()
    prompt = torch.tensor(
        vocabulary.encode(["<bos>", "<user>", "what", "tone", "<assistant>"])
    )
    low_answer = torch.tensor(vocabulary.encode(["low", "<eos>"]))
    high_answer = torch.tensor(vocabulary.encode(["high", "<eos>"]))
    text = torch.stack(
        (torch.cat((prompt, low_answer)), torch.cat((prompt, high_answer)))
    )
    model = TinyAudioLanguageModel(18, 6, 4, 2, 2, 10, 9, 8, 1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    targets = make_next_token_targets(text)
    initial_loss = None
    for step in range(8):
        optimizer.zero_grad()
        output = model(waveforms, lengths, text, targets)
        assert output.loss is not None
        if step == 0:
            initial_loss = output.loss.item()
        output.loss.backward()
        optimizer.step()

    generated = generate_audio_text(
        model,
        waveforms,
        lengths,
        prompt.unsqueeze(0).expand(2, -1),
        2,
        vocabulary.eos_token_id,
    )
    examples = (
        AudioEvalExample(
            waveforms[0], 18, prompt, low_answer, (low_answer, high_answer), 0
        ),
        AudioEvalExample(
            waveforms[1], 14, prompt, high_answer, (low_answer, high_answer), 1
        ),
    )
    summary = evaluate_audio_language_model(model, examples, 2, vocabulary.eos_token_id)

    print(f"waveform batch shape:        {tuple(waveforms.shape)}")
    print(f"sample lengths:              {lengths.tolist()}")
    print(f"audio prefix token count:    {model.sequence_embedding.audio_token_count}")
    print(f"initial training loss:       {initial_loss:.4f}")
    print(f"final training loss:         {output.loss.item():.4f}")
    print(f"generated text:              {vocabulary.decode(generated)}")
    print(f"generation exact match:      {summary.generation_exact_match:.3f}")
    print(f"candidate ranking accuracy:  {summary.candidate_accuracy:.3f}")
    print("Generation and evaluation were carried forward, not learner TODOs.")


if __name__ == "__main__":
    main()
