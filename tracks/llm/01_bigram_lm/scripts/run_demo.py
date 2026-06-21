"""Train the learner's bigram model on a tiny inline character corpus."""

import torch

from implementation import BigramLanguageModel


def main() -> None:
    torch.manual_seed(42)

    text = "research gym makes small models understandable.\n" * 12
    characters = sorted(set(text))
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for character, index in stoi.items()}
    data = torch.tensor([stoi[character] for character in text], dtype=torch.long)

    inputs = data[:-1]
    targets = data[1:]
    model = BigramLanguageModel(vocab_size=len(characters))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.08)

    with torch.no_grad():
        _, initial_loss = model(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert initial_loss is not None

    for _ in range(80):
        _, loss = model(inputs.unsqueeze(0), targets.unsqueeze(0))
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, final_loss = model(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert final_loss is not None

    prompt = torch.tensor([[stoi["r"]]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=100)[0].tolist()
    sample = "".join(itos[token] for token in generated)

    print(f"Initial loss: {initial_loss.item():.4f}")
    print(f"Final loss:   {final_loss.item():.4f}")
    print("Generated sample:")
    print(sample)


if __name__ == "__main__":
    main()
