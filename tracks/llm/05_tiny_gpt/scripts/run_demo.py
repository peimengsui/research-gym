"""Train the learner's tiny GPT on a tiny character corpus."""

import torch

from implementation import TinyGPT, make_lm_batch


def main() -> None:
    torch.manual_seed(42)

    text = (
        "research gym builds tiny models. tiny models make research ideas inspectable. "
    ) * 12
    chars = sorted(set(text))
    stoi = {ch: index for index, ch in enumerate(chars)}
    itos = {index: ch for ch, index in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    block_size = 16
    model = TinyGPT(
        vocab_size=len(chars),
        block_size=block_size,
        embed_dim=32,
        num_layers=2,
        expansion_factor=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    generator = torch.Generator().manual_seed(7)

    inputs, targets = make_lm_batch(
        data,
        block_size=block_size,
        batch_size=32,
        generator=generator,
    )
    with torch.no_grad():
        _, initial_loss = model(inputs, targets)
        assert initial_loss is not None

    for _ in range(160):
        inputs, targets = make_lm_batch(
            data,
            block_size=block_size,
            batch_size=32,
            generator=generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    inputs, targets = make_lm_batch(
        data,
        block_size=block_size,
        batch_size=32,
        generator=generator,
    )
    with torch.no_grad():
        _, final_loss = model(inputs, targets)
        assert final_loss is not None
        prompt = torch.tensor([[stoi["r"]]], dtype=torch.long)
        generated = model.generate(prompt, max_new_tokens=80)[0].tolist()

    sample = "".join(itos[token] for token in generated)
    print(f"Vocabulary size: {len(chars)}")
    print(f"Initial loss:    {initial_loss.item():.4f}")
    print(f"Final loss:      {final_loss.item():.4f}")
    print("Sample:")
    print(sample)


if __name__ == "__main__":
    main()
