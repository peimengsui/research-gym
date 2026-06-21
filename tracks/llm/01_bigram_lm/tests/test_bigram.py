import torch

from implementation import BigramLanguageModel


def test_model_instantiates() -> None:
    model = BigramLanguageModel(vocab_size=7)

    assert isinstance(model, BigramLanguageModel)


def test_forward_returns_expected_logits_shape() -> None:
    model = BigramLanguageModel(vocab_size=7)
    idx = torch.randint(0, 7, (2, 4))

    logits, loss = model(idx)

    assert logits.shape == (2, 4, 7)
    assert loss is None


def test_forward_returns_scalar_loss_with_targets() -> None:
    model = BigramLanguageModel(vocab_size=7)
    idx = torch.randint(0, 7, (2, 4))
    targets = torch.randint(0, 7, (2, 4))

    _, loss = model(idx, targets)

    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_forward_returns_no_loss_without_targets() -> None:
    model = BigramLanguageModel(vocab_size=7)
    idx = torch.randint(0, 7, (2, 4))

    _, loss = model(idx)

    assert loss is None


def test_generate_extends_sequence() -> None:
    torch.manual_seed(0)
    model = BigramLanguageModel(vocab_size=7)
    prompt = torch.zeros((2, 3), dtype=torch.long)

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (2, 8)
    assert torch.equal(generated[:, :3], prompt)


def test_gradients_flow_to_embedding_parameters() -> None:
    model = BigramLanguageModel(vocab_size=7)
    idx = torch.randint(0, 7, (2, 4))
    targets = torch.randint(0, 7, (2, 4))

    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()

    gradient = model.token_embedding_table.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
