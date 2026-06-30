import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    TinyPreferenceLM,
    completion_logprobs,
    dpo_loss,
    make_toy_preference_data,
    token_logprobs,
    train_dpo_step,
)


class TableLM(nn.Module):
    """Deterministic next-token model for exact log-prob tests."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table[input_ids]


def test_tiny_preference_lm_returns_logits() -> None:
    model = TinyPreferenceLM(vocab_size=12, hidden_dim=8)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    logits = model(input_ids)

    assert logits.shape == (2, 3, 12)


def test_token_logprobs_gathers_targets() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]],
            [[1.0, 2.0, 0.0], [-2.0, 0.0, 2.0]],
        ]
    )
    targets = torch.tensor([[0, 2], [1, 0]])

    actual = token_logprobs(logits, targets)
    expected = torch.stack(
        [
            F.log_softmax(logits[0, 0], dim=-1)[0],
            F.log_softmax(logits[0, 1], dim=-1)[2],
            F.log_softmax(logits[1, 0], dim=-1)[1],
            F.log_softmax(logits[1, 1], dim=-1)[0],
        ]
    ).reshape(2, 2)

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected)


def test_completion_logprobs_scores_only_completion_tokens() -> None:
    table = torch.zeros(6, 6)
    table[2, 3] = 4.0
    table[3, 4] = 3.0
    table[1, 2] = -5.0
    model = TableLM(table)
    prompts = torch.tensor([[1, 2]])
    completions = torch.tensor([[3, 4]])

    actual = completion_logprobs(model, prompts, completions)
    expected = F.log_softmax(table[2], dim=-1)[3] + F.log_softmax(table[3], dim=-1)[4]

    assert actual.shape == (1,)
    assert torch.allclose(actual, expected.unsqueeze(0))


def test_dpo_loss_rewards_policy_margin_over_reference_margin() -> None:
    policy_chosen = torch.tensor([-1.0, -1.0])
    policy_rejected = torch.tensor([-4.0, -3.0])
    reference_chosen = torch.tensor([-2.0, -2.0])
    reference_rejected = torch.tensor([-3.0, -3.0])

    loss, stats = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )

    assert loss < torch.log(torch.tensor(2.0))
    assert stats.policy_margin > stats.reference_margin
    assert stats.preference_accuracy == 1.0


def test_dpo_loss_is_large_when_policy_prefers_rejected_completion() -> None:
    loss, stats = dpo_loss(
        policy_chosen_logps=torch.tensor([-4.0]),
        policy_rejected_logps=torch.tensor([-1.0]),
        reference_chosen_logps=torch.tensor([-2.0]),
        reference_rejected_logps=torch.tensor([-3.0]),
        beta=1.0,
    )

    assert loss > torch.log(torch.tensor(2.0))
    assert stats.preference_accuracy == 0.0


def test_train_dpo_step_updates_policy_but_not_reference() -> None:
    torch.manual_seed(0)
    policy = TinyPreferenceLM(vocab_size=12, hidden_dim=8)
    reference = TinyPreferenceLM(vocab_size=12, hidden_dim=8)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    prompts, chosen, rejected = make_toy_preference_data()
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference.named_parameters()
    }

    stats = train_dpo_step(
        policy,
        reference,
        optimizer,
        prompts,
        chosen,
        rejected,
        beta=0.5,
    )

    assert torch.isfinite(stats.loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in policy.parameters()
    )
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter, reference_before[name])


def test_dpo_training_improves_chosen_margin() -> None:
    torch.manual_seed(1)
    policy = TinyPreferenceLM(vocab_size=12, hidden_dim=12)
    reference = TinyPreferenceLM(vocab_size=12, hidden_dim=12)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.08)
    prompts, chosen, rejected = make_toy_preference_data()

    with torch.no_grad():
        initial_margin = (
            completion_logprobs(policy, prompts, chosen)
            - completion_logprobs(policy, prompts, rejected)
        ).mean()

    for _ in range(40):
        train_dpo_step(
            policy,
            reference,
            optimizer,
            prompts,
            chosen,
            rejected,
            beta=0.5,
        )

    with torch.no_grad():
        final_margin = (
            completion_logprobs(policy, prompts, chosen)
            - completion_logprobs(policy, prompts, rejected)
        ).mean()

    assert final_margin > initial_margin + 2.0
