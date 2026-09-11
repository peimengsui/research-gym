# Concept: uncertainty is not one thing

`wm.13` represented several valid futures with a mixture model. A wide
distribution can therefore mean the world is genuinely multimodal, not that the
model is ignorant. Penalizing all predictive spread would make the planner avoid
valid choices merely because they have several possible outcomes.

An ensemble gives us a useful controlled distinction:

- **Aleatoric uncertainty** is variation among mixture components inside a
  model. It represents outcome ambiguity that the model believes is inherent.
- **Epistemic uncertainty** is disagreement among different models' expected
  futures. It represents uncertainty about what dynamics to believe.

## The controlled comparison

This lesson provides two action candidates:

```text
familiar multimodal detour
unfamiliar shortcut
```

For the familiar plan, all three WAM members predict the same two outcome modes.
Its within-model spread is high, but ensemble disagreement is zero.

For the shortcut, every individual model is confident, but the models predict
success, early stopping, and overshoot respectively. Its within-model spread is
zero while ensemble disagreement is high.

Both candidates have the same ensemble-mean future. The shortcut uses less
action effort, so a nominal planner prefers it. An epistemic-risk penalty changes
the choice to the familiar plan.

## Decomposing mixture-ensemble variance

First compute each member's mixture expectation:

```text
member_mean[m] = sum_k p[m, k] * future[m, k]
```

Aleatoric uncertainty measures component spread around that member mean:

```text
E_members E_components[||future_component - member_mean||^2]
```

Epistemic uncertainty measures member means around the ensemble mean:

```text
E_members[||member_mean - ensemble_mean||^2]
```

This lesson averages over horizon and state dimensions so each candidate gets
one scalar of each type. It uses population means because the objective is a
small deterministic planning signal, not an unbiased statistical estimator.

## Risk-adjusted planning

The nominal score is:

```text
nominal = -(final goal error + action penalty * action effort)
```

The uncertainty-aware score is:

```text
adjusted = nominal - risk_coefficient * epistemic uncertainty
```

Aleatoric uncertainty is reported but deliberately not penalized in this
controlled example: both familiar modes are considered valid. In a real safety
problem, some outcome modes may themselves have harmful consequences and should
be evaluated by the reward or constraint model. Separating uncertainty types
does not mean aleatoric risk is universally harmless.
