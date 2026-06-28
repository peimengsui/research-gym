# Concept: CEM planning

A world model can answer:

```text
If I start here and take this action sequence, where will I end up?
```

Planning turns that question around:

```text
Which action sequence gets me closest to a goal?
```

Cross-entropy method (CEM) is a simple model-based planner:

```text
1. sample many action sequences from a Gaussian
2. imagine each sequence with the world model
3. score each imagined trajectory
4. keep the elite top-k sequences
5. refit the Gaussian to those elites
6. repeat
```

## Why plan in imagination?

Real environment interaction can be slow, expensive, or unsafe. If the world
model is good enough, you can search for actions entirely inside the model.

## Why CEM?

CEM is easy to implement and works well on low-dimensional continuous control
problems. It does not need gradients through the reward or the rollout.

The tradeoff is sample efficiency. CEM needs many imagined rollouts, but for a
tiny lesson-scale model that is acceptable.

## What counts as success?

In this lesson, success means the imagined final observation is close to a
target state. The reward is negative squared distance:

```text
reward = -|| o_T - target ||^2
```

Higher reward is better.

## Connection to the world-model loop

This lesson assumes you already have:

```text
initial observation -> rollout(actions) -> imagined observations
```

CEM wraps that rollout inside a search loop.
