# Supervised Fine-Tuning Data

Supervised fine-tuning, or SFT, teaches a language model to imitate desired
assistant responses. The model is still trained with next-token prediction, but
the labels are masked so the loss is computed only on assistant answer tokens.

In this lesson you will implement the small but important data pipeline:

- represent a chat as role-tagged tokens
- build `input_ids` and shifted `labels`
- replace non-assistant labels with `-100`
- pad multiple examples into a batch
- compute masked causal-LM cross-entropy

The demo trains a tiny next-token model on toy chat examples. The goal is not a
useful chatbot; it is to make the SFT tensor contract feel obvious.

## Start

```bash
uv run rgym start llm.08_sft_data
cd workspace/llm.08_sft_data
uv run rgym test
uv run rgym run
```
