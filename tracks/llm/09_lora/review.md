# Review questions

- Why does LoRA freeze the base weights?
- Why does initializing `lora_b` to zero preserve the base model at step 0?
- What shape is `lora_b.weight @ lora_a.weight`?
- Why might you merge LoRA weights before inference?
- What tradeoff changes when you increase the rank?
