# Hints

## Hint 1

The combined conditioning vector can be as simple as:

```text
time_mlp(time_embedding) + class_embedding(class_labels)
```

## Hint 2

Create a drop mask with `torch.rand(class_labels.shape) < drop_probability`,
then use `torch.where` to choose between the null and original labels.

## Hint 3

The null labels for inference can be created with
`torch.full_like(class_labels, model.null_class)`.

## Hint 4

Test the guidance equation at `w=0` and `w=1` first. Those endpoints should
exactly recover the unconditional and conditional predictions.

## Hint 5

Guidance combines predicted noise, not generated images and not class
embeddings.
