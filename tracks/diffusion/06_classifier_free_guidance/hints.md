# Hints

## Hint 1

The class embedding table needs `num_classes + 1` rows. The last row is selected
when the class condition is dropped.

## Hint 2

The combined conditioning vector can be as simple as:

```text
time_mlp(time_embedding) + class_embedding(class_labels)
```

## Hint 3

Create a drop mask with `torch.rand(class_labels.shape) < drop_probability`,
then use `torch.where` to choose between the null and original labels.

## Hint 4

The null labels for inference can be created with
`torch.full_like(class_labels, model.null_class)`.

## Hint 5

Test the guidance equation at `w=0` and `w=1` first. Those endpoints should
exactly recover the unconditional and conditional predictions.

## Hint 6

Guidance combines predicted noise, not generated images and not class
embeddings.
