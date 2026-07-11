# Hints

## Hint 1

To add a timestep embedding to image features:

```python
time_bias = self.time_projection(time_emb)
time_bias = time_bias[:, :, None, None]
x = x + time_bias
```

## Hint 2

`torch.cat((upsampled, skip), dim=1)` concatenates image features along the
channel dimension.

## Hint 3

The U-Net output predicts noise, so it should have the same channel count as the
input image.

## Hint 4

If the upsampled tensor and skip tensor do not have the same height/width,
check that your input image size is even and that you pooled/upsampled by 2.
