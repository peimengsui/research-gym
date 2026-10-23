# Concept: hard visual constraints and soft text guidance

Conditioned video generation combines two different kinds of information.

The first frame is a hard condition: the final video must contain exactly those
pixels at frame zero. The text is a soft condition: it should influence how the
future develops, but classifier-free guidance can strengthen or weaken that
influence.

## Conditioning the denoiser

The provided model repeats the clean first frame across time and concatenates
it with the noisy video as extra input channels. It also encodes a tiny text
sequence such as `[move, right]`. Spatiotemporal feature tokens cross-attend to
those text vectors before the final spatial block.

```text
noisy video + repeated first frame -> spatial/temporal features
text tokens -> context vectors --------------------|
                                                   v
                                           cross-attention
                                                   |
                                                   v
                                           predicted epsilon
```

During training, some text prompts are replaced with null tokens. The model
therefore learns both text-conditional and text-unconditional predictions while
always retaining the first-frame condition.

## Classifier-free text guidance

At sampling time, the model runs twice at each reverse step:

```text
epsilon_uncond = model(x_t, first_frame, null_text)
epsilon_cond   = model(x_t, first_frame, requested_text)

epsilon_guided = epsilon_uncond
               + scale * (epsilon_cond - epsilon_uncond)
```

Here, “unconditional” means unconditional with respect to text. Both passes
still receive the first frame. A scale of zero ignores the text prediction, one
uses the ordinary conditional prediction, and values above one amplify the
text-specific difference.

## Why not paste the clean first frame at every noisy step?

At an intermediate timestep, the model expects every generated pixel to have
the corresponding noise level. Inserting a perfectly clean frame into `x_t`
creates an off-distribution mixture.

Instead, sample one fixed first-frame noise tensor and place the known frame on
its own forward trajectory:

```text
known_t = sqrt(alpha_bar_t) * first_frame
        + sqrt(1 - alpha_bar_t) * fixed_noise
```

After each reverse update, replace frame zero with `known_s` for the next
selected timestep `s`. When `s = -1`, insert the original clean first frame.
Using the same fixed noise at every step gives one coherent known-pixel path.

## Direction alignment

The toy vocabulary supports `right`, `down`, `left`, and `up`. Compute the soft
foreground center of each frame, subtract the first center from the last, and
take cosine similarity with the requested unit direction.

```text
 1  -> aligned with the instruction
 0  -> no net movement or orthogonal movement
-1  -> moved in the opposite direction
```

This metric checks instruction following only. It does not measure visual
quality or temporal smoothness.

## Scope boundary

This lesson uses tiny tokenized commands and an exact first-frame clamp. It does
not use a pretrained text encoder, free-form language, optical flow, latent
video compression, or long clips. Those additions would hide the interaction
between conditioning, guidance, and reverse diffusion.
