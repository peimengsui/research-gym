# Concept: flow matching

This lesson uses a straight path between a Gaussian noise sample and a data
sample:

```text
x_0 = noise
x_1 = data
x_t = (1 - t) * noise + t * data,  t in [0, 1]
```

Differentiating with respect to continuous time gives a constant target
velocity for each paired example:

```text
u_t = dx_t/dt = data - noise
```

A neural network receives a path point and its time and predicts this velocity:

```text
v_theta(x_t, t)
```

The training objective is:

```text
L_FM = E[||v_theta(x_t, t) - (data - noise)||^2]
```

Although each training pair follows a straight line, the learned marginal
velocity field need not be globally constant. At a given location and time, the
model averages information from many possible noise-data pairings.

## Generation as an ODE

After training, begin at Gaussian noise and solve:

```text
dx/dt = v_theta(x, t),  from t=0 to t=1
```

The simplest numerical update is explicit Euler:

```text
x_next = x + dt * v_theta(x, t)
```

The midpoint method evaluates the field twice:

```text
k1 = v_theta(x, t)
x_mid = x + (dt / 2) * k1
k2 = v_theta(x_mid, t + dt / 2)
x_next = x + dt * k2
```

Midpoint is second-order accurate and is often much more accurate than Euler at
the same step count, though it requires twice as many model evaluations.

## How this differs from DDPM and DDIM

This lesson has no beta schedule and no discrete noise-prediction target.
Training samples continuous times directly, and generation uses a general ODE
solver. DDIM can also be interpreted through an ODE perspective, but flow
matching defines its training target directly as a velocity along a chosen
probability path.

## What is omitted

Modern systems may use optimal-transport couplings, conditional paths, latent
spaces, classifier-free guidance, adaptive solvers, or alternative parameter
choices. The straight path and fixed-step solvers here expose the essential
mechanics without those additions.
