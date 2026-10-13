# Review checklist

- Does velocity combine noise and clean sample with the correct signs?
- Are `alpha` and `sigma` square roots of schedule powers?
- Can every target reconstruct both `x_0` and epsilon?
- Does conversion preserve arbitrary sample dimensions?
- Is SNR signal power divided by noise power?
- Are Min-SNR weights different for epsilon, sample, and velocity prediction?
- Is MSE reduced per example before timestep weighting?
- Does the weighted loss preserve gradients to the prediction?
- Can you explain why equivalent targets still produce different loss scales?
