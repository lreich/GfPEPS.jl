# Kitaev energy from the covariance matrix

This note explains how the mean energy density of the vortex-free Kitaev model is obtained directly from the momentum-space covariance matrix produced by the Gaussian map when the fiducial state carries a single physical fermion ($N_f = 1$).

## Setup

For each crystal momentum $k$, the Fourier-transformed covariant matrix returned by `GaussianMap` is a $2 \times 2$ antisymmetric matrix
$$
\Gamma_k = \begin{pmatrix}
\Gamma_{11}(k) & \Gamma_{12}(k) \\
\Gamma_{21}(k) & \Gamma_{22}(k)
\end{pmatrix}, \qquad \Gamma_{21}(k) = -\Gamma_{12}(k).
$$
The Majorana ordering used in the code is $(c_{1,k}, c_{2,k})$, with the corresponding complex (Dirac) fermion
$$
 f_k = \tfrac{1}{2}\bigl(c_{1,k} + i c_{2,k}\bigr), \qquad f_k^\dagger = \tfrac{1}{2}\bigl(c_{1,k} - i c_{2,k}\bigr).
$$
All expectation values quoted below are taken in the Gaussian state determined by $\Gamma_k$.

The Kitaev Hamiltonian in momentum space splits into a normal term labelled by $\xi_k$ and a pairing term proportional to $\Delta_k$:
$$
 H = \sum_k \Bigl[\xi_k\, \bigl(f_k^\dagger f_k - \tfrac{1}{2}\bigr) + \Delta_k f_{-k} f_k + \overline{\Delta_k} f_k^\dagger f_{-k}^\dagger\Bigr].
$$
Consequently, the energy density is the sum of a normal and an anomalous contribution that we can read off from the covariance matrix.

## Normal (occupancy) channel

The occupation number is reconstructed from the off-diagonal Majorana correlator:
$$
 n_k \equiv \langle f_k^\dagger f_k \rangle = \tfrac{1}{2}\bigl(1 - \Gamma_{12}(k)\bigr).
$$
Subtracting the $1/2$ that appears in the Hamiltonian gives the **normal contribution**
$$
 E^{\text{normal}}_k = \xi_k \bigl(n_k - \tfrac{1}{2}\bigr) = -\tfrac{1}{2}\, \xi_k\, \Gamma_{12}(k).
$$
This is exactly the term implemented in `loss.jl` as `normal = -0.5 * dot(ξk_batched, γ12)`.

## Anomalous (pairing) channel

The pair expectation value $\langle f_{-k} f_k \rangle$ is obtained from the full covariance matrix. Using the Majorana-to-Dirac conversion,
$$
 F_k \equiv \langle f_{-k} f_k \rangle = \tfrac{1}{4}\Bigl(\Gamma_{11}(k) + i\,\Gamma_{12}(k) + i\,\Gamma_{21}(k) - \Gamma_{22}(k)\Bigr).
$$
The **anomalous contribution** is the real part of $\Delta_k F_k$:
$$
 E^{\text{anom}}_k = \operatorname{Re}\bigl[\Delta_k F_k\bigr].
$$
This matches the implementation `anomalous = real(sum(Δk_batched .* Fk))` where `Fk` evaluates the combination above for every $k$.

## Total energy density

The mean energy density is the Brillouin-zone average of both channels:
$$
 \varepsilon = \frac{1}{N_k} \sum_k \Bigl( E^{\text{normal}}_k + E^{\text{anom}}_k \Bigr) = \frac{1}{N_k} \sum_k \left[ -\tfrac{1}{2}\, \xi_k\, \Gamma_{12}(k) + \operatorname{Re}\bigl(\Delta_k F_k\bigr) \right].
$$
Here $N_k$ is the number of sampled momenta. In code this is expressed as
```
E = (normal + anomalous) * invN
```
inside `energy_loss(::Kitaev, ::BrillouinZone2D)`.

## Practical remarks

- The derivation only assumes the qq Majorana ordering $(c_{1,k}, c_{2,k})$ used throughout the library. A different ordering would permute the signs in the formulas for $n_k$ and $F_k$.
- For $N_f > 1` the expressions generalise, but additional block structure appears in $\Gamma_k$; the current implementation asserts `size(CM_out, 2) == 2` to keep the single-fermion case explicit.
- The same reconstruction strategy can be reused in other models: identify which Majorana correlators encode the Dirac bilinears appearing in the Hamiltonian, then average over the Brillouin zone.
