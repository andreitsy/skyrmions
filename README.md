# Skyrmion crystal calculations

- [`Skyrmions_Aristov_Tsypilnikov.nb`](Skyrmions_Aristov_Tsypilnikov.nb) — Mathematica notebook for [Aristov–Tsypilnikov](https://arxiv.org/abs/1705.00196). Evaluate the notebook in a fresh kernel. Older notebooks are in [`archive/`](archive/).
- [`Skyrmions_Pires_classical.jl`](Skyrmions_Pires_classical.jl) — Pluto notebook for the classical model of [Pires](https://arxiv.org/abs/2610.23490).
- [`Skyrmions_anisotropy_classical.jl`](Skyrmions_anisotropy_classical.jl) — same model with easy-axis anisotropy.
- [`phase_diagram_ba.jl`](phase_diagram_ba.jl) — classical `(b, a)` ground-state diagram at fixed skyrmion radius (`J = D = 1`).
- [`monte_carlo_2d.jl`](monte_carlo_2d.jl) — Metropolis Monte Carlo of the same model on a periodic lattice and its `(b, θ)` phase diagram at fixed anisotropy; method in [`README_monte_carlo_2d.md`](README_monte_carlo_2d.md).

## Pluto

Julia 1.10+. The notebook installs its packages on first open.

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'   # once
julia -e 'using Pluto; Pluto.run(notebook="Skyrmions_Pires_classical.jl")'
```

## Phase diagram

Compares ferromagnet, helix, helicoid, cone, and skyrmion. Radius `R` is fixed (default 10); there is no minimisation over `R`. Writes `outputs/phase_diagram_ba.png` and `.csv`. Run with `-t auto` so the grid is parallel.

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=scripts -t auto phase_diagram_ba.jl --diagram-only --step 0.05 --R 10
julia --project=scripts phase_diagram_ba.jl --help
```

## Monte Carlo (b, θ) diagram

Single-spin Metropolis on a periodic `Nx × Ny` square lattice with `J_lat = 1`, `D_lat = h`, `B_lat = b h²`, `K_lat = a h²` (`h` = mesh spacing in units of `J/D`), temperature `θ = k_B T / J_lat`. Cooling from random spins and heating from seeded helix / skyrmion-lattice / polarized states run as independent trajectories on Julia threads; each `(b, θ)` point is labelled Paramagnet, Polarized, Helical, Skyrmion crystal or Skyrmions (disordered) from the Berg–Lüscher charge, the Bragg peaks of `S(q)`, the sixfold order `ψ₆` of the detected cores and `⟨m_z⟩`. Writes `outputs/monte_carlo_2d/<run-id>/{summaries.csv, phase_diagram.csv, observables.csv, metadata.toml, snapshots/, plots/}`.

```bash
julia --project=scripts -t auto monte_carlo_2d.jl --selftest   # correctness checks (README_monte_carlo_2d.md Sec. 13.2)
julia --project=scripts -t auto monte_carlo_2d.jl              # default survey: a = 0, h = 0.5, 72², ~5 min on 8 threads
julia --project=scripts monte_carlo_2d.jl --help
```
