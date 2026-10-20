# Skyrmion crystal calculations

- [`Skyrmions_Aristov_Tsypilnikov.nb`](Skyrmions_Aristov_Tsypilnikov.nb) — Mathematica notebook for [Aristov–Tsypilnikov](https://arxiv.org/abs/1705.00196). Evaluate the notebook in a fresh kernel. Older notebooks are in [`archive/`](archive/).
- [`Skyrmions_Pires_classical.jl`](Skyrmions_Pires_classical.jl) — Pluto notebook for the classical model of [Pires](https://arxiv.org/abs/2610.23490).
- [`Skyrmions_anisotropy_classical.jl`](Skyrmions_anisotropy_classical.jl) — same model with easy-axis anisotropy.
- [`phase_diagram_ba.jl`](phase_diagram_ba.jl) — classical `(b, a)` ground-state diagram at fixed skyrmion radius (`J = D = 1`).
- [`monte_carlo_2d.jl`](monte_carlo_2d.jl) — Metropolis Monte Carlo of the same model: `(b, θ)` phase diagram at fixed anisotropy on a periodic lattice, or `(b, a)` diagram at fixed θ of a small disk with a pinned rim (the circular cell of `phase_diagram_ba.jl` at finite temperature); method in [`README_monte_carlo_2d.md`](README_monte_carlo_2d.md).

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

## Monte Carlo

Single-spin Metropolis on a square lattice with `J_lat = 1`, `D_lat = h`, `B_lat = b h²`, `K_lat = a h²` (`h` = mesh spacing in units of `J/D`), temperature `θ = k_B T / J_lat`. Two geometries:

- **Periodic lattice, `(b, θ)` diagram at fixed `a`** (default). Cooling from random spins and heating from seeded helix / skyrmion-lattice / polarized states run as independent trajectories on Julia threads; each `(b, θ)` point is labelled Paramagnet, Polarized, Helical, Skyrmion crystal or Skyrmions (disordered) from the Berg–Lüscher charge, the Bragg peaks of `S(q)`, the sixfold order `ψ₆` of the detected cores and `⟨m_z⟩`.
- **Pinned-rim disk, `(b, a)` diagram at fixed `θ`** (`--geometry disk --scan ba`). Only the sites within radius `R` are updated; the rest are frozen at `+z`, which is the Dirichlet rim of the circular cell in `phase_diagram_ba.jl`. The cell holds one skyrmion or none, so the labels are Skyrmion, Multiple skyrmions, Polarized, Helical or Paramagnet, and the quenched energy density is directly comparable with `ρ_C` of the continuum solver (1.5 % at `h = 0.25`). Use `h ≤ 0.25`.

Writes `outputs/monte_carlo_2d/<run-id>/{summaries.csv, phase_diagram.csv, observables.csv, metadata.toml, snapshots/, plots/}`. Keep local runs to the self-test and smoke tests; full grids are cluster jobs.

```bash
julia --project=scripts -t auto monte_carlo_2d.jl --selftest   # correctness checks (README_monte_carlo_2d.md Sec. 13.2, 15.5)
julia --project=scripts -t auto monte_carlo_2d.jl              # default survey: a = 0, h = 0.5, 72², ~5 min on 8 threads (cluster)
julia --project=scripts -t auto monte_carlo_2d.jl --geometry disk --scan ba --h 0.25 --R 4.5 --theta 0.1   # pinned-rim disk, (b, a) at θ = 0.1, ~2 min on 8 threads (cluster)
julia --project=scripts monte_carlo_2d.jl --help
```
