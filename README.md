# Skyrmion crystal calculations

- [`Skyrmions_Aristov_Tsypilnikov.nb`](Skyrmions_Aristov_Tsypilnikov.nb) — Mathematica notebook for [Aristov–Tsypilnikov](https://arxiv.org/abs/1705.00196). Evaluate the notebook in a fresh kernel. Older notebooks are in [`archive/`](archive/).
- [`Skyrmions_Pires_classical.jl`](Skyrmions_Pires_classical.jl) — Pluto notebook for the classical model of [Pires](https://arxiv.org/abs/2610.23490).
- [`Skyrmions_anisotropy_classical.jl`](Skyrmions_anisotropy_classical.jl) — same model with easy-axis anisotropy.
- [`phase_diagram_ba.jl`](phase_diagram_ba.jl) — classical `(b, a)` ground-state diagram at fixed skyrmion radius (`J = D = 1`).

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
