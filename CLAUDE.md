# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for skyrmion crystals in a two-dimensional continuum chiral ferromagnet: exchange, bulk (Bloch) DMI, perpendicular field and optional easy-axis anisotropy. There is no package, build or test suite. Each entry point is self-contained (it computes everything and loads no data files), and the implementations are cross-validated against each other:

- `Skyrmions_Aristov_Tsypilnikov.nb` — Mathematica reference for Aristov–Tsypilnikov (arXiv:1705.00196): 24-digit shooting profile, Bessel-basis BdG magnon spectrum, quantum energy, lattice bands, 3D/thermal extensions. Source of the benchmark numbers used elsewhere.
- `Skyrmions_Pires_classical.jl` — Pluto notebook for the classical part of Pires (arXiv:2610.23490); anisotropy only in its Sec. 3.3.
- `Skyrmions_anisotropy_classical.jl` — Pluto notebook: the same model with easy-axis anisotropy `K_z`, sliders for J, D, B, K_z.
- `phase_diagram_ba.jl` — CLI script for the classical (b, a) ground-state diagram at a fixed disk radius; its environment is `scripts/` (Plots only).
- `monte_carlo_2d.jl` — CLI script: Metropolis Monte Carlo of the same model on a periodic square lattice and the finite-temperature (b, θ) phase diagram at fixed a; with `--geometry disk --scan ba` the (b, a) diagram at fixed θ of a small disk whose rim is frozen at +z (the circular-cell approximation of `phase_diagram_ba.jl` at finite temperature). Same `scripts/` environment. Method and the implementation choices: `README_monte_carlo_2d.md` (Sec. 15 documents the script, 15.5 the disk).

## Commands

Julia 1.10+. Phase diagram:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=scripts -t auto phase_diagram_ba.jl --diagram-only --step 0.05 --R 10
julia --project=scripts phase_diagram_ba.jl --help
# smoke test against the Mathematica benchmark: the Skyrmion row must print ρ ≈ -0.069124 (exact: -0.0691244243)
julia --project=scripts phase_diagram_ba.jl --b 0.6 --a 0 --R 4.018913753672924 --nb 2 --na 2 --outdir "$(mktemp -d)"
```

Use `-t auto` for real grids, because grid points run in parallel threads. Output files are always named `phase_diagram_ba.{csv,png}` in `--outdir` (default `outputs/`), so give each run its own directory. `--step 0.005` (32 361 points) takes 3–7 min on 8 threads. `include("phase_diagram_ba.jl")` runs a full default diagram, because `ARGS` parsing and `main()` sit at top level.

Monte Carlo ((b, θ) diagram on the periodic lattice, or (b, a) diagram of the pinned-rim disk):

```bash
julia --project=scripts -t auto monte_carlo_2d.jl --selftest   # 20 checks of README Sec. 13.2 and 15.5 (~10 s), exit 1 on failure
julia --project=scripts -t auto monte_carlo_2d.jl --bench      # attempts/s for the configured lattice
julia --project=scripts -t auto monte_carlo_2d.jl --L 24 --nb 3 --ntheta 3 --neq 50 --nprod 50 --outdir "$(mktemp -d)"   # pipeline smoke test, ~20 s
julia --project=scripts -t auto monte_carlo_2d.jl --L 48 --nb 11 --ntheta 12 --neq 300 --nprod 300 --nquench 100 --outdir "$(mktemp -d)"   # physics sanity test, ~30 s: H → S → F at low θ, P on top
julia --project=scripts -t auto monte_carlo_2d.jl              # default: a = 0, h = 0.5, 72², 21 b × 24 θ, 4 histories, ~5 min on 8 threads
julia --project=scripts -t auto monte_carlo_2d.jl --geometry disk --scan ba --h 0.25 --R 4.5 --theta 0.1 --nb 6 --na 5 --neq 300 --nprod 300 --outdir "$(mktemp -d)"   # disk (b, a) smoke test, ~20 s: Skyrmion below a line from b ≈ 0.7 (a = 0) to b ≈ 0.1 (a = 0.8), Polarized above
julia --project=scripts -t auto monte_carlo_2d.jl --geometry disk --scan ba --h 0.25 --R 4.5 --theta 0.1 --neq 2000 --nprod 2000   # disk (b, a) diagram, 21 × 17, ~2 min on 8 threads: cluster job
julia --project=scripts monte_carlo_2d.jl --help
```

**Do not run heavy jobs locally.** The local machine is for the self-test, smoke tests and tiny grids that finish in well under a minute or two; anything larger (the default survey, full 21 × 17 or 21 × 24 grids, 10³–10⁵ sweeps per point, chained runs, parameter sweeps) is a Dirac cluster job (`dirac-slurm` skill): prepare the command and the job script instead of running it here, and never re-run a multi-minute chain locally after a code change. The help text gives the cost formula and a Slurm-scale command. A disk (b, a) diagram at R = 4.5, h = 0.25 with 2 × 10³ + 2 × 10³ sweeps is ≈ 4 × 10⁹ attempts (≈ 2 min on 8 threads): cluster.

Each run writes its own `outputs/monte_carlo_2d/<run-id>/` (`--run-id`, default from a, h, L and a timestamp; `disk_R<R>_theta<θ>_h<h>_<timestamp>` for the disk). Trajectories (history × seed × b[× a]) run on a thread pool; threads left over on single-point runs go to a checkerboard-parallel sweep. `main(parse_args(ARGS))` runs only when the file is the program (`include` loads the definitions, which is how to test functions from a scratch script). Run the self-test after changing the Hamiltonian, the local field, the proposal, the charge, the structure factor or the disk mask.

Pluto notebooks (Pluto in the global environment; each notebook installs its packages from its embedded environment on first open):

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'   # once
julia -e 'using Pluto; Pluto.run(notebook="Skyrmions_Pires_classical.jl")'
```

Headless check of a Pluto notebook, the closest thing to a test run. It evaluates every cell without writing the file, then exits 1 on any errored cell or any `false` in a rendered table (each notebook has a final pass/fail check table). It takes ~30–45 s once packages are cached:

```bash
julia -e 'using Pluto; s = Pluto.ServerSession(); s.options.server.disable_writing_notebook_files = true
nb = Pluto.SessionActions.open(s, abspath(ARGS[1]); run_async=false)
bad = filter(c -> c.errored || (c.output.body isa String && occursin(r"<td[^>]*>\s*false\s*</td>", c.output.body)), nb.cells)
foreach(c -> println(c.errored ? "ERRORED " : "FALSE ", c.cell_id, "  ", c.errored ? c.output.body[:msg] : ""), bad)
println(length(nb.cells), " cells, ", length(bad), " bad"); Pluto.SessionActions.shutdown(s, nb; verbose=false); exit(isempty(bad) ? 0 : 1)' Skyrmions_Pires_classical.jl
```

Mathematica: evaluate the whole notebook in a fresh kernel (Evaluation → Evaluate Notebook). A full serial run takes ~27 min, but results are saved in the file, including the final pass/fail table, so read the saved outputs instead of rerunning. Each of the 27 input cells has a unique `CellTags` name (`parameters`, `profile-definitions`, `bessel-bdg-definitions`, `quantum-definitions`, …, `verification-summary`); search for the name to find a cell. Run settings live in `parameters`.

## Model and conventions

All code uses C = J = D = 1: lengths in C/D, energies in C, densities in D²/C, always relative to the saturated ferromagnet. `b = BC/D²` and `a = K_z C/D²`; the `.nb` writes `b = BC/(sD²)` for spin s. Anisotropy enters as `a(1 − n_z²) = a sin²β`, so the ferromagnet stays at zero. The Bloch ansatz is `Φ = φ + π/2`, `Θ = β(r)` with `β(0) = π`, `β(R) = 0`. The skyrmion crystal is the circular-cell approximation: a disk with the hexagonal cell's area and a Dirichlet edge, `ρ_C = E/(πR²)`. The radial density

`ρ(r) = ½β'² + sin²β/(2r²) + β' + sin2β/(2r) + b(1 − cos β) + a sin²β`

and its EL equation are Aristov–Tsypilnikov Eqs. (8)–(9) plus the anisotropy terms.

- The topological-charge sign differs for the same texture: the Julia notebooks use the Pires Eq. (41) orientation (`Q = −1`), the `.nb` uses the paper's (`Q = +1`).
- Comments and markdown cite source equations and figures ("Pires Eq. (64)", "Aristov–Tsypilnikov Fig. 4"); keep doing so.

## Duplicated solver code

There is no shared module, because each Pluto notebook must be self-contained. The EL solver is copied into all three `.jl` files: `el_residual!`, `wall_guess`, `newton_profile` (damped Newton, tridiagonal Jacobian), `solve_profile` (several 360°-wall initial guesses; accepts only monotone profiles in [0, π]), `energy_density` and `golden`. `ρ_helix`, `cone_state`, `helicoid_density` and `helicoid_state` are copied between the anisotropy notebook and the script. A change to the functional, the EL equation or its Jacobian must go into every copy. The copies differ on purpose:

- radial intervals: `N_grid = 2000` in the notebooks, `--N 1200` in the script; default `golden` tol 1e-6 vs 1e-5;
- helicoid integrals: QuadGK in the notebooks, a trapezoid rule in the script (the `scripts` env has only Plots);
- the Pires notebook's `ρ_helix(b)`, `ρ_cone(b)` and `helicoid(b)` take no anisotropy;
- cell radius: `optimal_radius` (Pires) and `optimal_skyrmion` (anisotropy) minimise ρ_C over R (`R_scan` bracket, then golden section); the script's `skyrmion_fixed_R` keeps R fixed.

Both notebooks hard-code Mathematica results and check against them. At b = 0.6, a = 0: R₀ = 4.018913753672924, ρ_C = −0.0691244243113868, E = −3.507508827096083. `bench_R0`/`bench_ρ` hold the same quantities at 8 fields. After changing the solver, run the headless check on both notebooks and the script's smoke test.

## Phase-diagram caveats

- If Newton fails, `solve_profile` falls back to the wall guess and its energy still competes. The CSV (`b,a,ground_id,ground,rho_min`) has no convergence flag.
- `--N` counts intervals on [0, R], so the grid spacing grows with `--R`.
- The cone propagates along the field, so it is a bulk (3D) competitor; a strictly 2D film would exclude it. At a = 0 it lies below the optimal-radius skyrmion crystal at every field.

## Monte Carlo caveats

- Lattice units: `J_lat = 1`, `D_lat = h`, `B_lat = b h²`, `K_lat = a h²`; `θ = k_B T / J_lat`. The continuum-scaled density `ρ = (H/N + 2 + B_lat)/h²` is comparable with the script's ρ only after the T = 0 quench (`rho_quench` column); `⟨ρ⟩` at finite θ contains ≈ θ/h² of thermal energy.
- Cooling runs trap in stripe/skyrmion mixtures; the low-θ diagram is decided by the lowest-energy history (`--resolve energy`, cells marked with a dot, `agree = false` in `phase_diagram.csv`). This is a ground-state proxy, not a free-energy rule. The seeded crystal keeps the seed's skyrmion count (`--skx-cells`).
- Labels come from thresholds (`--peak-min`, `--qmin`, `--psi6-min`, `--mz-pol`); Polarized vs Paramagnet is a crossover. Charge convention: Bloch core down in an up background has Q = −1 (same as the Pluto notebooks, opposite to the `.nb`).
- Parallel checkerboard sweeps need even `Nx`, `Ny` (the script falls back to one block otherwise); `--site-order random` is always serial. Results with the same `--seed` are reproducible for the same `--inner-threads`.
- The T = 0 quench is `nquench/10` downhill Metropolis sweeps followed by local-field alignment sweeps (`align_sweep!`, monotone for easy-axis K ≥ 0) until the spin change per sweep is < 1e-9; the Metropolis stage alone does not converge the energy.
- Disk (`--geometry disk`): the box is odd and chosen from `--R` and `--h` (`disk_box`), `--L` is ignored, the rim is frozen at +z, one sweep is `N_free` attempts, `m_z` is over free spins, and `ρ` divides by `N_free h²`. Use `h ≤ 0.25` (the b = 0.6, R₀ benchmark is 1.5 % off at h = 0.25, 14 % at h = 0.5). It holds a single skyrmion, so its labels are Skyrmion / Multiple skyrmions / Polarized / Helical / Paramagnet (`classify_disk`, thresholds `--qsk-min`, `--qstd-max`, `--mz-pol`); helix, helicoid and cone cannot appear. The `--scan ba` diagram is taken at the lowest θ of the ladder; its dashed line is the zero contour of `rho_quench_skx`.
- Parallel tempering (README Sec. 11.1) is not implemented.

## Editing Pluto files

Each cell starts with `# ╔═╡ <uuid>`. The `# ╔═╡ Cell order:` list at the end sets display order (`╟─` hides code, `╠═` shows it); a cell missing from the list is appended at the bottom. Define each global in exactly one cell and keep scratch work in `let` blocks. Never hand-edit `PLUTO_PROJECT_TOML_CONTENTS` or `PLUTO_MANIFEST_TOML_CONTENTS`. To add a package, add it to the `using` cell, then run the headless check without the `disable_writing_notebook_files` line; Pluto resolves the package and rewrites both blocks.

## Gitignored local files

`outputs/`, `archive/` and `*.pdf` are not tracked:

- `archive/` holds the historical exploratory notebooks; nothing depends on them.
- `outputs/` holds phase-diagram runs from the Dirac cluster, renamed `phase_diagram_ba_R<R>_step<step>.*` with `p` for the decimal point, plus their `*_dirac.log`; `outputs/monte_carlo_2d/<run-id>/` holds Monte Carlo runs. All CSV files (including `dirac_slurm_nodes.csv`) live in `outputs/csv/`.
- `outputs/skyrmion_phase_diagram_literature.md` compares with the literature; Butenko/Leonov `(k, h) = (2a, b)`.
- `theory_pdf/` holds the Yu et al. (2010) supplement that the Monte Carlo spec follows.
