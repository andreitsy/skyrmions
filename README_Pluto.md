# Running the Pluto notebook

This guide explains how to open and explore [`Skyrmions_Pires_classical.jl`](Skyrmions_Pires_classical.jl), a Julia [Pluto](https://plutojl.org) notebook that reproduces the classical skyrmion model from [arXiv:2610.23490](https://arxiv.org/abs/2610.23490) (Pires), with numerical benchmarks against [`Skyrmions_Aristov_Tsypilnikov.nb`](Skyrmions_Aristov_Tsypilnikov.nb).

## Requirements

- [Julia](https://julialang.org/downloads/) 1.10 or newer (the notebook was last run with 1.13)
- A web browser (Pluto opens in the browser; no Jupyter needed)

## 1. Install Pluto (once)

In a terminal:

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'
```

## 2. Start Pluto

From this repository directory:

```bash
cd /path/to/skyrmions
julia -e 'using Pluto; Pluto.run()'
```

Or start a Julia REPL and run:

```julia
using Pluto
Pluto.run()
```

A browser tab opens with the Pluto home page. Click **Open a notebook**, then choose `Skyrmions_Pires_classical.jl` from this folder (or paste its full path).

To open the file directly:

```bash
julia -e 'using Pluto; Pluto.run(notebook="Skyrmions_Pires_classical.jl")'
```

## 3. First evaluation

Pluto evaluates cells reactively from top to bottom. On the first open:

1. The notebook’s **embedded package environment** installs `Plots`, `PlutoUI`, `QuadGK`, and `SpecialFunctions` (no separate `Project.toml` is required).
2. Wait until the status bar shows all cells done (about one minute after package precompilation; longer on the first install).

You do not need to run cells by hand. If something is stale, use Pluto’s **Run all cells** (or reopen the notebook).

## 4. Exploring the notebook

Use the **Contents** sidebar to jump between sections:

| Section | What to look at |
| --- | --- |
| 1. Conventions | Units, ansatz, equation checks |
| 2. Belavin–Polyakov | Profile, textures, topological charge \(Q\), Bogomolny bound |
| 3. Skyrmion with DMI | Euler–Lagrange profile, energy, optimal disk radius |
| 3.1 Isolated skyrmion | Collapse field \(b_c\) where \(E_{\rm iso}=0\) |
| 3.2 Domain-wall ansatz | Wall approximations vs the exact profile |
| 3.3 Easy-axis anisotropy | Scaling of radius and wall width |
| 4. Comparison | Helix, helicoid, cone, ferromagnet vs skyrmion crystal |
| 5. Edge | Linearised \(K_1\) / \(I_1\) tail on a disk |
| 6. Verification | Pass/fail checks against analytics and the Mathematica notebook |

Interactive controls (PlutoUI sliders) appear in Section 3: change the field \(b\) and disk radius \(R\) and the profile, energy density \(\rho_C\), and total energy update automatically.

Code cells are collapsed by default in static exports; in Pluto, click a cell’s left edge or use **View code** / expand to edit. Changing a definition re-runs only the cells that depend on it.

## 5. Optional: headless check

To verify the notebook runs without the browser UI (for CI or a quick sanity check):

```bash
julia -e '
using Pluto
nb = Pluto.SessionActions.open(Pluto.ServerSession(); path="Skyrmions_Pires_classical.jl")
# wait until evaluation finishes in your environment, then:
# println(count(c -> c.errored, nb.cells), " errored cells")
'
```

A full interactive session (`Pluto.run()`) is the intended way to explore plots and sliders.

## Troubleshooting

- **`Pluto` not found** — install it as in step 1, or use the same Julia binary you will use to run the notebook.
- **Packages stuck installing** — wait for the status bar; on a slow network the first open can take several minutes. Restart Pluto if the install fails, then reopen the notebook.
- **Plots blank or slow** — first plot compilation is slow; subsequent cells reuse the compiled backend.
- **Want a static HTML copy** — in Pluto: *Export* → *Static HTML*, or use `Pluto.generate_html` after a headless open.
- **Mathematica comparison** — the notebook does not open `.nb` files; it embeds the published numerical benchmarks from Aristov–Tsypilnikov for comparison only.

## Related files

- [`Skyrmions_Pires_classical.jl`](Skyrmions_Pires_classical.jl) — this Pluto notebook
- [`Skyrmions_Aristov_Tsypilnikov.nb`](Skyrmions_Aristov_Tsypilnikov.nb) — Mathematica reference (classical + spin-wave)
- [`README.md`](README.md) — repository overview and Mathematica notebook notes
