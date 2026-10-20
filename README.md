# Skyrmion crystal calculations

[Skyrmions_Aristov_Tsypilnikov.nb](Skyrmions_Aristov_Tsypilnikov.nb) is the canonical, self-contained Mathematica notebook for D. N. Aristov and A. V. Tsypilnikov, [*Magnon spectrum in two- and three-dimensional skyrmion crystals*](https://arxiv.org/abs/1705.00196v1).

The notebook contains 27 evaluated code cells, definitions, numerical data, plots, and checks. Open it in a modern Mathematica installation, start a fresh kernel, and select **Evaluation → Evaluate Notebook**. It needs no external data, packages, or definitions from the historical notebooks. The full serial run takes tens of minutes; Wolfram Engine can also evaluate its Wolfram Language calculations.

The default calculation uses dimensionless units, no anisotropy, reference field `b = 0.6`, 24-digit working precision for the profile boundary-value problem, 40 radial basis functions, 320 quadrature points, and angular cutoff `|m| ≤ 8`. These settings are explicit and editable in the notebook.

The notebook covers the equations and numerical constructions underlying Figures 1–11:

- Exact spin commutators, direct continuum spin-wave Hessian, classical energy, Euler–Lagrange equation, topology, optimized texture, disk/hexagon geometry, energy-density scans, and optimal radius versus field.
- Bosonic disk spectrum, angular and radial modes, radius dependence, and the signed instability of the `m = −2` mode.
- Finite-cutoff zero-point energy, spin-dependent energy density, conical-spiral comparison, and local quantum spin reduction.
- Two-component boundary matching, tunneling estimates, triangular-lattice bands, and the three-dimensional and thermal extensions.

The implementation corrects the historical anomalous matrix construction, mode normalization, quantum-energy counting, and basis truncation checks. It uses Appendix B's anomalous-coupling sign consistently, documents its difference from Eq. (11), includes the angular normalization factor in local spin reduction, and uses exact disk/hexagon area matching.

Results require the qualifications discussed in the notebook. The continuum quantum energy depends on its ultraviolet cutoff. Applying the printed Section III tunneling prescription literally gives broader bands than Figure 11; this is not a figure-perfect reproduction, and no empirical factor is inserted to force agreement. Equation (22)'s rigid three-dimensional energy shift is distinguished from including longitudinal exchange in the full Bogoliubov problem. The paper does not calculate the spiral's quantum correction or establish a quantum-corrected phase boundary.

The original exploratory notebooks are preserved in [archive/](archive/) for reference. They are not dependencies of the consolidated notebook.

Validated with Wolfram Language 15.0: all 27 code cells evaluated, all 11 numerical checks passed, and the spin-wave Hessian and spin commutators simplified to zero residuals. An independent SciPy solver confirmed the disk energies and literal tunneling estimates.

## Companion Pluto notebook: classical skyrmion model

[Skyrmions_Pires_classical.jl](Skyrmions_Pires_classical.jl) is a Julia [Pluto](https://plutojl.org) notebook that reproduces the classical part of A. S. T. Pires, [*Topological Magnetism in 2D and 3D: Skyrmions and Hopfions. Classical and quantum approaches*](https://arxiv.org/abs/2610.23490), in the same dimensionless units as the Mathematica notebook: the Belavin–Polyakov skyrmion, its Bogomolny bound and stereographic description; the topological charge evaluated on a Cartesian grid for Bloch, Néel, anti- and `m = 2` textures; the Euler–Lagrange profile with Dzyaloshinskii–Moriya interaction and field solved by Newton iteration, the optimal cell radius and the isolated-skyrmion collapse field; the domain-wall ansätze and the easy-axis anisotropy scaling; the helix, helicoid, cone and ferromagnet compared with the skyrmion crystal (critical fields `b₁ ≈ 0.216`, `b_c ≈ 0.801`, `b_hel = π²/16`); and the linearised `K₁`/`I₁` edge tail on a finite disk. The disk energies and optimal radii agree with the Mathematica notebook to `10⁻⁷` and `10⁻⁴` respectively, and the notebook ends with a table of independent pass/fail checks.

**How to run it:** see [README_Pluto.md](README_Pluto.md) (install Pluto, open the notebook, first-run packages, sliders, and troubleshooting). Short version:

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'   # once
julia -e 'using Pluto; Pluto.run(notebook="Skyrmions_Pires_classical.jl")'
```

The embedded package environment installs `Plots`, `PlutoUI`, `QuadGK` and `SpecialFunctions` on first use; full evaluation takes about a minute after precompilation.
