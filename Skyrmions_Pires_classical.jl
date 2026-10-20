### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d4b939fa-906a-4e36-9f80-5457de4804a1
begin
	using LinearAlgebra, Printf, Markdown
	using Plots, PlutoUI, QuadGK, SpecialFunctions
end

# ╔═╡ a720a99a-c873-4990-bdb6-04e21b07dd07
md"""
# Classical skyrmion model: energy, spiral and cone comparison, topological charge

**A. S. T. Pires, _Topological Magnetism in 2D and 3D: Skyrmions and Hopfions. Classical and quantum approaches_, [arXiv:2610.23490](https://arxiv.org/abs/2610.23490)** | companion Pluto notebook

Source: the review's Sections 2–5 (topology, continuum ferromagnet, Belavin–Polyakov skyrmion, Dzyaloshinskii–Moriya interaction), Section 6 (sigma-model solutions and the Bogomolny bound), Section 11 (skyrmion tail and repulsion) and Appendices A–B (exact solution and wall ansatz). This notebook computes its own profiles, energies, phase comparison and figures; it does not load external data. Where the review only quotes a result, the classical skyrmion-crystal calculation of D. N. Aristov and A. V. Tsypilnikov ([arXiv:1705.00196](https://arxiv.org/abs/1705.00196); Mathematica notebook `Skyrmions_Aristov_Tsypilnikov.nb` in this repository) is used as the independent numerical benchmark. Evaluate from the first cell; the complete run takes about a minute after package precompilation.
"""

# ╔═╡ a2643675-8b2f-4ae9-96d5-3584e717cd3d
TableOfContents(title="Contents", depth=3)

# ╔═╡ 2b0aaeaa-caed-4d8f-9a32-0481ffc525cd
"""
    md_table(header, rows; sigdigits=7)

Markdown table from a header vector and an iterable of rows (tuples or vectors); floats are rounded to `sigdigits`.
"""
function md_table(header, rows; sigdigits=7)
	f(x) = x isa AbstractFloat ? string(round(x; sigdigits=sigdigits)) : string(x)
	io = IOBuffer()
	println(io, "| ", join(header, " | "), " |")
	println(io, "|", repeat(" --- |", length(header)))
	for row in rows
		println(io, "| ", join(map(f, collect(row)), " | "), " |")
	end
	Markdown.parse(String(take!(io)))
end

# ╔═╡ 278d6a7f-b131-453a-a863-d9b9959d5217
md"""
## 1. Conventions, parameters and equation checks

Continuum energy of a two-dimensional chiral ferromagnet with bulk (Bloch-type) Dzyaloshinskii–Moriya interaction in a perpendicular field, Pires Eq. (28):

```math
E = \int d^2r\left[\frac{C}{2}(\nabla\mathbf n)^2 + D\,\mathbf n\cdot(\nabla\times\mathbf n) - B\,n_z\right],\qquad C = JS^2,\quad D \to DS^2,\quad B \to BS .
```

Lengths are measured in ``C/D``, energies in ``C``, energy densities in ``D^2/C`` and the field through ``b = BC/D^2`` (for a lattice spin-``S`` model this is ``b = BC/(SD^2)``, the Aristov–Tsypilnikov convention). In these units ``C = D = 1`` and the dimensionless density, relative to the saturated ferromagnet ``\mathbf n = \hat z``, is

```math
\rho[\mathbf n] = \tfrac12(\nabla\mathbf n)^2 + \mathbf n\cdot(\nabla\times\mathbf n) + b\,(1 - n_z) + a\,(1 - n_z^2),
```

with an optional easy-axis anisotropy ``a = AC/D^2`` (Pires Eq. (63); ``a = 0`` everywhere except Section 3.3). The axisymmetric ansatz of Pires Eqs. (31) and (43) is ``\mathbf n = (\cos\Phi\sin\Theta, \sin\Phi\sin\Theta, \cos\Theta)`` with ``\Theta = \Theta(r)`` and ``\Phi = m\varphi + \gamma`` (vorticity ``m``, helicity ``\gamma``). The Bloch skyrmion has ``m = 1`` and ``\gamma = \pm\pi/2``; the sign of ``\gamma`` (equivalently of ``D``) is chosen so that the DMI density is ``+(\Theta' + \sin 2\Theta/2r)`` with ``\Theta`` decreasing from ``\pi`` at the core to ``0`` outside, which makes the DMI energy negative. The topological charge uses the orientation of Pires Eq. (41), ``Q = \frac{1}{4\pi}\int \mathbf n\cdot(\partial_x\mathbf n\times\partial_y\mathbf n)\,d^2r``, for which this skyrmion has ``Q = -m = -1`` (Pires Eq. (44)); reversing the order of the derivatives gives ``+1``.
"""

# ╔═╡ 95103ea8-29d7-4e98-863e-0c40752a4d29
begin
	b_ref  = 0.6       # reference field b = BC/D² (also the Aristov–Tsypilnikov reference value)
	N_grid = 2000      # radial intervals of the finite-difference profile solver
	R_scan = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]   # coarse disk radii bracketing the optimal cell
	R_iso  = 40.0      # disk radius standing in for the infinite plane (isolated skyrmion)
	N_iso  = 8000
	md_table(["Parameter", "Value"], [
		("b (reference field)", b_ref),
		("Radial grid intervals N", N_grid),
		("Bracket grid for the optimal radius", string(R_scan)),
		("Isolated-skyrmion disk radius / intervals", "$R_iso / $N_iso"),
		("Anisotropy a", "0 (except Section 3.3)"),
		("Newton tolerance on the profile", "1e-11")])
end

# ╔═╡ 74b89641-18a6-4c78-8b78-00c3006e2edd
begin
	# Pires Eq. (43): n = (cosΦ sinΘ, sinΦ sinΘ, cosΘ), Φ = mφ + γ
	function nvec(x, y, Θ; m=1, γ=π/2)
		Φ = m*atan(y, x) + γ; t = Θ(hypot(x, y))
		[cos(Φ)*sin(t), sin(Φ)*sin(t), cos(t)]
	end

	# Cartesian gradients by central differences: independent of the polar-coordinate algebra
	function grad_n(x, y, Θ; m=1, γ=π/2, h=1e-5)
		(nvec(x+h, y, Θ; m, γ) - nvec(x-h, y, Θ; m, γ)) / 2h,
		(nvec(x, y+h, Θ; m, γ) - nvec(x, y-h, Θ; m, γ)) / 2h
	end

	# local densities computed from the full vector field
	function densities(x, y, Θ; m=1, γ=π/2)
		n = nvec(x, y, Θ; m, γ); dx, dy = grad_n(x, y, Θ; m, γ)
		(exchange = dot(dx, dx) + dot(dy, dy),                        # (∇n)²
		 bloch    = dot(n, [dy[3], -dx[3], dx[2] - dy[1]]),           # n·(∇×n), Eq. (27)
		 neel     = n[3]*(dx[1] + dy[2]) - n[1]*dx[3] - n[2]*dy[3],   # n_z ∇·n − (n·∇)n_z, Eq. (56)
		 topo     = dot(n, cross(dx, dy)))                            # n·(∂ₓn × ∂ᵧn), Eq. (41)
	end
end

# ╔═╡ d565a9da-bddb-4ab4-9010-41bc74981e1b
let
	# test profile: Belavin–Polyakov Θ = 2 arctan(R/r) (Eq. (39)) with analytic derivatives
	R = 1.3
	Θ(r) = 2atan(R, r); dΘ(r) = -2R/(r^2 + R^2); d2Θ(r) = 4R*r/(r^2 + R^2)^2
	pts = [(0.7, 0.4), (-1.1, 0.9), (0.3, -1.6), (2.2, 1.7)]
	resid(f) = maximum(abs(f(x, y)) for (x, y) in pts)
	checks = [
		("Eq. (33): (∇n)² = Θ'² + m² sin²Θ/r²  (m = 1)",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ).exchange - (dΘ(r)^2 + sin(Θ(r))^2/r^2)))),
		("Eq. (33): (∇n)² = Θ'² + m² sin²Θ/r²  (m = 2)",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ; m=2).exchange - (dΘ(r)^2 + 4sin(Θ(r))^2/r^2)))),
		("Eq. (60): n·(∇×n) = sinγ (Θ' + sin2Θ/2r), Bloch helicity γ = π/2",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ; γ=π/2).bloch - (dΘ(r) + sin(2Θ(r))/(2r))))),
		("Eq. (61): bulk DMI density vanishes for Néel helicity γ = 0",
			resid((x, y) -> densities(x, y, Θ; γ=0.0).bloch)),
		("Eq. (62): n_z∇·n − (n·∇)n_z = cosγ (Θ' + sin2Θ/2r), Néel helicity γ = 0",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ; γ=0.0).neel - (dΘ(r) + sin(2Θ(r))/(2r))))),
		("Eq. (44): n·(∂ₓn × ∂ᵧn) = m Θ' sinΘ / r  (m = 1)",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ).topo - dΘ(r)*sin(Θ(r))/r))),
		("Eq. (44): n·(∂ₓn × ∂ᵧn) = m Θ' sinΘ / r  (m = −1, antiskyrmion)",
			resid((x, y) -> (r = hypot(x, y); densities(x, y, Θ; m=-1).topo + dΘ(r)*sin(Θ(r))/r))),
		("Eq. (45): (∇n)² = (∂ₓn + n×∂ᵧn)² + 2 n·(∂ₓn×∂ᵧn)",
			resid((x, y) -> begin
				n = nvec(x, y, Θ); dx, dy = grad_n(x, y, Θ); v = dx + cross(n, dy)
				dot(dx, dx) + dot(dy, dy) - dot(v, v) - 2dot(n, cross(dx, dy))
			end)),
		("Eq. (38): Belavin–Polyakov profile solves Θ'' + Θ'/r − sinΘ cosΘ/r² = 0",
			maximum(abs(d2Θ(r) + dΘ(r)/r - sin(Θ(r))*cos(Θ(r))/r^2) for r in (0.3, 1.0, 2.5, 7.0))),
	]
	md_table(["Identity (Pires equation), evaluated from the full vector field", "Max residual over test points"], checks; sigdigits=3)
end

# ╔═╡ 68ba4c8c-a9c9-4fd5-8359-1852aaf8e89a
md"""
## 2. Belavin–Polyakov skyrmion, topological charge and the Bogomolny bound

With exchange only, the static equation ``\mathbf n\times\nabla^2\mathbf n = 0`` reduces for ``\Theta(r)``, ``\Phi = m\varphi + \gamma`` to Pires Eq. (38), ``\Theta'' + \Theta'/r - (m^2/r^2)\sin\Theta\cos\Theta = 0``, solved for ``m = 1`` by the Belavin–Polyakov profile ``\Theta = 2\arctan(R/r)`` (Eq. (39), Appendix A), ``n_z = (r^2 - R^2)/(r^2 + R^2)`` (Eq. (40)). The topological charge, Eq. (41), is evaluated here on a Cartesian grid directly from the vector field and compared with the endpoint formula Eq. (44), ``Q = m[\cos\Theta(0) - \cos\Theta(\infty)]/2``. The exchange energy obeys the Bogomolny bound ``E \ge 4\pi C|Q|`` (Eqs. (46)–(48) and (116)), saturated by the Belavin–Polyakov texture for every radius: pure exchange has no length scale.
"""

# ╔═╡ 6d334429-2322-4cdd-b979-f3693a452537
begin
	ΘBP(r; R=1.0) = 2atan(R, r)                      # Pires Eq. (39); atan(R, r) gives π at r = 0
	dΘBP(r; R=1.0) = -2R/(r^2 + R^2)
	nzBP(r; R=1.0) = (r^2 - R^2)/(r^2 + R^2)          # Pires Eq. (40)
end

# ╔═╡ 5dfcea4c-ecaf-4bb3-9c52-f92a1e592e28
let
	rs = range(0, 6, length=301)
	p1 = plot(rs, ΘBP.(rs), lw=2, label="Θ(r) = 2 arctan(R/r)", xlabel="r / R", ylabel="Θ",
	          title="Fig. 9: Belavin–Polyakov profile", framestyle=:box, legend=:topright, titlefontsize=10)
	hline!(p1, [π, π/2], ls=:dash, color=:gray, label="")
	p2 = plot(rs, cos.(ΘBP.(rs)), lw=2, label="cos Θ(r)", xlabel="r / R", ylabel="n_z",
	          title="Eq. (40): n_z = (r² − R²)/(r² + R²)", framestyle=:box, legend=:bottomright, titlefontsize=10)
	scatter!(p2, rs[1:20:end], nzBP.(rs[1:20:end]), ms=3, label="Eq. (40)")
	plot(p1, p2, layout=(1, 2), size=(900, 340), margin=5Plots.mm)
end

# ╔═╡ f0a0ce6e-dc8c-4893-ba57-4fe6be9ceec0
"""
    texture_plot(Θ; m, γ, L, nq, ngrid, title)

Heat map of ``n_z`` with the in-plane components ``(n_x, n_y)`` drawn as arrows, for the ansatz ``\\Phi = m\\varphi + \\gamma``.
"""
function texture_plot(Θ; m=1, γ=π/2, L=2.5, nq=14, ngrid=101, title="")
	xs = range(-L, L, length=ngrid)
	nz = [nvec(x, y, Θ; m, γ)[3] for y in xs, x in xs]
	p = heatmap(xs, xs, nz; c=:RdBu, clims=(-1, 1), aspect_ratio=1, xlims=(-L, L), ylims=(-L, L),
	            xlabel="x", ylabel="y", title=title, colorbar_title="n_z", titlefontsize=9)
	qs = range(-L*(1 - 1/nq), L*(1 - 1/nq), length=nq)
	s = 0.9*step(qs)
	X = [x for y in qs, x in qs]; Y = [y for y in qs, x in qs]
	U = [s*nvec(x, y, Θ; m, γ)[1] for y in qs, x in qs]
	V = [s*nvec(x, y, Θ; m, γ)[2] for y in qs, x in qs]
	quiver!(p, vec(X .- U/2), vec(Y .- V/2); quiver=(vec(U), vec(V)), color=:black, lw=1)
	p
end

# ╔═╡ fd1993a3-cf33-453e-8374-04c1ca9b9e9f
let
	Θ = r -> ΘBP(r; R=1.0)
	panels = [
		texture_plot(Θ; m=1,  γ=π/2,  title="Bloch skyrmion: m = 1, γ = π/2  (Fig. 8, Q = −1)"),
		texture_plot(Θ; m=1,  γ=-π/2, title="Bloch skyrmion: m = 1, γ = −π/2  (Q = −1)"),
		texture_plot(Θ; m=1,  γ=0.0,  title="Néel skyrmion: m = 1, γ = 0  (Q = −1)"),
		texture_plot(Θ; m=1,  γ=π,    title="Néel skyrmion: m = 1, γ = π  (Q = −1)"),
		texture_plot(Θ; m=-1, γ=0.0,  title="antiskyrmion: m = −1  (Q = +1)"),
		texture_plot(Θ; m=2,  γ=0.0,  title="m = 2 texture, Eq. (125)  (Q = −2)"),
	]
	plot(panels..., layout=(2, 3), size=(1350, 850), margin=3Plots.mm)
end

# ╔═╡ 26da870f-0750-4845-af36-eee13bd4bcff
"""
    topological_charge(Θ; m, γ, L, h)

Pires Eq. (41), ``Q = \\frac{1}{4\\pi}\\int \\mathbf n\\cdot(\\partial_x\\mathbf n\\times\\partial_y\\mathbf n)\\,d^2r``, evaluated on the square
``[-L, L]^2`` with spacing `h` by central differences of the full vector field (no use of the polar-coordinate reduction).
"""
function topological_charge(Θ; m=1, γ=π/2, L=20.0, h=0.1)
	xs = range(-L, L, step=h)
	M = length(xs)
	n = [nvec(x, y, Θ; m, γ) for x in xs, y in xs]
	Q = 0.0
	for i in 2:M-1, j in 2:M-1
		dx = (n[i+1, j] - n[i-1, j]) / 2h
		dy = (n[i, j+1] - n[i, j-1]) / 2h
		Q += dot(n[i, j], cross(dx, dy))
	end
	Q * h^2 / 4π
end

# ╔═╡ f1c60c82-6729-44ce-98ca-cc7d65c25452
let
	R = 2.0
	rows = Any[]
	for L in (5.0, 10.0, 20.0, 40.0)
		Q = topological_charge(r -> ΘBP(r; R); L, h=0.1)
		push!(rows, ("m = 1, γ = π/2 (Bloch)", L, Q, (cos(ΘBP(0.0; R)) - cos(ΘBP(L; R)))/2))
	end
	for (m, γ, lab) in ((1, 0.0, "m = 1, γ = 0 (Néel)"), (-1, 0.0, "m = −1 (antiskyrmion)"), (2, 0.0, "m = 2"))
		Q = topological_charge(r -> ΘBP(r; R); m, γ, L=40.0, h=0.1)
		push!(rows, (lab, 40.0, Q, m*(cos(ΘBP(0.0; R)) - cos(ΘBP(40.0; R)))/2))
	end
	md_table(["Texture, R = 2", "Box half-width L", "Q from Eq. (41) on the Cartesian grid", "Eq. (44) with Θ(∞) → Θ(L)"], rows; sigdigits=5)
end

# ╔═╡ 4e7cf273-2d05-46d2-864c-0bc806b5bc20
md"""
The Belavin–Polyakov texture approaches the ferromagnet only algebraically, ``1 - n_z \simeq 2R^2/r^2``, so a finite box of half-width ``L`` misses a charge of order ``(R/L)^2``; the grid values converge to the quantised ``Q = -m`` as ``L`` grows, and the helicity ``\gamma`` plays no role. The texture stabilised by the DMI and the field (Section 3) decays exponentially, and there the Cartesian integral returns ``-1`` to the discretisation error.
"""

# ╔═╡ 63314e78-de74-4716-9385-0c41cbee06a4
let
	R = 1.0
	bz(r) = dΘBP(r; R)*sin(ΘBP(r; R))/r                    # Eq. (42): n·(∂ₓn × ∂ᵧn) = Θ' sinΘ / r = −4R²/(r² + R²)²
	rs = range(0.01, 5, length=400)
	Qcum = [quadgk(r -> 2π*r*bz(r), 0, rr)[1]/4π for rr in rs]
	p1 = plot(rs, bz.(rs), lw=2, label="Eq. (42): b_z(r) = Θ' sinΘ / r", xlabel="r / R", ylabel="b_z",
	          title="Fictitious field b_z = n·(∂ₓn × ∂ᵧn)", framestyle=:box, titlefontsize=10)
	plot!(p1, rs, -4R^2 ./ (rs.^2 .+ R^2).^2, ls=:dash, lw=2, label="−4R²/(r² + R²)²")
	p2 = plot(rs, Qcum, lw=2, label="Q(r) = (1/4π) ∫₀ʳ b_z 2πr' dr'", xlabel="r / R", ylabel="Q(r)",
	          title="Cumulative topological charge → −1", framestyle=:box, legend=:topright, titlefontsize=10)
	hline!(p2, [-1], ls=:dash, color=:gray, label="")
	plot(p1, p2, layout=(1, 2), size=(900, 330), margin=5Plots.mm)
end

# ╔═╡ 59d8bd9a-1dad-4777-8366-d8bb3b245045
let
	# Bogomolny bound: E = ∫ d²r ½(∇n)² = π ∫ r (Θ'² + sin²Θ/r²) dr equals 4π|Q| for every R (scale invariance)
	Eex(R) = π*quadgk(r -> r*(dΘBP(r; R)^2 + sin(ΘBP(r; R))^2/r^2), 0, Inf; rtol=1e-10)[1]
	md_table(["R", "E_exchange / C, Eqs. (33), (46)", "Bogomolny bound 4π·abs(Q), Q = −1", "residual"], [(R, Eex(R), 4π, Eex(R) - 4π) for R in (0.5, 1.0, 3.0)]; sigdigits=9)
end

# ╔═╡ 36d75ef1-71d4-4916-afe7-fd4e5b43c99a
let
	# Pires Eqs. (98)–(122): stereographic coordinate w = tan(Θ/2) e^{iΦ}. For Θ(0) = π the m = 1 skyrmion is
	# w = R e^{iγ}/z̄ (antiholomorphic: ∂_z w = 0, Q = −1); m = −1 gives w = R e^{iγ}/z (holomorphic, Q = +1), Eqs. (117), (132).
	R = 1.0; γ = π/2
	w(x, y; m) = (n = nvec(x, y, r -> ΘBP(r; R); m, γ); (n[1] + im*n[2])/(1 + n[3]))
	h = 1e-5
	dz(x, y; m)  = ((w(x+h, y; m) - w(x-h, y; m))/2h - im*(w(x, y+h; m) - w(x, y-h; m))/2h)/2   # ∂_z = (∂ₓ − i∂ᵧ)/2
	dzb(x, y; m) = ((w(x+h, y; m) - w(x-h, y; m))/2h + im*(w(x, y+h; m) - w(x, y-h; m))/2h)/2   # ∂_z̄ = (∂ₓ + i∂ᵧ)/2
	pts = [(0.7, 0.4), (-1.1, 0.9), (0.3, -1.6)]
	rows = [
		("m = 1:  abs(w − R e^{iγ}/z̄)", maximum(abs(w(x, y; m=1) - R*cis(γ)/conj(complex(x, y))) for (x, y) in pts)),
		("m = 1:  abs(∂_z w)  (antiholomorphic ⇒ 0)", maximum(abs(dz(x, y; m=1)) for (x, y) in pts)),
		("m = 1:  abs(∂_z̄ w)", maximum(abs(dzb(x, y; m=1)) for (x, y) in pts)),
		("m = −1: abs(w − R e^{iγ}/z)", maximum(abs(w(x, y; m=-1) - R*cis(γ)/complex(x, y)) for (x, y) in pts)),
		("m = −1: abs(∂_z̄ w)  (holomorphic ⇒ 0)", maximum(abs(dzb(x, y; m=-1)) for (x, y) in pts)),
		("m = −1: abs(∂_z w)", maximum(abs(dz(x, y; m=-1)) for (x, y) in pts)),
	]
	md_table(["Stereographic field w = tan(Θ/2) e^{iΦ}, Eqs. (121)–(122), (132); max over test points", "value / residual"], rows; sigdigits=3)
end

# ╔═╡ e87d0f6b-9e17-4d9b-a20e-31d13f222a09
md"""
## 3. Skyrmion with Dzyaloshinskii–Moriya interaction and field: profile and energy

Pires Eq. (64) with the helicity fixed as above, the Zeeman term and the optional anisotropy gives the radial energy density (energy per area in units ``D^2/C``, relative to the ferromagnet)

```math
\rho(r) = \tfrac12\beta'^2 + \frac{\sin^2\beta}{2r^2} + \beta' + \frac{\sin 2\beta}{2r} + b\,(1-\cos\beta) + a\sin^2\beta,
```

which for ``a = 0`` is Eq. (8) of Aristov–Tsypilnikov, and the Euler–Lagrange equation (Pires Eqs. (66)–(67) plus the field term; Aristov–Tsypilnikov Eq. (9))

```math
\beta'' + \frac{\beta'}{r} - \frac{\sin\beta\cos\beta}{r^2} + \frac{2\sin^2\beta}{r} - b\sin\beta - a\sin 2\beta = 0,\qquad \beta(0) = \pi,\quad \beta(R) = 0 .
```

As in the skyrmion-crystal calculation, the texture lives on a disk of radius ``R`` whose area equals that of the hexagonal cell; the spins at the cell edge point along the field. The disk energy density ``\rho_C(R) = E/\pi R^2`` is then minimised over ``R``. The nonlinear boundary-value problem is solved by damped Newton iteration on a second-order finite-difference grid (tridiagonal Jacobian, several 360°-wall initial guesses, monotonicity check); the discretisation error is ``O(h^2)`` and is checked below against the 24-digit shooting solution of the Mathematica notebook.
"""

# ╔═╡ a5ca2db0-455a-48a8-8a4e-8996118f3471
begin
	# residual of the Euler–Lagrange equation at the interior grid points
	function el_residual!(F, β, r, h, b, a)
		N = length(r) - 1
		@inbounds for i in 1:N-1
			ri = r[i+1]; bi = β[i+1]; bm = β[i]; bp = β[i+2]
			F[i] = (bp - 2bi + bm)/h^2 + (bp - bm)/(2h*ri) - sin(bi)*cos(bi)/ri^2 + 2sin(bi)^2/ri - b*sin(bi) - a*sin(2bi)
		end
		F
	end

	# 360° wall of radius Rw and width w (Pires Appendix B, Eq. (B.19)) used as initial guess
	wall_guess(r, Rw, w) = [ri == 0 ? float(π) : 2atan(sinh(Rw/w)/sinh(ri/w)) for ri in r]

	# damped Newton iteration with a tridiagonal Jacobian
	function newton_profile(b, R, β0; a=0.0, maxiter=200, tol=1e-11)
		N = length(β0) - 1; h = R/N; r = range(0, R, length=N+1)
		β = copy(β0); β[1] = π; β[end] = 0.0
		n = N - 1
		F = zeros(n); Ft = zeros(n); dl = zeros(n-1); d = zeros(n); du = zeros(n-1); βt = similar(β)
		el_residual!(F, β, r, h, b, a)
		for it in 1:maxiter
			@inbounds for i in 1:n
				ri = r[i+1]; bi = β[i+1]
				d[i] = -2/h^2 - cos(2bi)/ri^2 + 2sin(2bi)/ri - b*cos(bi) - 2a*cos(2bi)
				i < n && (du[i] = 1/h^2 + 1/(2h*ri))
				i > 1 && (dl[i-1] = 1/h^2 - 1/(2h*ri))
			end
			δ = Tridiagonal(dl, d, du) \ F
			nF = norm(F); λ = 1.0; accepted = false
			while λ ≥ 1/1024
				βt .= β; βt[2:N] .-= λ .* δ
				el_residual!(Ft, βt, r, h, b, a)
				if norm(Ft) < (1 - 1e-4λ)*nF
					β .= βt; F .= Ft; accepted = true; break
				end
				λ /= 2
			end
			accepted || return β, false
			maximum(abs, λ .* δ) < tol && return β, true
		end
		β, false
	end

	"""
	    solve_profile(b, R; a=0, N=N_grid) -> (r, β, converged)

	Skyrmion profile β(r) on the disk 0 ≤ r ≤ R with β(0) = π, β(R) = 0, from several wall initial guesses;
	the returned solution is monotone and lies in [0, π].
	"""
	function solve_profile(b, R; a=0.0, N=N_grid)
		r = range(0, R, length=N+1)
		for (Rw, w) in ((min(R/2, 2.0), 1.0), (R/2, 1.0), (min(R/2, 3.0), 0.7), (min(R/3, 1.5), 1.0), (0.5, 0.5), (0.25, 0.25))
			β, ok = newton_profile(b, R, wall_guess(r, Rw, w); a)
			if ok && all(-1e-9 .<= β .<= π + 1e-9) && all(diff(β) .<= 1e-7)
				return collect(r), β, true
			end
		end
		collect(r), wall_guess(r, R/2, 1.0), false
	end
end

# ╔═╡ 4a116270-e675-48f4-987e-a3dfd8cdf170
begin
	"""
	    energy_density(r, β, b; a=0) -> (ρ, E)

	Disk energy density ρ = E/πR² (units D²/C) and energy E (units C) of a grid profile, by the midpoint rule.
	"""
	function energy_density(r, β, b; a=0.0)
		N = length(r) - 1; h = r[2] - r[1]; R = r[end]
		E = 0.0
		@inbounds for i in 1:N
			rm = (r[i] + r[i+1])/2; βm = (β[i] + β[i+1])/2; dβ = (β[i+1] - β[i])/h
			E += rm*(0.5dβ^2 + sin(βm)^2/(2rm^2) + dβ + sin(2βm)/(2rm) + b*(1 - cos(βm)) + a*sin(βm)^2)*h
		end
		E *= 2π
		E/(π*R^2), E
	end

	# golden-section minimisation on [lo, hi]
	function golden(f, lo, hi; tol=1e-6)
		φ = (√5 - 1)/2
		x1 = hi - φ*(hi - lo); x2 = lo + φ*(hi - lo); f1 = f(x1); f2 = f(x2)
		while hi - lo > tol
			if f1 < f2
				hi = x2; x2 = x1; f2 = f1; x1 = hi - φ*(hi - lo); f1 = f(x1)
			else
				lo = x1; x1 = x2; f1 = f2; x2 = lo + φ*(hi - lo); f2 = f(x2)
			end
		end
		(lo + hi)/2
	end

	function disk_density(b, R; a=0.0, N=N_grid)
		r, β, _ = solve_profile(b, R; a, N)
		energy_density(r, β, b; a)[1]
	end

	"""
	    optimal_radius(b; a=0, N=N_grid)

	Coarse scan over `R_scan` followed by a golden-section search of the disk radius minimising ρ_C(R).
	"""
	function optimal_radius(b; a=0.0, N=N_grid)
		vals = [disk_density(b, R; a, N) for R in R_scan]
		i = argmin(vals)
		(i == 1 || i == length(R_scan)) && @warn "optimal radius not bracketed by R_scan" b
		R0 = golden(R -> disk_density(b, R; a, N), R_scan[max(i-1, 1)], R_scan[min(i+1, end)]; tol=1e-5)
		r, β, ok = solve_profile(b, R0; a, N)
		ρ, E = energy_density(r, β, b; a)
		(b=b, R0=R0, ρ=ρ, E=E, r=r, β=β, converged=ok)
	end

	# piecewise-linear Θ(x) from a grid profile, Θ = 0 outside the disk
	function profile_interp(r, β)
		R = r[end]
		x -> x ≥ R ? 0.0 : begin
			i = clamp(searchsortedlast(r, x), 1, length(r) - 1)
			t = (x - r[i])/(r[i+1] - r[i])
			(1 - t)*β[i] + t*β[i+1]
		end
	end
end

# ╔═╡ ee899d84-a831-4698-92a3-0a7cb88a581c
begin
	ref = optimal_radius(b_ref)
	ref_res = maximum(abs, el_residual!(zeros(length(ref.r) - 2), ref.β, ref.r, ref.r[2] - ref.r[1], b_ref, 0.0))
	ref_Q = topological_charge(profile_interp(ref.r, ref.β); L=ref.R0, h=ref.R0/400)
	md_table(["Reference disk, b = $b_ref", "This notebook", "Mathematica notebook (24-digit shooting)", "Difference"], [
		("Optimal radius R₀", ref.R0, 4.018913753672924, ref.R0 - 4.018913753672924),
		("Classical density ρ_C", ref.ρ, -0.0691244243113868, ref.ρ + 0.0691244243113868),
		("Classical disk energy E / C", ref.E, -3.507508827096083, ref.E + 3.507508827096083),
		("Max Euler–Lagrange residual on the grid", ref_res, "", ""),
		("Newton converged, monotone profile", ref.converged, "", ""),
		("Q from Eq. (41) on the disk, Cartesian grid", ref_Q, -1, ref_Q + 1),
		("Q from Eq. (44), [cos β(0) − cos β(R)]/2, m = 1", (cos(ref.β[1]) - cos(ref.β[end]))/2, -1, 0.0),
	])
end

# ╔═╡ c0f6824b-63a0-4260-baff-f49d0b3d575e
let
	r, β, R = ref.r, ref.β, ref.R0
	p1 = plot(r, β, lw=2, label="β(r)", xlabel="r", ylabel="β, n_z", framestyle=:box, titlefontsize=10,
	          title="Profile of the optimal disk, b = $b_ref, R₀ = $(round(R, digits=3))")
	plot!(p1, r, cos.(β), lw=2, ls=:dash, label="n_z = cos β")
	hline!(p1, [π/2, 0], color=:gray, ls=:dot, label="")
	p2 = texture_plot(profile_interp(r, β); m=1, γ=π/2, L=R, nq=16, title="Optimised Bloch skyrmion (Aristov–Tsypilnikov Fig. 1)")
	plot(p1, p2, layout=(1, 2), size=(1000, 420), margin=5Plots.mm)
end

# ╔═╡ e5078a52-d62d-475b-97fc-c999a987d023
md"""
Explore the disk solution — field ``b`` = $(@bind b_live Slider(0.05:0.05:0.95; default=0.6, show_value=true)) ⠀⠀⠀ disk radius ``R`` = $(@bind R_live Slider(1.5:0.25:12.0; default=4.0, show_value=true))
"""

# ╔═╡ 8039c401-aa49-4aa0-8155-3d09910bc9bf
let
	r, β, ok = solve_profile(b_live, R_live)
	ρ, E = energy_density(r, β, b_live)
	p = plot(r, β, lw=2, label="β(r)", xlabel="r", ylabel="β, n_z", framestyle=:box, titlefontsize=10, size=(700, 360),
	         title=@sprintf("b = %.2f, R = %.2f:  ρ_C = %.5f,  E/C = %.4f,  converged = %s", b_live, R_live, ρ, E, ok))
	plot!(p, r, cos.(β), lw=2, ls=:dash, label="n_z")
	p
end

# ╔═╡ e84aa37a-b4db-41f9-b864-ce8912d40ce2
let
	Rs = 1.5:0.25:8.0
	bench = Dict(
		0.4 => [(3.0, -0.13876759066624242), (4.0, -0.1742110368227792), (5.0, -0.15467122715324266), (6.0, -0.12569808023791434), (8.0, -0.07820466505855134)],
		0.6 => [(3.0, -0.03232734300856554), (4.0, -0.06911801658124107), (5.0, -0.05913194693977164), (6.0, -0.04454776088651344), (8.0, -0.025682808203627033)],
		0.8 => [(3.0, 0.055747892818157695), (4.0, 0.007592541054184787), (5.0, 0.000686269812105484), (6.0, -0.0001000207245240506), (8.0, -0.00012163743017715635)])
	p = plot(xlabel="R", ylabel="ρ_C", title="Fig. 3 (Aristov–Tsypilnikov): classical disk energy density", framestyle=:box,
	         ylims=(-0.22, 0.25), legend=:bottomright, titlefontsize=10, size=(760, 440))
	for (k, b) in enumerate((0.4, 0.6, 0.8))
		plot!(p, Rs, [disk_density(b, R) for R in Rs], lw=2, color=k, label="b = $b")
		scatter!(p, first.(bench[b]), last.(bench[b]), color=k, ms=5, label=(k == 1 ? "Mathematica notebook" : ""))
	end
	hline!(p, [0], color=:gray, ls=:dash, label="")
	p
end

# ╔═╡ a0bcd687-3ffa-4d47-8aac-ca84516a8d7c
begin
	bench_fields = [0.2, 0.35, 0.5, 0.6, 0.7, 0.76, 0.79, 0.8]
	bench_R0 = [4.005024998740641, 3.9036188162202423, 3.9160412962808095, 4.018913753672924, 4.2959148585337035, 4.751402579129283, 5.4841792410360455, 6.751295615818922]
	bench_ρ  = [-0.3075021916927812, -0.20522941293549196, -0.11818145196399568, -0.0691244243113868, -0.028417274566331495, -0.009199827455421224, -0.001892875257177382, -0.00014885036152058236]
	optima = [optimal_radius(b) for b in bench_fields]
	skx_fields = collect(0.05:0.05:0.8)
	skx = [optimal_radius(b) for b in skx_fields]
end;

# ╔═╡ cb022040-e601-4744-bb60-524fab3550f8
let
	p = plot(skx_fields, [o.R0 for o in skx], lw=2, marker=:circle, ms=3, label="this notebook", xlabel="b", ylabel="R₀",
	         title="Fig. 4 (Aristov–Tsypilnikov): optimal disk radius", framestyle=:box, legend=:topleft, titlefontsize=10, size=(700, 380))
	scatter!(p, bench_fields, bench_R0, marker=:x, ms=7, color=:red, label="Mathematica notebook")
	p
end

# ╔═╡ 9fc4534b-2d68-4d4d-aae7-ce59fa3141c8
md_table(["b", "R₀", "R₀ (Mathematica)", "ρ_C", "ρ_C (Mathematica)", "ρ_C difference"],
	[(b, o.R0, R0m, o.ρ, ρm, o.ρ - ρm) for (b, o, R0m, ρm) in zip(bench_fields, optima, bench_R0, bench_ρ)])

# ╔═╡ baf38015-0a10-4ba0-8eb1-80507d8ce01d
md"""
### 3.1 Isolated skyrmion in the plane and the field at which it costs nothing

Sending the disk radius far beyond the exponential tail (Section 5) turns the disk solution into an isolated skyrmion in the saturated ferromagnet, the setting of Pires' Sections 4–5. Its energy ``E_{\rm iso}(b)`` relative to the ferromagnet, in units of ``C``, is positive at strong field (a metastable particle) and crosses zero at ``b_c``; below ``b_c`` skyrmions proliferate and, because they repel (Section 5), condense into the skyrmion crystal. ``b_c`` therefore coincides with the field at which the optimal-cell density ``\rho_C`` of Fig. 4 reaches zero and ``R_0`` diverges.
"""

# ╔═╡ 3220c423-3efb-4068-8d81-ec77a0e0aab5
begin
	function isolated_skyrmion(b; a=0.0, R=R_iso, N=N_iso)
		r, β, ok = solve_profile(b, R; a, N)
		ok || @warn "isolated skyrmion: Newton did not converge" b
		ρ, E = energy_density(r, β, b; a)
		i = findfirst(<(π/2), β)                         # core radius r_s: β(r_s) = π/2
		r_s = r[i-1] + (β[i-1] - π/2)/(β[i-1] - β[i])*(r[i] - r[i-1])
		(b=b, E=E, r_s=r_s, r=r, β=β, converged=ok)
	end
	iso_fields = collect(0.5:0.05:1.0)
	iso = [isolated_skyrmion(b) for b in iso_fields]
	# bisection for E_iso(b_c) = 0
	b_c = let lo = 0.7, hi = 0.9
		for _ in 1:30
			mid = (lo + hi)/2
			isolated_skyrmion(mid).E > 0 ? (hi = mid) : (lo = mid)
		end
		(lo + hi)/2
	end
end;

# ╔═╡ 23985e1d-e274-4571-93b2-e26034aafe73
let
	p1 = plot(iso_fields, [s.E for s in iso], lw=2, marker=:circle, label="E_iso / C", xlabel="b", ylabel="E_iso / C",
	          title=@sprintf("Isolated skyrmion energy; zero at b_c = %.4f", b_c), framestyle=:box, titlefontsize=10, legend=:topleft)
	hline!(p1, [0], color=:gray, ls=:dash, label="")
	vline!(p1, [b_c], color=:red, ls=:dot, label="b_c")
	p2 = plot(iso_fields, [s.r_s for s in iso], lw=2, marker=:circle, label="r_s: β(r_s) = π/2", xlabel="b", ylabel="core radius r_s",
	          title="Isolated skyrmion core radius", framestyle=:box, titlefontsize=10)
	plot(p1, p2, layout=(1, 2), size=(900, 340), margin=5Plots.mm)
end

# ╔═╡ 78e1b275-e3de-4316-9a38-df38063de737
md"""
### 3.2 Domain-wall ansatz (Pires Eq. (70), Appendix B)

Pires approximates the skyrmion by a circular 360° wall, ``\Theta = 2\arctan\exp[(R-r)/\Delta]`` (Eq. (70)), and notes in Appendix B that the substitution ``x - x_0 \to r - R`` does not give the right behaviour at the origin; the hyperbolic form ``\Theta = 2\arctan[\sinh(R/w)/\sinh(r/w)]`` (Eq. (B.19)) has ``\Theta(0) = \pi``, ``\Theta(R) = \pi/2`` and ``\Theta(\infty) = 0``. The energy functional of Eq. (B.19) is evaluated by adaptive quadrature and minimised over ``(R, w)``, and the result is compared with the exact isolated-skyrmion profile at ``b = b_{\rm ref}``. Eq. (70) cannot be treated the same way: because ``\Theta(0) \ne \pi`` the term ``\sin^2\Theta/2r^2`` makes its exchange integral logarithmically divergent at the origin, so its energy is reported at the Eq. (B.19) parameters for two origin cutoffs to show the drift — which is the quantitative form of Pires' remark.
"""

# ╔═╡ 39234454-4051-47c0-8566-6d2deb613c95
let
	b = b_ref
	Θwall(r, R, Δ) = 2atan(exp((R - r)/Δ))                                     # Pires Eq. (70)
	dΘwall(r, R, Δ) = -sin(Θwall(r, R, Δ))/Δ
	Θsinh(r, R, w) = r == 0 ? float(π) : 2atan(sinh(R/w)/sinh(r/w))            # Pires Eq. (B.19)
	dΘsinh(r, R, w) = r == 0 ? 0.0 : -2sinh(R/w)*cosh(r/w)/w/(sinh(r/w)^2 + sinh(R/w)^2)
	function E_of(Θ, dΘ; rmin=0.0, rmax=R_iso)
		f(r) = r*(0.5dΘ(r)^2 + sin(Θ(r))^2/(2r^2) + dΘ(r) + sin(2Θ(r))/(2r) + b*(1 - cos(Θ(r))))
		2π*quadgk(f, rmin, rmax; rtol=1e-9)[1]
	end
	Esinh(R, w) = E_of(r -> Θsinh(r, R, w), r -> dΘsinh(r, R, w))
	Ewall(R, Δ; rmin=1e-6) = E_of(r -> Θwall(r, R, Δ), r -> dΘwall(r, R, Δ); rmin)
	function minimise2(f)                                                       # nested golden-section search
		Rb = golden(R -> f(R, golden(w -> f(R, w), 0.2, 3.0; tol=1e-4)), 0.3, 8.0; tol=1e-4)
		wb = golden(w -> f(Rb, w), 0.2, 3.0; tol=1e-5)
		Rb, wb, f(Rb, wb)
	end
	Rs, ws, Es = minimise2(Esinh)
	exact = isolated_skyrmion(b)
	tbl = md_table(["Isolated skyrmion at b = $b", "R (core radius)", "w or Δ", "E / C", "E − E_exact"], [
		("Exact profile, Section 3.1", exact.r_s, "", exact.E, 0.0),
		("Eq. (B.19) sinh ansatz, variational minimum", Rs, ws, Es, Es - exact.E),
		("Eq. (70) wall ansatz with the same R, Δ = w; origin cutoff 1e-3", Rs, ws, Ewall(Rs, ws; rmin=1e-3), Ewall(Rs, ws; rmin=1e-3) - exact.E),
		("Eq. (70) wall ansatz with the same R, Δ = w; origin cutoff 1e-6", Rs, ws, Ewall(Rs, ws), Ewall(Rs, ws) - exact.E),
	])
	rr = range(0, 10, length=400)
	p = plot(exact.r, exact.β, lw=2.5, label="exact", xlabel="r", ylabel="Θ", xlims=(0, 10), framestyle=:box, titlefontsize=10,
	         title="Wall ansätze versus the exact isolated-skyrmion profile, b = $b", size=(700, 380))
	plot!(p, rr, Θsinh.(rr, Rs, ws), lw=2, ls=:dash, label="Eq. (B.19): R = $(round(Rs, digits=3)), w = $(round(ws, digits=3))")
	plot!(p, rr, Θwall.(rr, Rs, ws), lw=2, ls=:dot, label="Eq. (70) with the same R, Δ = w")
	md"""
	$(p)
	$(tbl)
	"""
end

# ╔═╡ e96b437e-2fee-4b2e-baa7-f3269b149dda
md"""
### 3.3 Easy-axis anisotropy and the scaling of Pires Eqs. (71)–(74)

With ``b = 0`` and an easy-axis anisotropy ``a = AC/D^2 > 0`` (Pires Eq. (63)), the wall picture predicts a wall width ``\Delta^* = \sqrt{J/A} \to 1/\sqrt{2a}`` (Eq. (72), with ``J \to C/2``) and, away from the threshold ``D_{\rm crit} \sim \sqrt{JA}``, a radius ``R^* \propto D/A \propto 1/a`` (Eq. (74)). In the present units the threshold is the field-free helicoid instability of the ferromagnet: a 180° wall costs ``2\sqrt{2a}`` and gains ``\pi`` from the DMI, so the saturated state is the ground state only for ``a > a_c = \pi^2/8 \approx 1.234``, where the isolated skyrmion is a metastable particle. Its exact core radius ``R^\star`` (where ``\beta = \pi/2``) and local wall width ``\Delta^\star = 1/|\beta'(R^\star)|`` are tabulated below: ``\Delta^\star\sqrt{2a} \to 1`` and ``R^\star/\Delta^\star \gg 1`` in the thin-wall regime near ``a_c``. At strong anisotropy the skyrmion is a small core, ``R^\star \lesssim \Delta^\star``, whose Belavin–Polyakov-like algebraic tail is cut off only at ``\Delta^\star``; both the DMI gain and the anisotropy cost then acquire a factor ``\ln(\Delta^\star/R^\star)``, so ``a R^\star`` keeps decreasing slowly instead of settling at a constant: the ``R^\star \propto D/A`` law of Eq. (74) holds up to logarithmic corrections.
"""

# ╔═╡ 34a20422-f071-46cd-8317-0b261a024de2
let
	rows = Any[]
	for a in (1.3, 1.5, 2.0, 3.0, 5.0, 8.0)
		r, β, ok = solve_profile(0.0, 20.0; a, N=8000)
		ρ, E = energy_density(r, β, 0.0; a)
		i = findfirst(<(π/2), β)
		Rs = r[i-1] + (β[i-1] - π/2)/(β[i-1] - β[i])*(r[i] - r[i-1])
		Δ = (r[i+1] - r[i-1])/abs(β[i+1] - β[i-1])
		push!(rows, (a, ok, E, Rs, a*Rs, Δ, Δ*sqrt(2a), Rs/Δ))
	end
	md_table(["a", "converged", "E_iso / C", "R⋆ (β = π/2)", "a·R⋆", "Δ⋆ = 1/abs(β'(R⋆))", "Δ⋆·√(2a)", "R⋆/Δ⋆"], rows; sigdigits=4)
end

# ╔═╡ 1acd803c-d165-4ef8-8fe1-8e1c23d9ff8c
md"""
## 4. Comparison with the spiral, the helicoid and the cone

The competing one-dimensional states of the same functional (Pires Sections 3 and 5: "helical, conical, skyrmion-lattice and ferromagnetic states") are evaluated in the same units, relative to the ferromagnet ``\rho_{\rm FM} = 0``.

* **Helix (spiral)**: ``\mathbf n = (0, \sin\theta, \cos\theta)``, ``\theta = -x``, wave vector ``q = D/C = 1``, giving ``\rho_{\rm helix} = -\tfrac12 + b``. The rotation plane contains the field, so the Zeeman energy averages to zero.
* **Helicoid** (field-distorted helix, chiral soliton lattice): the same geometry with ``\theta(x)`` obeying ``\theta'' = b\sin\theta``. With the first integral ``\tfrac12\theta'^2 = 2b\sin^2(\theta/2) + c`` the period ``L(c)`` and the energy per period are elliptic integrals, and ``\rho_{\rm hel} = \min_c\{[\int_0^{2\pi}|\theta'|\,d\theta - 2\pi]/L(c) - c\}`` is minimised numerically. The period diverges and ``\rho_{\rm hel}\to 0^-`` at ``b_{\rm hel} = \pi^2/16 \approx 0.617``, where a single 2π kink costs ``8\sqrt b - 2\pi = 0``.
* **Cone** (conical spiral with the axis along the field; Aristov–Tsypilnikov Appendix C): ``\mathbf n = (\sin\beta\cos qx, \sin\beta\sin qx, \cos\beta)`` with the field along the cone axis gives ``\rho = \tfrac12 q^2\sin^2\beta - q\sin^2\beta + b(1-\cos\beta)``, minimised by ``q = 1`` and ``\cos\beta = b``: ``\rho_{\rm cone} = -\tfrac12(1-b)^2`` for ``b < 1`` and ``0`` above saturation. The cone propagates along the field, so for a film in a perpendicular field it is a three-dimensional (bulk) reference state: the state the skyrmion crystal has to beat in bulk chiral magnets.
* **Skyrmion crystal**: the optimal-disk density ``\rho_C(b)`` of Section 3.
"""

# ╔═╡ b36b11f0-3729-437a-8113-f7490a689648
begin
	ρ_helix(b) = -0.5 + b
	ρ_cone(b)  = b < 1 ? -(1 - b)^2/2 : 0.0
	b_hel = π^2/16

	# helicoid: density and period for a given first-integral constant c, then minimised over c
	function helicoid_density(b, c)
		f(θ) = sqrt(4b*sin(θ/2)^2 + 2c)
		I = quadgk(f, 0, 2π; rtol=1e-12)[1]
		L = quadgk(θ -> 1/f(θ), 0, 2π; rtol=1e-12)[1]
		(I - 2π)/L - c, L
	end
	function helicoid(b)
		b ≥ b_hel && return (ρ=0.0, L=Inf, c=0.0)
		u = golden(u -> helicoid_density(b, 10.0^u)[1], -12.0, 2.0; tol=1e-7)
		ρ, L = helicoid_density(b, 10.0^u)
		(ρ=ρ, L=L, c=10.0^u)
	end
end

# ╔═╡ 3cf1d5aa-a811-41ba-9408-9d464c93ed0b
let
	ρc(q, β, b) = 0.5q^2*sin(β)^2 - q*sin(β)^2 + b*(1 - cos(β))
	rows = Any[]
	for b in (0.3, 0.6, 0.9)
		gridmin = minimum(ρc(q, β, b) for q in range(0, 2, length=401), β in range(0, π, length=721))
		push!(rows, ("Cone: brute-force minimum over (q, β) minus −(1−b)²/2 at b = $b", gridmin - ρ_cone(b)))
	end
	push!(rows, ("Helicoid at b = 0: ρ + 1/2 (undistorted helix)", helicoid(0.0).ρ + 0.5))
	push!(rows, ("Helicoid at b = 0: period − 2π", helicoid(0.0).L - 2π))
	push!(rows, ("Helicoid period at b = 0.6 (diverges at b_hel = π²/16 = $(round(b_hel, digits=5)))", helicoid(0.6).L))
	push!(rows, ("Helicoid density at b = 0.61 (→ 0⁻ at b_hel)", helicoid(0.61).ρ))
	md_table(["Check", "value / residual"], rows; sigdigits=4)
end

# ╔═╡ 3699a233-39e9-4d65-ac70-9eb77df79db9
let
	bs = range(0.0, 1.0, length=201)
	p = plot(xlabel="b", ylabel="energy density (D²/C, relative to the ferromagnet)", framestyle=:box, legend=:bottomright,
	         title="Fig. 9 (Aristov–Tsypilnikov): classical energies of the competing states", titlefontsize=10, size=(820, 480))
	plot!(p, bs, ρ_helix.(bs), lw=2, label="helix (undistorted): −1/2 + b")
	plot!(p, bs, [helicoid(b).ρ for b in bs], lw=2, label="helicoid (field-distorted helix)")
	plot!(p, bs, ρ_cone.(bs), lw=2, label="cone (3D reference): −(1 − b)²/2")
	hline!(p, [0], lw=2, color=:black, label="ferromagnet")
	plot!(p, skx_fields, [o.ρ for o in skx], lw=2.5, marker=:circle, ms=3, color=:red, label="skyrmion crystal: optimal disk ρ_C")
	vline!(p, [b_hel], ls=:dot, color=:gray, label="b_hel = π²/16")
	ylims!(p, -0.52, 0.12)
	p
end

# ╔═╡ 5217da5a-2cb8-46bd-b613-10798b37feba
begin
	# crossing of the helicoid and skyrmion-crystal densities (film, perpendicular field)
	b_1 = let lo = 0.1, hi = 0.5
		for _ in 1:25
			mid = (lo + hi)/2
			optimal_radius(mid).ρ - helicoid(mid).ρ < 0 ? (hi = mid) : (lo = mid)
		end
		(lo + hi)/2
	end
	cone_margin = maximum(ρ_cone(b) - o.ρ for (b, o) in zip(skx_fields, skx))
	md_table(["b", "R₀", "ρ_C (SkX disk)", "ρ_helicoid", "ρ_cone", "ρ_C − ρ_cone", "lowest 2D state"],
		[(b, o.R0, o.ρ, helicoid(b).ρ, ρ_cone(b), o.ρ - ρ_cone(b),
		  o.ρ < min(helicoid(b).ρ, 0) ? "skyrmion crystal" : (helicoid(b).ρ < 0 ? "helicoid" : "ferromagnet"))
		 for (b, o) in zip(bench_fields, optima)])
end

# ╔═╡ 82f43975-5c2e-4420-bbae-b3c8546d6eb9
md_table(["Critical field (film, perpendicular field; units D²/C)", "value", "origin"], [
	("b₁: helicoid → skyrmion crystal", b_1, "bisection of ρ_C(b) − ρ_hel(b)"),
	("b_c: skyrmion crystal → ferromagnet", b_c, "zero of the isolated-skyrmion energy, Section 3.1"),
	("b_hel: metastability limit of the helicoid", b_hel, "π²/16, single-kink energy 8√b − 2π = 0"),
	("max over b of ρ_cone − ρ_C", cone_margin, "negative: the bulk cone is below the skyrmion crystal at every field"),
])

# ╔═╡ d832c3cd-9d0d-4759-b625-30c502412e42
md"""
**Classical phase sequence of the film (field perpendicular to the plane):** helicoid for ``b < b_1`` = $(round(b_1, digits=4)), skyrmion crystal for ``b_1 < b < b_c`` = $(round(b_c, digits=4)), saturated ferromagnet above ``b_c``; the helicoid survives as a metastable state up to ``b_{\rm hel} = \pi^2/16`` = $(round(b_hel, digits=4)). The bulk cone lies below the skyrmion crystal at every field (the largest value of ``\rho_{\rm cone} - \rho_C`` over the scan is $(round(cone_margin, sigdigits=3)) < 0), which is the classical content of Aristov–Tsypilnikov Fig. 9: in a bulk cubic helimagnet the skyrmion crystal is not the mean-field ground state, and fluctuation corrections, thermal or quantum, are needed to stabilise it.
"""

# ╔═╡ 7ac1725b-194a-4ef1-b1ae-fbb3f688f185
md"""
## 5. Edge of the classical skyrmion: linearised tail and the disk boundary

Far from the core Pires linearises in the in-plane components (Eqs. (185)–(197)): with ``n_x + i n_y = n_p(r)e^{i\Phi}`` the equation ``\nabla^2 n_p - n_p/r^2 - n_p/\delta^2 = 0`` is the modified Bessel equation of order one, ``n_p = C_1 I_1(r/\delta) + C_2 K_1(r/\delta)``, with ``\delta = \sqrt{C/B} = 1/\sqrt b`` in the present units. For the isolated skyrmion only ``K_1`` survives, so the texture decays as ``e^{-r/\delta}/\sqrt r`` and two skyrmions repel with ``E_{\rm int}\propto e^{-d/\delta}/\sqrt d`` (Eq. (203)); the DMI term does not enter at linear order, as Pires remarks. On the finite disk of the skyrmion-crystal model the edge condition ``\beta(R) = 0`` admixes the growing solution, ``\beta \propto K_1(r/\delta) - [K_1(R/\delta)/I_1(R/\delta)]\,I_1(r/\delta)``, which is the classical edge state of the cell. Both forms are matched to the numerical profiles at a single radius and compared with them further out.
"""

# ╔═╡ 925d3d40-90eb-4b19-85f2-58e699e06da3
let
	b = b_ref; k = sqrt(b)                                            # k = 1/δ
	iso_sol = isolated_skyrmion(b)                                    # R = R_iso ≫ δ: isolated skyrmion
	cell = optimal_radius(b)                                          # skyrmion-crystal cell, R₀ ≈ 4
	r12, β12, _ = solve_profile(b, 12.0; N=6000)                      # intermediate disk: edge admixture visible
	tail_iso(r) = besselk(1, k*r)
	tail_disk(r, R) = besselk(1, k*r) - besselk(1, k*R)/besseli(1, k*R)*besseli(1, k*r)
	asym(r) = sqrt(π/(2k*r))*exp(-k*r)                                # Pires Eq. (197)
	fitamp(r, β, model, r1) = (i = findfirst(>=(r1), r); β[i]/model(r[i]))
	A_iso = fitamp(iso_sol.r, iso_sol.β, tail_iso, 6.0)
	A_12  = fitamp(r12, β12, r -> tail_disk(r, 12.0), 6.0)
	A_12K = fitamp(r12, β12, tail_iso, 6.0)
	A_0   = fitamp(cell.r, cell.β, r -> tail_disk(r, cell.R0), 3.0)
	p = plot(xlabel="r", ylabel="β(r)", yscale=:log10, ylims=(1e-7, 4), xlims=(0, 16), framestyle=:box, legend=:bottomleft,
	         title="Tail of β(r) at b = $b: K₁(r/δ) and the edge admixture of I₁ on a disk", titlefontsize=10, size=(820, 460))
	plot!(p, iso_sol.r, max.(iso_sol.β, 1e-12), lw=2.5, label="numerical, isolated (R = $(R_iso))")
	plot!(p, r12, max.(β12, 1e-12), lw=2.5, label="numerical, disk R = 12")
	plot!(p, cell.r, max.(cell.β, 1e-12), lw=2.5, label="numerical, optimal cell R₀ = $(round(cell.R0, digits=3))")
	rr = range(2.0, 16.0, length=300)
	plot!(p, rr, A_iso*tail_iso.(rr), ls=:dash, lw=2, color=:black, label="A K₁(r/δ), matched at r = 6")
	plot!(p, rr, A_iso*tail_iso(6.0)/asym(6.0)*asym.(rr), ls=:dot, lw=2, color=:gray, label="asymptotic √(πδ/2r) e^{−r/δ}, Eq. (197)")
	rr12 = range(2.0, 12.0, length=300)
	plot!(p, rr12, max.(A_12*tail_disk.(rr12, 12.0), 1e-12), ls=:dash, lw=2, color=:red, label="A [K₁ − (K₁/I₁)(R) I₁], R = 12, matched at r = 6")
	rr0 = range(1.5, cell.R0, length=200)
	plot!(p, rr0, max.(A_0*tail_disk.(rr0, cell.R0), 1e-12), ls=:dash, lw=2, color=:green, label="same with R = R₀, matched at r = 3")
	resid(r, β, model, A, r1, r2) = (idx = findall(x -> r1 <= x <= r2, r); maximum(abs(β[i] - A*model(r[i]))/β[i] for i in idx))
	tbl = md_table(["Profile and matched linear edge solution", "max relative deviation on the test interval"], [
		("isolated: A K₁(r/δ) on 6 ≤ r ≤ 14", resid(iso_sol.r, iso_sol.β, tail_iso, A_iso, 6.0, 14.0)),
		("disk R = 12: A K₁ only (no edge term) on 6 ≤ r ≤ 11.5", resid(r12, β12, tail_iso, A_12K, 6.0, 11.5)),
		("disk R = 12: A [K₁ − (K₁/I₁)(R) I₁] on 6 ≤ r ≤ 11.5", resid(r12, β12, r -> tail_disk(r, 12.0), A_12, 6.0, 11.5)),
		("optimal cell R₀: A [K₁ − (K₁/I₁)(R₀) I₁] on 3 ≤ r ≤ 0.98 R₀", resid(cell.r, cell.β, r -> tail_disk(r, cell.R0), A_0, 3.0, 0.98cell.R0)),
	]; sigdigits=3)
	md"""
	$(p)
	$(tbl)
	"""
end

# ╔═╡ 797975cc-adb0-4820-bc52-382f712bea11
md"""
## 6. Verification and scope
"""

# ╔═╡ c4efa4ea-ef42-4b05-aa63-0e18bda43931
let
	checks = [
		("Belavin–Polyakov charge on the Cartesian grid within 1% of −1 (L = 40, R = 2)", abs(topological_charge(r -> ΘBP(r; R=2.0); L=40.0, h=0.1) + 1) < 0.01),
		("Exchange energy of the Belavin–Polyakov texture equals 4π to 1e-6", abs(π*quadgk(r -> r*(dΘBP(r)^2 + sin(ΘBP(r))^2/r^2), 0, Inf)[1] - 4π) < 1e-6),
		("Reference disk: Newton converged, monotone profile, exact edge value", ref.converged && ref.β[end] == 0.0),
		("Reference disk density agrees with the Mathematica notebook to 1e-6", abs(ref.ρ + 0.0691244243113868) < 1e-6),
		("Optimal radii agree with the Mathematica notebook to 1e-3 at all benchmark fields", maximum(abs.([o.R0 for o in optima] .- bench_R0)) < 1e-3),
		("Topological charge of the reference disk is −1 to 1e-3", abs(ref_Q + 1) < 1e-3),
		("Isolated skyrmion energy changes sign in 0.75 < b < 0.85", 0.75 < b_c < 0.85),
		("Helicoid density is continuous at b_hel = π²/16", abs(helicoid(b_hel - 1e-4).ρ) < 1e-3),
		("Cone formula reproduced by brute-force minimisation at b = 0.6", abs(minimum(0.5q^2*sin(β)^2 - q*sin(β)^2 + 0.6(1 - cos(β)) for q in range(0, 2, length=401), β in range(0, π, length=721)) - ρ_cone(0.6)) < 1e-5),
		("Helicoid–skyrmion-crystal crossing lies in 0.1 < b₁ < 0.5", 0.1 < b_1 < 0.5),
		("Cone below the skyrmion crystal at every scanned field", cone_margin < 0),
	]
	md_table(["Independent numerical check", "Pass"], checks)
end

# ╔═╡ a6e80748-27d6-41f1-892a-508fa8fe66d7
md"""
**Coverage.** Topology and the Cartesian evaluation of the charge (Pires Eqs. (13), (41)–(44), Figs. 8–10); Belavin–Polyakov solution, Bogomolny bound and the stereographic/holomorphic description (Eqs. (39)–(49), (98)–(132), Appendix A); Bloch and Néel DMI densities (Eqs. (56)–(62)); the Euler–Lagrange profile with DMI and field (Eqs. (64)–(67)), its disk energy, optimal cell radius and the benchmark against Aristov–Tsypilnikov Eqs. (8)–(9) and Figs. 3–4; wall ansätze (Eq. (70), Appendix B) and the anisotropy scaling (Eqs. (71)–(74)); helix, helicoid, cone and ferromagnet versus the skyrmion crystal (Aristov–Tsypilnikov Appendix C, Fig. 9); the linearised tail and the edge admixture (Eqs. (185)–(197), (203)).

**Limits.** Everything here is classical and at zero temperature; the review's dynamical (LLG, Thiele), antiferromagnetic, three-dimensional (hopfion) and quantum sections are outside the scope of this notebook, as is the magnon spectrum of the skyrmion crystal (computed in the Mathematica notebook). The disk with a Dirichlet edge is the circular-cell approximation of the hexagonal lattice. The finite-difference profile is second-order accurate (relative energy error about 5e-7 at N = 2000; Richardson extrapolation of N = 2000 and 4000 reproduces the 24-digit shooting value to 1e-9). The Cartesian charge on a finite box misses the algebraic tail of the Belavin–Polyakov texture, so that check is only at the 1% level. Printed-text points noted while reproducing: the review lists "m = −1" for Néel skyrmions, whereas with Eq. (44) and Fig. 10 both Bloch and Néel skyrmions have m = 1 and Q = −1, and m = −1 is the antiskyrmion with Q = +1; Eq. (98) projects from the north pole while Eq. (121) uses the south-pole form, which is the one used here.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
Plots = "~1.41.7"
PlutoUI = "~0.7.83"
QuadGK = "~2.11.3"
SpecialFunctions = "~2.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "3e2aab0cc6e2376c55af156f02cfcfc5a92c0a9d"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "c0767b264422f9b301a3bdf70513ff3c733f59f7"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.10.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "0392e5e3c5327b90a487b0ff3e818c4aecd3660b"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "704274ca960c42049fd9bda089fc4be32af5cdca"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.5+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─a720a99a-c873-4990-bdb6-04e21b07dd07
# ╠═d4b939fa-906a-4e36-9f80-5457de4804a1
# ╟─a2643675-8b2f-4ae9-96d5-3584e717cd3d
# ╠═2b0aaeaa-caed-4d8f-9a32-0481ffc525cd
# ╟─278d6a7f-b131-453a-a863-d9b9959d5217
# ╠═95103ea8-29d7-4e98-863e-0c40752a4d29
# ╠═74b89641-18a6-4c78-8b78-00c3006e2edd
# ╠═d565a9da-bddb-4ab4-9010-41bc74981e1b
# ╟─68ba4c8c-a9c9-4fd5-8359-1852aaf8e89a
# ╠═6d334429-2322-4cdd-b979-f3693a452537
# ╠═5dfcea4c-ecaf-4bb3-9c52-f92a1e592e28
# ╠═f0a0ce6e-dc8c-4893-ba57-4fe6be9ceec0
# ╠═fd1993a3-cf33-453e-8374-04c1ca9b9e9f
# ╠═26da870f-0750-4845-af36-eee13bd4bcff
# ╠═f1c60c82-6729-44ce-98ca-cc7d65c25452
# ╟─4e7cf273-2d05-46d2-864c-0bc806b5bc20
# ╠═63314e78-de74-4716-9385-0c41cbee06a4
# ╠═59d8bd9a-1dad-4777-8366-d8bb3b245045
# ╠═36d75ef1-71d4-4916-afe7-fd4e5b43c99a
# ╟─e87d0f6b-9e17-4d9b-a20e-31d13f222a09
# ╠═a5ca2db0-455a-48a8-8a4e-8996118f3471
# ╠═4a116270-e675-48f4-987e-a3dfd8cdf170
# ╠═ee899d84-a831-4698-92a3-0a7cb88a581c
# ╠═c0f6824b-63a0-4260-baff-f49d0b3d575e
# ╟─e5078a52-d62d-475b-97fc-c999a987d023
# ╠═8039c401-aa49-4aa0-8155-3d09910bc9bf
# ╠═e84aa37a-b4db-41f9-b864-ce8912d40ce2
# ╠═a0bcd687-3ffa-4d47-8aac-ca84516a8d7c
# ╠═cb022040-e601-4744-bb60-524fab3550f8
# ╠═9fc4534b-2d68-4d4d-aae7-ce59fa3141c8
# ╟─baf38015-0a10-4ba0-8eb1-80507d8ce01d
# ╠═3220c423-3efb-4068-8d81-ec77a0e0aab5
# ╠═23985e1d-e274-4571-93b2-e26034aafe73
# ╟─78e1b275-e3de-4316-9a38-df38063de737
# ╠═39234454-4051-47c0-8566-6d2deb613c95
# ╟─e96b437e-2fee-4b2e-baa7-f3269b149dda
# ╠═34a20422-f071-46cd-8317-0b261a024de2
# ╟─1acd803c-d165-4ef8-8fe1-8e1c23d9ff8c
# ╠═b36b11f0-3729-437a-8113-f7490a689648
# ╠═3cf1d5aa-a811-41ba-9408-9d464c93ed0b
# ╠═3699a233-39e9-4d65-ac70-9eb77df79db9
# ╠═5217da5a-2cb8-46bd-b613-10798b37feba
# ╠═82f43975-5c2e-4420-bbae-b3c8546d6eb9
# ╟─d832c3cd-9d0d-4759-b625-30c502412e42
# ╟─7ac1725b-194a-4ef1-b1ae-fbb3f688f185
# ╠═925d3d40-90eb-4b19-85f2-58e699e06da3
# ╟─797975cc-adb0-4820-bc52-382f712bea11
# ╠═c4efa4ea-ef42-4b05-aa63-0e18bda43931
# ╟─a6e80748-27d6-41f1-892a-508fa8fe66d7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
