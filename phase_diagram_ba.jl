#!/usr/bin/env julia
#=
phase_diagram_ba.jl — classical energies and (b, a) ground-state phase diagram

Full documentation: README.md

Summary
-------
Compares Ferromagnet / Helix / Helicoid / Cone / Skyrmion energy densities at
J = D = 1, builds a parallel (b, a) phase diagram, writes outputs/phase_diagram_ba.{png,csv}.

  b = B C / D² ,   a = K_z C / D² ,   C = J
  Skyrmion: fixed disk radius R (default 10); ρ_C = E/(π R²), no min over R.

Quick start (from repository root)
----------------------------------
  julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
  julia --project=scripts -t auto phase_diagram_ba.jl
  julia --project=scripts -t auto phase_diagram_ba.jl --diagram-only --step 0.05 --R 10
  julia --project=scripts phase_diagram_ba.jl --help
=#

using LinearAlgebra
using Base.Threads
using Printf
using Dates
using Plots

# Avoid BLAS oversubscription when many Julia threads each run dense linear algebra.
BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
function parse_args(args)
	opts = Dict{String,Any}(
		"b" => 0.6,
		"a" => 0.2,
		"R" => 10.0,            # fixed skyrmion disk radius
		"nb" => 21,
		"na" => 17,
		"db" => nothing,        # step in b; if set, overrides --nb
		"da" => nothing,        # step in a; if set, overrides --na
		"bmin" => 0.0,
		"bmax" => 1.0,
		"amin" => 0.0,
		"amax" => 0.8,
		"N" => 1200,
		"diagram_only" => false,
		"outdir" => joinpath(@__DIR__, "outputs"),
	)
	i = 1
	while i ≤ length(args)
		arg = args[i]
		if arg == "--diagram-only"
			opts["diagram_only"] = true
		elseif arg == "--b" && i < length(args)
			opts["b"] = parse(Float64, args[i+=1])
		elseif arg == "--a" && i < length(args)
			opts["a"] = parse(Float64, args[i+=1])
		elseif arg in ("--R", "--rmin") && i < length(args)
			# --rmin kept as alias for older command lines
			opts["R"] = parse(Float64, args[i+=1])
		elseif arg == "--rmax"
			# ignored: no R-window minimisation anymore
			i < length(args) && (i += 1)
			@warn "--rmax is ignored; skyrmion uses fixed --R"
		elseif arg == "--nb" && i < length(args)
			opts["nb"] = parse(Int, args[i+=1])
		elseif arg == "--na" && i < length(args)
			opts["na"] = parse(Int, args[i+=1])
		elseif arg in ("--db", "--bstep", "--step-b") && i < length(args)
			opts["db"] = parse(Float64, args[i+=1])
		elseif arg in ("--da", "--astep", "--step-a") && i < length(args)
			opts["da"] = parse(Float64, args[i+=1])
		elseif arg in ("--step", "--d") && i < length(args)
			# same step for both axes
			s = parse(Float64, args[i+=1])
			opts["db"] = s
			opts["da"] = s
		elseif arg == "--bmin" && i < length(args)
			opts["bmin"] = parse(Float64, args[i+=1])
		elseif arg == "--bmax" && i < length(args)
			opts["bmax"] = parse(Float64, args[i+=1])
		elseif arg == "--amin" && i < length(args)
			opts["amin"] = parse(Float64, args[i+=1])
		elseif arg == "--amax" && i < length(args)
			opts["amax"] = parse(Float64, args[i+=1])
		elseif arg == "--N" && i < length(args)
			opts["N"] = parse(Int, args[i+=1])
		elseif arg == "--outdir" && i < length(args)
			opts["outdir"] = args[i+=1]
		elseif arg in ("-h", "--help")
			println("""
			phase_diagram_ba.jl — classical energies and (b, a) phase diagram

			Docs: README.md

			USAGE
			  julia --project=scripts -t auto phase_diagram_ba.jl [options]
			  julia --project=scripts phase_diagram_ba.jl --help

			Use -t auto or -t N so each (b, a) grid point runs on its own thread.

			WHAT IT DOES
			  1) Optional table: energy densities at one (b, a) for
			     Ferromagnet, Helix, Helicoid, Cone, Skyrmion (fixed disk R).
			  2) Phase diagram on a (b, a) grid → outputs/phase_diagram_ba.{png,csv}
			     Plot: a on x-axis, b on y-axis.

			UNITS (always J = D = 1, C = J)
			  b = B C / D² ,  a = K_z C / D²
			  Densities ρ in units D²/J, relative to the ferromagnet (ρ_FM = 0).
			  Skyrmion: ρ_C = E/(π R²) at fixed R (default 10); no minimisation over R.

			OPTIONS
			  Single-point comparison
			    --b VAL           field b for the table          (default 0.6)
			    --a VAL           anisotropy a for the table     (default 0.2)
			    --diagram-only    skip the table; only the diagram

			  Skyrmion disk
			    --R VAL           fixed disk radius              (default 10)
			    --N N             radial grid for EL solver      (default 1200)

			  Diagram window
			    --bmin/--bmax     range of b                     (default 0 … 1)
			    --amin/--amax     range of a                     (default 0 … 0.8)

			  Grid resolution (steps override point counts)
			    --db / --bstep    step in b                      (overrides --nb)
			    --da / --astep    step in a                      (overrides --na)
			    --step VAL        same step for b and a
			    --nb N / --na N   number of points if no step    (default 21 × 17)

			  Output
			    --outdir DIR      PNG/CSV directory              (default ./outputs)

			EXAMPLES
			  julia --project=scripts -t auto phase_diagram_ba.jl
			  julia --project=scripts -t auto phase_diagram_ba.jl --diagram-only --step 0.05
			  julia --project=scripts -t 8 phase_diagram_ba.jl --db 0.05 --da 0.04 --R 10
			  julia --project=scripts -t auto phase_diagram_ba.jl --b 0.6 --a 0.2 --nb 15 --na 11
			""")
			exit(0)
		else
			error("unknown argument: $arg (use --help)")
		end
		i += 1
	end
	opts["R"] > 0 || error("--R must be positive (got $(opts["R"]))")
	opts["db"] === nothing || opts["db"] > 0 || error("--db must be positive")
	opts["da"] === nothing || opts["da"] > 0 || error("--da must be positive")
	return opts
end

"""
    axis_grid(lo, hi; n=nothing, step=nothing) -> Vector{Float64}

Build a 1D scan from `lo` to `hi`. Prefer `step` when given (`lo:step:hi`, with `hi`
appended if the colon range misses it); otherwise use `range(lo, hi; length=n)`.
"""
function axis_grid(lo, hi; n::Union{Nothing,Int}=nothing, step::Union{Nothing,Float64}=nothing)
	lo ≤ hi || error("axis range requires lo ≤ hi (got $lo > $hi)")
	if step !== nothing
		step > 0 || error("step must be positive")
		xs = collect(lo:step:hi)
		isempty(xs) && return [lo, hi]
		# include the upper endpoint if floating-point stepping skipped it
		if xs[end] < hi - 1e-9 * max(1.0, abs(hi))
			push!(xs, hi)
		end
		return xs
	end
	n === nothing && error("axis_grid needs n or step")
	n ≥ 2 || error("need at least 2 grid points")
	return collect(range(lo, hi; length=n))
end

const OPTS = parse_args(ARGS)

# ---------------------------------------------------------------------------
# Euler–Lagrange skyrmion on a disk
# ---------------------------------------------------------------------------
function el_residual!(F, β, r, h, b, a)
	N = length(r) - 1
	@inbounds for i in 1:N-1
		ri = r[i+1]; bi = β[i+1]; bm = β[i]; bp = β[i+2]
		F[i] = (bp - 2bi + bm)/h^2 + (bp - bm)/(2h*ri) - sin(bi)*cos(bi)/ri^2 +
		       2sin(bi)^2/ri - b*sin(bi) - a*sin(2bi)
	end
	F
end

wall_guess(r, Rw, w) = [ri == 0 ? float(π) : 2atan(sinh(Rw/w)/sinh(ri/w)) for ri in r]

function newton_profile(b, R, β0; a=0.0, maxiter=200, tol=1e-11)
	N = length(β0) - 1; h = R/N; r = range(0, R, length=N+1)
	β = copy(β0); β[1] = π; β[end] = 0.0
	n = N - 1
	F = zeros(n); Ft = zeros(n); dl = zeros(n-1); d = zeros(n); du = zeros(n-1); βt = similar(β)
	el_residual!(F, β, r, h, b, a)
	for _ in 1:maxiter
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

function solve_profile(b, R; a=0.0, N=1200)
	r = range(0, R, length=N+1)
	for (Rw, w) in ((min(R/2, 2.0), 1.0), (R/2, 1.0), (min(R/2, 3.0), 0.7),
	                (min(R/3, 1.5), 1.0), (0.5, 0.5), (0.25, 0.25))
		β, ok = newton_profile(b, R, wall_guess(r, Rw, w); a)
		if ok && all(-1e-9 .<= β .<= π + 1e-9) && all(diff(β) .<= 1e-7)
			return collect(r), β, true
		end
	end
	collect(r), wall_guess(r, R/2, 1.0), false
end

function energy_density(r, β, b; a=0.0)
	N = length(r) - 1; h = r[2] - r[1]; R = r[end]
	E = 0.0
	@inbounds for i in 1:N
		rm = (r[i] + r[i+1])/2; βm = (β[i] + β[i+1])/2; dβ = (β[i+1] - β[i])/h
		E += rm*(0.5dβ^2 + sin(βm)^2/(2rm^2) + dβ + sin(2βm)/(2rm) +
		         b*(1 - cos(βm)) + a*sin(βm)^2)*h
	end
	E *= 2π
	E/(π*R^2), E
end

function golden(f, lo, hi; tol=1e-5)
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

"""
    skyrmion_fixed_R(b, a; R=10, N=1200) -> (ρ, E, R, converged)

Skyrmion on a disk of fixed radius R (default 10). No minimisation over R.
"""
function skyrmion_fixed_R(b, a; R=10.0, N=1200)
	r, β, ok = solve_profile(b, R; a, N)
	ρ, E = energy_density(r, β, b; a)
	(ρ=ρ, E=E, R=R, converged=ok)
end

# ---------------------------------------------------------------------------
# Competing extended states (energy density relative to FM)
# ---------------------------------------------------------------------------
ρ_FM() = 0.0
ρ_helix(b, a) = -0.5 + b + a/2

function cone_state(b, a)
	f(β) = (a - 0.5)*sin(β)^2 + b*(1 - cos(β))   # at optimal q = 1
	βbest = 0.0; ρbest = 0.0
	if abs(1 - 2a) > 1e-14
		c = b / (1 - 2a)
		if abs(c) ≤ 1 + 1e-14
			βc = acos(clamp(c, -1.0, 1.0))
			fc = f(βc)
			fc < ρbest && (ρbest = fc; βbest = βc)
		end
	end
	for β in range(0, π; length=721)
		fc = f(β)
		fc < ρbest && (ρbest = fc; βbest = β)
	end
	q = abs(sin(βbest)) < 1e-12 ? 0.0 : 1.0
	(ρ=ρbest, β=βbest, q=q)
end

function helicoid_density(b, a, c)
	V(θ) = 2b*sin(θ/2)^2 + a*sin(θ)^2 + c
	vmin = minimum(V, range(0, 2π; length=2001))
	vmin ≤ 1e-14 && return Inf, Inf
	spd(θ) = sqrt(2V(θ))
	θs = range(0, 2π; length=4001)
	dθ = θs[2] - θs[1]
	sp = spd.(θs)
	I = sum(sp[1:end-1] .+ sp[2:end]) * dθ / 2
	inv = 1.0 ./ sp
	L = sum(inv[1:end-1] .+ inv[2:end]) * dθ / 2
	(I - 2π)/L - c, L
end

function helicoid_state(b, a)
	best = (ρ=Inf, L=Inf, c=NaN)
	for u in range(-12, 2; length=90)
		ρ, L = helicoid_density(b, a, 10.0^u)
		ρ < best.ρ && (best = (ρ=ρ, L=L, c=10.0^u))
	end
	if isfinite(best.ρ) && best.ρ < -1e-12
		u0 = log10(best.c)
		u = golden(u -> helicoid_density(b, a, 10.0^u)[1], u0 - 0.6, u0 + 0.6; tol=1e-5)
		ρ, L = helicoid_density(b, a, 10.0^u)
		return (ρ=ρ, L=L, c=10.0^u)
	end
	(ρ=0.0, L=Inf, c=0.0)
end

# ---------------------------------------------------------------------------
# Ground-state selection
# ---------------------------------------------------------------------------
"""
    compare_states(b, a; R=10, N=1200)

Densities in units D²/J relative to the ferromagnet, and the lowest-energy state.
Skyrmion uses a fixed disk radius R (default 10). When several densities agree
within 1e-8 (e.g. cone/helicoid collapsed to FM), Ferromagnet is preferred.
"""
function compare_states(b, a; R=10.0, N=1200)
	sk = skyrmion_fixed_R(b, a; R, N)
	cone = cone_state(b, a)
	hel = helicoid_state(b, a)
	candidates = (
		("Ferromagnet", ρ_FM()),
		("Helix", ρ_helix(b, a)),
		("Helicoid", hel.ρ),
		("Cone", cone.ρ),
		("Skyrmion", sk.ρ),
	)
	ρmin = minimum(last, candidates)
	tol = 1e-8
	# preference among numerical ties at the minimum
	prefer = ("Ferromagnet", "Helicoid", "Cone", "Helix", "Skyrmion")
	winner = "Skyrmion"
	for name in prefer
		ρ = last(candidates[findfirst(c -> first(c) == name, candidates)])
		if ρ ≤ ρmin + tol
			winner = name
			break
		end
	end
	ρmap = Dict(name => ρ for (name, ρ) in candidates)
	(; b, a,
	 Ferromagnet=ρmap["Ferromagnet"], Helix=ρmap["Helix"], Helicoid=ρmap["Helicoid"],
	 Cone=ρmap["Cone"], Skyrmion=ρmap["Skyrmion"],
	 sk_R=sk.R, sk_E=sk.E, sk_ok=sk.converged,
	 cone_β=cone.β, hel_L=hel.L, ground=winner, ρmin)
end

function print_comparison(res)
	println()
	println("="^64)
	@printf("Classical energies at b = %.4f, a = %.4f  (J = D = 1)\n", res.b, res.a)
	@printf("Skyrmion disk: fixed R = %.4f, E = %.6f, converged = %s\n", res.sk_R, res.sk_E, res.sk_ok)
	println("-"^64)
	@printf("%-14s %14s\n", "State", "ρ (D²/J)")
	println("-"^64)
	for (name, ρ) in (
		("Ferromagnet", res.Ferromagnet),
		("Helix", res.Helix),
		("Helicoid", res.Helicoid),
		("Cone", res.Cone),
		("Skyrmion", res.Skyrmion),
	)
		mark = name == res.ground ? "  <-- lowest" : ""
		@printf("%-14s %14.8f%s\n", name, ρ, mark)
	end
	println("-"^64)
	println("Ground state: ", res.ground, "   ρ_min = ", res.ρmin)
	println("="^64)
end

# ---------------------------------------------------------------------------
# Phase diagram on (b, a)
# ---------------------------------------------------------------------------
function phase_diagram(; bmin=0.0, bmax=1.0, amin=0.0, amax=0.8,
                       nb=21, na=17, db=nothing, da=nothing, R=10.0, N=1200,
                       bs=nothing, as_=nothing)
	bs = bs === nothing ? axis_grid(bmin, bmax; n=nb, step=db) : collect(bs)
	as_ = as_ === nothing ? axis_grid(amin, amax; n=na, step=da) : collect(as_)
	nb = length(bs); na = length(as_)
	id_of = Dict("Ferromagnet"=>1, "Helix"=>2, "Helicoid"=>3, "Cone"=>4, "Skyrmion"=>5)
	grid = zeros(Int, na, nb)
	ρgrid = fill(NaN, na, nb)
	total = nb * na
	done = Atomic{Int}(0)
	t0 = time()
	nthreads = Threads.nthreads()
	db_eff = nb > 1 ? (bs[end] - bs[1]) / (nb - 1) : 0.0
	da_eff = na > 1 ? (as_[end] - as_[1]) / (na - 1) : 0.0
	@printf("  grid: b = %.4g … %.4g  (%d pts, Δb ≈ %.4g)%s\n",
		bs[1], bs[end], nb, db_eff, db === nothing ? "" : "  [--db]")
	@printf("         a = %.4g … %.4g  (%d pts, Δa ≈ %.4g)%s\n",
		as_[1], as_[end], na, da_eff, da === nothing ? "" : "  [--da]")
	@printf("  parallel phase diagram on %d threads (%d grid points, fixed R = %.4f)\n",
		nthreads, total, R)
	if nthreads == 1
		println("  hint: restart with  julia -t auto --project=scripts phase_diagram_ba.jl")
	end

	# Flattened index k → (i, j) with i over a, j over b; each point is independent.
	Threads.@threads for k in 1:total
		j = (k - 1) ÷ na + 1
		i = (k - 1) % na + 1
		res = compare_states(bs[j], as_[i]; R, N)
		grid[i, j] = id_of[res.ground]
		ρgrid[i, j] = res.ρmin
		ndone = atomic_add!(done, 1) + 1
		if ndone % max(1, total ÷ 20) == 0 || ndone == total
			@printf("\r  phase diagram: %d / %d  (%.1f s, %d threads)   ",
				ndone, total, time() - t0, nthreads)
			flush(stdout)
		end
	end
	println()
	(; bs, as_=as_, grid, ρgrid, elapsed=time() - t0, nthreads, R, db=db_eff, da=da_eff)
end

function save_phase_plot(diag, outpath; R)
	labels = ["Ferromagnet", "Helix", "Helicoid", "Cone", "Skyrmion"]
	colors = [
		RGB(0.85, 0.85, 0.85),
		RGB(0.30, 0.55, 0.85),
		RGB(0.20, 0.65, 0.40),
		RGB(0.95, 0.65, 0.20),
		RGB(0.85, 0.20, 0.25),
	]
	cmap = cgrad(colors, 5; categorical=true)
	# axes: a horizontal, b vertical (grid rows = a, cols = b → transpose for heatmap)
	p = heatmap(diag.as_, diag.bs, transpose(diag.grid);
		xlabel="a = K_z C / D²",
		ylabel="b = B C / D²",
		title="Classical ground state (J = D = 1, skyrmion R = $(round(R; digits=2)))",
		titlefontsize=11,
		color=cmap,
		clims=(0.5, 5.5),
		colorbar=:none,
		framestyle=:box,
		size=(860, 560),
		dpi=160,
		# GR under-estimates the room needed for the rotated y-label, so without an
		# explicit margin "b = B C / D²" is clipped at the left edge of the PNG.
		left_margin=6Plots.mm,
		bottom_margin=3Plots.mm,
		legend=:outertopright)
	for (k, lab) in enumerate(labels)
		# only show states that appear on the grid
		count(==(k), diag.grid) == 0 && continue
		scatter!(p, [NaN], [NaN]; color=colors[k], label=lab,
			markershape=:rect, markersize=8, markerstrokewidth=0)
	end
	mkpath(dirname(outpath))
	savefig(p, outpath)
	println("Wrote ", outpath)
	return p
end

function write_summary_csv(diag, outpath)
	names = ("Ferromagnet", "Helix", "Helicoid", "Cone", "Skyrmion")
	open(outpath, "w") do io
		println(io, "b,a,ground_id,ground,rho_min")
		for (j, b) in enumerate(diag.bs)
			for (i, a) in enumerate(diag.as_)
				id = diag.grid[i, j]
				@printf(io, "%.6f,%.6f,%d,%s,%.10f\n", b, a, id, names[id], diag.ρgrid[i, j])
			end
		end
	end
	println("Wrote ", outpath)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
	J = 1.0; D = 1.0
	@printf("J = %.1f, D = %.1f  ⇒  C = J, lengths in J/D = %.1f\n", J, D, J/D)
	@printf("Skyrmion disk: fixed R = %.4f (no minimisation over R)\n", OPTS["R"])
	@printf("Radial grid N = %d\n", OPTS["N"])
	println("Docs: README.md")
	println("Started ", Dates.now())

	if !OPTS["diagram_only"]
		res = compare_states(OPTS["b"], OPTS["a"]; R=OPTS["R"], N=OPTS["N"])
		print_comparison(res)
	end

	println()
	if OPTS["db"] !== nothing || OPTS["da"] !== nothing
		@printf("Building phase diagram with steps: db = %s, da = %s\n",
			OPTS["db"] === nothing ? "(from --nb)" : string(OPTS["db"]),
			OPTS["da"] === nothing ? "(from --na)" : string(OPTS["da"]))
	else
		@printf("Building phase diagram: b ∈ [%.2f, %.2f] (%d pts), a ∈ [%.2f, %.2f] (%d pts)\n",
			OPTS["bmin"], OPTS["bmax"], OPTS["nb"], OPTS["amin"], OPTS["amax"], OPTS["na"])
	end
	@printf("Julia threads: %d\n", Threads.nthreads())
	diag = phase_diagram(;
		bmin=OPTS["bmin"], bmax=OPTS["bmax"],
		amin=OPTS["amin"], amax=OPTS["amax"],
		nb=OPTS["nb"], na=OPTS["na"],
		db=OPTS["db"], da=OPTS["da"],
		R=OPTS["R"], N=OPTS["N"])
	@printf("Phase diagram finished in %.1f s (%d threads)\n", diag.elapsed, diag.nthreads)

	names = ("Ferromagnet", "Helix", "Helicoid", "Cone", "Skyrmion")
	println("Ground-state counts on the grid:")
	for id in 1:5
		n = count(==(id), diag.grid)
		n > 0 && @printf("  %-12s %4d\n", names[id], n)
	end

	outdir = OPTS["outdir"]
	mkpath(outdir)
	png = joinpath(outdir, "phase_diagram_ba.png")
	csv = joinpath(outdir, "phase_diagram_ba.csv")
	write_summary_csv(diag, csv)
	save_phase_plot(diag, png; R=OPTS["R"])
	println("Done ", Dates.now())
end

main()
